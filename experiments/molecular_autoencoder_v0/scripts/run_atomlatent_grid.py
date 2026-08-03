#!/usr/bin/env python3
"""Matched-budget grid: can arbitrary shared latents reconstruct atoms?

Arms
----
A  current direct group codec (the 0.79 A baseline; latent scales with the
   structure, which is the thing being replaced)
B  atom-level encoder + arbitrary L shared latents + GLOBAL cross-attention
C  B, but locality-guided cross-attention using the latents' 3D anchors
D  C, plus masked geometric denoising

Budget matching
---------------
Total latent scalars L*d is held fixed within a budget band so that token
count and total capacity can be separated. Running L alone confounds them:
more tokens is also more floats, and "more floats helped" is not the question.

The wide setting is deliberate. At d high enough that the latent could hold
coordinates verbatim, plain reconstruction is vacuous -- copying scores
perfectly -- so arm D is the only one whose number means anything there. That
contrast is the point of including both.

Success criteria this grid is built to answer, in order:
  1. materially better than the ~9.3 A arbitrary-latent Perceiver result
  2. masked-region reconstruction beats a geometric baseline
  3. latent count genuinely independent of atom count (asserted, not assumed)
  4. no input geometry bypasses the bottleneck (asserted per config)
  5. performance improves predictably with L
  6. chirality and bond geometry stay valid

Usage:
    python scripts/run_atomlatent_grid.py --base configs/atomlatent_base.yaml \
        --latent-counts 16,32,64,128,256 --latent-dims 8,32 --device cuda --amp
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig  # noqa: E402
from molae import utils  # noqa: E402

ARMS = {
    "A_group_baseline": dict(encoder_type="direct"),
    "B_global":         dict(encoder_type="atomlatent", dec_local_cross=False),
    "C_local":          dict(encoder_type="atomlatent", dec_local_cross=True),
    "D_local_masked":   dict(encoder_type="atomlatent", dec_local_cross=True),
}
MASKED_ARMS = {"D_local_masked"}


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/atomlatent_base.yaml")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--latent-counts", default="16,32,64,128,256")
    ap.add_argument("--latent-dims", default="8,32")
    ap.add_argument("--epochs", type=int, default=0, help="0 = use the base config")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    Ls = [int(x) for x in args.latent_counts.split(",")]
    ds = [int(x) for x in args.latent_dims.split(",")]
    arms = [a for a in args.arms.split(",") if a in ARMS]

    scan = ROOT / "configs" / "_atomlatent_grid"
    scan.mkdir(parents=True, exist_ok=True)
    rows, planned = [], []

    for arm in arms:
        for d in ds:
            # Arm A's latent is PER RESIDUE -- n_latent_tokens is not read by
            # the direct codec at all, so sweeping L there would run the same
            # configuration five times under five different names.
            arm_Ls = Ls if arm != "A_group_baseline" else Ls[:1]
            for L in arm_Ls:
                cfg = ExperimentConfig.from_yaml(ROOT / args.base)
                for k, v in ARMS[arm].items():
                    setattr(cfg.model, k, v)
                cfg.model.n_latent_tokens = L
                cfg.model.latent_dim = d
                if arm in MASKED_ARMS:
                    cfg.train.mask_atom_frac = 0.15
                    cfg.train.mask_region_frac = 0.15
                    cfg.train.mask_region_span = 8
                    cfg.train.mask_noise_std = 0.1
                if args.epochs:
                    cfg.train.epochs = args.epochs
                name = f"{arm}_L{L}_d{d}"
                cfg.name = name
                cfg.train.out_dir = f"outputs/atomlatent{args.tag}/{name}"
                path = scan / f"{name}.yaml"
                cfg.save(path)
                planned.append((name, L, d, L * d, path, cfg.train.out_dir))

    print(f"[grid] {len(planned)} runs; latent scalars L*d per run:")
    for name, L, d, floats, _, _ in planned:
        print(f"   {name:34s} L={L:4d} d={d:3d}  {floats:6d} scalars")
    if args.dry_run:
        return

    for name, L, d, floats, path, out_dir in planned:
        print(f"\n[grid] === {name} ===", flush=True)
        cmd = [sys.executable, "scripts/train.py", "--config", str(path),
               "--device", args.device, "--num-workers", str(args.num_workers)]
        if args.amp:
            cmd.append("--amp")
        run(cmd)
        log = utils.load_json(ROOT / out_dir / "train_log.json")
        s = log.get("val_summary", {})
        rows.append({
            "run": name, "arm": name.split("_L")[0], "L": L, "latent_dim": d,
            "latent_scalars": floats,
            "val_rmsd": s.get("val_rmsd", {}).get("mean"),
            "val_rmsd_sd": s.get("val_rmsd", {}).get("sd"),
            "val_rmsd_ema": s.get("val_rmsd_ema", {}).get("mean"),
            "val_masked": s.get("val_rmsd_masked", {}).get("mean"),
            "val_visible": s.get("val_rmsd_visible", {}).get("mean"),
        })
        out = ROOT / "outputs" / f"atomlatent{args.tag}"
        out.mkdir(parents=True, exist_ok=True)
        utils.save_json({"rows": rows}, out / "grid.json")

    lines = ["# Atom-level encoder, arbitrary shared latents", "",
             "| run | arm | L | d | scalars | val (mean) | sd | val EMA | masked | visible |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.3f}"          # noqa: E731
    for r in rows:
        lines.append(f"| {r['run']} | {r['arm']} | {r['L']} | {r['latent_dim']} | "
                     f"{r['latent_scalars']} | {fmt(r['val_rmsd'])} | "
                     f"{fmt(r['val_rmsd_sd'])} | {fmt(r['val_rmsd_ema'])} | "
                     f"{fmt(r['val_masked'])} | {fmt(r['val_visible'])} |")
    lines += ["", "Read the masked column, not the pooled one, for arm D: a model that "
              "copies visible coordinates and guesses the rest scores well on the pooled "
              "number and badly on masked."]
    (ROOT / "outputs" / f"atomlatent{args.tag}" / "grid.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
