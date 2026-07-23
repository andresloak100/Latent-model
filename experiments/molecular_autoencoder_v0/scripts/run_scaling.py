#!/usr/bin/env python3
"""Stage C (small): scaling sweep over the latent budget.

Trains and held-out-evaluates the autoencoder at several latent sizes and
writes a scaling table (reconstruction quality vs latent-float budget /
compression ratio). Each point reuses the resumable train.py / eval.py via a
generated config, so the sweep itself is restartable.

Usage:
    python scripts/run_scaling.py --base configs/stage_b_heldout.yaml --epochs 500
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

# (n_latent_tokens, latent_dim) -> latent floats
LATENT_POINTS = [(8, 4), (16, 8), (16, 16), (32, 16)]  # 32, 128, 256, 512 floats


def run(cmd):
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/stage_b_heldout.yaml")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scan_cfg_dir = ROOT / "configs" / "_scan"
    scan_cfg_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for (L, d) in LATENT_POINTS:
        cfg = ExperimentConfig.from_yaml(ROOT / args.base)
        cfg.name = f"scan_L{L}_d{d}"
        cfg.model.n_latent_tokens = L
        cfg.model.latent_dim = d
        cfg.train.epochs = args.epochs
        cfg.train.out_dir = f"outputs/scan/L{L}_d{d}"
        cfg_path = scan_cfg_dir / f"L{L}_d{d}.yaml"
        cfg.save(cfg_path)

        metrics_path = ROOT / cfg.train.out_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"[scan] L={L} d={d}: metrics exist, skipping train/eval")
        else:
            print(f"[scan] L={L} d={d} (latent floats={L*d})")
            run([sys.executable, "scripts/train.py", "--config", str(cfg_path)])
            run([sys.executable, "scripts/eval.py", "--config", str(cfg_path), "--n-examples", "0"])

        m = utils.load_json(metrics_path)
        lm = m["learned_model"]
        rows.append({
            "n_latent_tokens": L,
            "latent_dim": d,
            "latent_floats": L * d,
            "mean_compression_ratio": lm.get("compression_ratio"),
            "heldout_all_atom_rmsd": lm.get("all_atom_rmsd"),
            "heldout_backbone_rmsd": lm.get("backbone_rmsd"),
            "heldout_bond_length_error": lm.get("bond_length_error"),
            "heldout_contact_f1": lm.get("contact_f1"),
        })

    out_dir = ROOT / "outputs" / "scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.save_json({"rows": rows, "epochs": args.epochs}, out_dir / "scaling.json")

    lines = ["# Stage C (small): latent-budget scaling (held-out)", "",
             f"Epochs per point: {args.epochs}. Held-out val split.", "",
             "| latent tokens | latent dim | latent floats | mean compression | "
             "held-out all-atom RMSD (A) | held-out backbone RMSD (A) | bond err (A) | contact F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['n_latent_tokens']} | {r['latent_dim']} | {r['latent_floats']} | "
            f"{r['mean_compression_ratio']:.3g} | {r['heldout_all_atom_rmsd']:.3g} | "
            f"{r['heldout_backbone_rmsd']:.3g} | {r['heldout_bond_length_error']:.3g} | "
            f"{r['heldout_contact_f1']:.3g} |"
        )
    (out_dir / "scaling_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[scan] wrote {out_dir/'scaling_table.md'}")


if __name__ == "__main__":
    main()
