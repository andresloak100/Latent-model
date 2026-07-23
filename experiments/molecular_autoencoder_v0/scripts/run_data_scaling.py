#!/usr/bin/env python3
"""Data-scale sweep: does held-out reconstruction improve with more structures?

Trains the SAME model at a fixed compute budget on increasing numbers of
training structures and evaluates each on the SAME held-out val split. This is
the central experiment the milestone pointed to: the tiny-data run showed the
codec memorises; this measures whether the generalisation gap closes as data
grows (the premise behind pretraining before latent diffusion).

Compute is held fixed in *optimizer steps* (not epochs) so small-N runs are not
under-trained nor large-N over-trained.

Usage (on a GPU box, after fetch + prepare of a few-thousand-structure set):
    python scripts/run_data_scaling.py --base configs/stage_b_heldout.yaml \
        --sizes 100,300,1000,3000 --total-steps 60000 --device cuda --amp
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig  # noqa: E402
from molae import utils  # noqa: E402


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/stage_b_heldout.yaml")
    ap.add_argument("--sizes", default="100,300,1000,3000")
    ap.add_argument("--total-steps", type=int, default=60000)
    ap.add_argument("--latent-tokens", type=int, default=16)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = ExperimentConfig.from_yaml(ROOT / args.base)
    master = utils.load_json(ROOT / base.data.splits_file)
    train_pool = list(master["train"])
    val_keys = list(master["val"])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_pool)
    sizes = [int(s) for s in args.sizes.split(",")]

    scan_dir = ROOT / "configs" / "_data_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for n in sizes:
        n = min(n, len(train_pool))
        subset = sorted(train_pool[:n])
        splits_path = ROOT / "data" / f"splits_n{n}.json"
        utils.save_json({"train": subset, "val": val_keys, "note": f"data-scale n={n}"},
                        splits_path)

        steps_per_epoch = max(1, math.ceil(n / base.train.batch_size))
        epochs = max(1, math.ceil(args.total_steps / steps_per_epoch))

        cfg = ExperimentConfig.from_yaml(ROOT / args.base)
        cfg.name = f"data_n{n}"
        cfg.data.splits_file = str(splits_path.relative_to(ROOT))
        cfg.model.n_latent_tokens = args.latent_tokens
        cfg.model.latent_dim = args.latent_dim
        cfg.train.epochs = epochs
        cfg.train.overfit = False
        cfg.train.out_dir = f"outputs/data_scan/n{n}"
        cfg.train.log_every = max(1, epochs // 20)
        cfg.train.ckpt_every = max(1, epochs // 5)
        cfg_path = scan_dir / f"n{n}.yaml"
        cfg.save(cfg_path)

        print(f"\n[data-scan] n={n} train, {len(val_keys)} val, "
              f"{epochs} epochs (~{args.total_steps} steps), latent={args.latent_tokens*args.latent_dim}f")
        train_cmd = [sys.executable, "scripts/train.py", "--config", str(cfg_path),
                     "--device", args.device, "--num-workers", str(args.num_workers)]
        if args.amp:
            train_cmd.append("--amp")
        run(train_cmd)
        run([sys.executable, "scripts/eval.py", "--config", str(cfg_path),
             "--device", args.device, "--n-examples", "0"])

        m = utils.load_json(ROOT / cfg.train.out_dir / "metrics.json")
        lm = m["learned_model"]
        rows.append({
            "n_train": n,
            "epochs": epochs,
            "latent_floats": args.latent_tokens * args.latent_dim,
            "heldout_all_atom_rmsd": lm["all_atom_rmsd"],
            "heldout_backbone_rmsd": lm["backbone_rmsd"],
            "heldout_contact_f1": lm["contact_f1"],
            "centroid_baseline_rmsd": m["trivial_all_atom_baselines"]["centroid_all_atom_rmsd"],
        })

    out = ROOT / "outputs" / "data_scan"
    out.mkdir(parents=True, exist_ok=True)
    utils.save_json({"rows": rows, "total_steps": args.total_steps,
                     "val_size": len(val_keys)}, out / "data_scaling.json")

    lines = ["# Data-scale sweep (held-out)", "",
             f"Fixed compute (~{args.total_steps} steps), latent "
             f"{args.latent_tokens*args.latent_dim} floats, {len(val_keys)} held-out val.",
             "The key question: does held-out RMSD drop below the centroid baseline "
             "and toward PCA as data grows?", "",
             "| n_train | epochs | held-out all-atom RMSD (A) | held-out backbone RMSD (A) | "
             "contact F1 | centroid baseline (A) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['n_train']} | {r['epochs']} | {r['heldout_all_atom_rmsd']:.3g} | "
                     f"{r['heldout_backbone_rmsd']:.3g} | {r['heldout_contact_f1']:.3g} | "
                     f"{r['centroid_baseline_rmsd']:.3g} |")
    (out / "data_scaling_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[data-scan] wrote {out/'data_scaling_table.md'}")


if __name__ == "__main__":
    main()
