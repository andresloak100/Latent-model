#!/usr/bin/env python3
"""Export held-out (val) structures as PDB files — input for an external model.

Writes one PDB per held-out structure so an external autoencoder (e.g.
ProteinAE) can reconstruct them; the reconstructions are then scored on our
leak-free bar with score_external_recon.py. Coordinates come straight from the
processed .npz (same held-out folds our own eval uses).

Usage:
    python scripts/export_heldout_pdbs.py --config configs/stage_b_heldout.yaml \
        --out-dir outputs/external/true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.config import ExperimentConfig  # noqa: E402
from molae.pdb_io import write_pdb  # noqa: E402
from molae import utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--out-dir", default="outputs/external/true")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    splits = utils.load_json(ROOT / cfg.data.splits_file)
    keys = (splits["val"] if args.split == "val"
            else splits["train"] if args.split == "train"
            else list(splits["train"]) + list(splits["val"]))
    processed = ROOT / cfg.data.processed_dir
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for k in keys:
        p = processed / f"{k}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        write_pdb(out / f"{k}.pdb", d["coords"].astype(np.float64),
                  d["residue_idx"], d["atom_name_idx"], d["res_seq"],
                  [str(x) for x in d["element_symbol"]], chain_id="A")
        n += 1
    print(f"[export] wrote {n} {args.split} structures -> {out}")


if __name__ == "__main__":
    main()
