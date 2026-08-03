#!/usr/bin/env python3
"""Precompute molecule-general atom addressing once, to disk.

Graph addressing needs Weisfeiler-Lehman classes and a canonical rank per
atom. Those are a pure function of the structure and never change, but the
in-process cache that was supposed to hold them does not survive DataLoader
workers: without ``persistent_workers`` every epoch forks fresh processes with
an empty cache, so a 37 ms canonical ordering was being paid for all 4,848
structures every epoch -- 3 minutes per epoch, about 45 hours over a 900-epoch
run. That is what timed the graph arms out.

Written as a sidecar ``<key>.graph.npz`` rather than folded into the structure
file, so the corpus is never rewritten and a bad run of this script cannot
damage data that took hours to build.

Usage:
    python scripts/precompute_graph_features.py --processed-dir data/processed_complex
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.graph_identity import graph_atom_features  # noqa: E402

FIELDS = ("formal_charge", "wl_class", "canonical_rank", "class_ordinal",
          "degree", "max_bond_type")


def sidecar_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(".graph.npz")


def compute_for(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    el = np.asarray(d["element_idx"]).astype(np.int64)
    bonds = np.asarray(d["bonds"]).astype(np.int64).reshape(-1, 2)
    charges = (np.asarray(d["formal_charge"]).astype(np.int64)
               if "formal_charge" in d else None)
    g = graph_atom_features(el, charges, bonds, None)
    return {k: np.asarray(v, dtype=np.int64) for k, v in g.items() if k in FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--force", action="store_true", help="recompute existing sidecars")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.processed_dir)
    if not d.is_absolute():
        d = ROOT / d
    paths = sorted(p for p in d.glob("*.npz") if not p.name.endswith(".graph.npz"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"[graph] {len(paths)} structures in {d}")

    done = skipped = failed = 0
    t0 = time.time()
    for i, p in enumerate(paths):
        out = sidecar_path(p)
        if out.exists() and not args.force:
            skipped += 1
            continue
        try:
            np.savez_compressed(out, **compute_for(p))
            done += 1
        except Exception as e:                      # never abort the whole corpus
            failed += 1
            print(f"  {p.name}: FAILED ({e})")
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paths)}  ({time.time()-t0:.0f}s)")

    print(f"[graph] wrote {done}, skipped {skipped}, failed {failed} "
          f"in {time.time()-t0:.0f}s")
    if failed:
        print("[graph] structures that failed will fall back to computing at "
              "load time, which is slow but correct")


if __name__ == "__main__":
    main()
