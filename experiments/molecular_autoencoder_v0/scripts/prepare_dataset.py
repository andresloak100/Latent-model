#!/usr/bin/env python3
"""Download, parse, filter, and split a small protein-structure dataset.

Only the PDB ids in the provided list are fetched (no bulk download, no ESM
Atlas). Each structure is parsed and cleaned (see molae.parsing), filtered by
residue/atom band, saved as an .npz, and recorded in a manifest. A train/val
split is produced by single-linkage clustering on sequence similarity so that
similar chains never straddle the split (leakage control).

Usage:
    python scripts/prepare_dataset.py --config configs/stage_a_overfit.yaml
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.parsing import parse_structure, to_npz_dict  # noqa: E402
from molae import utils  # noqa: E402
from molae.config import ExperimentConfig  # noqa: E402


def download_cif(pdb_id: str, cache_dir: Path, retries: int = 4) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pdb_id}.cif"
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    delay = 2
    for attempt in range(retries):
        r = subprocess.run(
            ["curl", "-sS", "-f", "-o", str(out), url],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
        if attempt < retries - 1:
            import time
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"failed to download {pdb_id}: {r.stderr.strip()}")


def read_id_list(path: Path):
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line.upper())
    return ids


def similarity_split(sequences: dict, val_fraction: float, threshold: float, seed: int):
    """Single-linkage cluster by sequence similarity, then split clusters.

    ``sequences`` maps key -> one-letter sequence. Returns (train_keys, val_keys).
    """
    keys = list(sequences.keys())
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            ratio = difflib.SequenceMatcher(None, sequences[ki], sequences[kj]).ratio()
            if ratio >= threshold:
                union(ki, kj)

    clusters = {}
    for k in keys:
        clusters.setdefault(find(k), []).append(k)
    cluster_list = sorted(clusters.values(), key=lambda c: (-len(c), c[0]))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cluster_list))
    n_val_target = max(1, int(round(val_fraction * len(keys)))) if val_fraction > 0 else 0
    val_keys, train_keys = [], []
    for idx in order:
        c = cluster_list[idx]
        if len(val_keys) < n_val_target:
            val_keys.extend(c)
        else:
            train_keys.extend(c)
    if not train_keys:  # tiny datasets: keep everything in train
        train_keys, val_keys = keys, []
    return sorted(train_keys), sorted(val_keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pdb-list", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--sim-threshold", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    processed_dir = ROOT / cfg.data.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else ROOT / "data" / "raw"
    pdb_list = Path(args.pdb_list) if args.pdb_list else ROOT / "data" / "pdb_ids_stage_a.txt"

    ids = read_id_list(pdb_list)
    print(f"[prepare] {len(ids)} candidate ids from {pdb_list.name}")

    manifest = {"kept": [], "rejected": [], "filters": {
        "min_residues": cfg.data.min_residues,
        "max_residues": cfg.data.max_residues,
        "max_atoms": cfg.data.max_atoms,
    }}
    sequences = {}

    for pid in ids:
        try:
            cif = download_cif(pid, cache_dir)
        except Exception as e:
            manifest["rejected"].append({"pdb_id": pid, "reason": f"download_failed: {e}"})
            print(f"  {pid}: DOWNLOAD FAILED ({e})")
            continue
        ps = parse_structure(str(cif), pdb_id=pid)
        if ps is None:
            manifest["rejected"].append({"pdb_id": pid, "reason": "no_peptide_chain"})
            print(f"  {pid}: no peptide chain")
            continue
        nres, natoms = ps.n_residues, ps.n_atoms
        if nres < cfg.data.min_residues or nres > cfg.data.max_residues:
            manifest["rejected"].append(
                {"pdb_id": pid, "reason": f"residues_out_of_band({nres})", **ps.record}
            )
            print(f"  {pid}: {nres} res out of band [{cfg.data.min_residues},{cfg.data.max_residues}]")
            continue
        if natoms > cfg.data.max_atoms:
            manifest["rejected"].append(
                {"pdb_id": pid, "reason": f"too_many_atoms({natoms})", **ps.record}
            )
            print(f"  {pid}: {natoms} atoms > max {cfg.data.max_atoms}")
            continue

        key = f"{ps.pdb_id}_{ps.chain_id}"
        np.savez(processed_dir / f"{key}.npz", **to_npz_dict(ps))
        sequences[key] = ps.sequence
        manifest["kept"].append(ps.record)
        print(f"  {pid}: kept chain {ps.chain_id}  {nres} res  {natoms} atoms  {ps.record['n_bonds']} bonds")

    # Split.
    train_keys, val_keys = similarity_split(
        sequences, args.val_fraction, args.sim_threshold, args.seed
    )
    splits = {"train": train_keys, "val": val_keys,
              "sim_threshold": args.sim_threshold, "val_fraction": args.val_fraction}
    utils.save_json(splits, ROOT / cfg.data.splits_file)
    utils.save_json(manifest, ROOT / "data" / "manifest.json")

    print(f"\n[prepare] kept {len(manifest['kept'])}, rejected {len(manifest['rejected'])}")
    print(f"[prepare] split: {len(train_keys)} train / {len(val_keys)} val")
    print(f"[prepare] processed -> {processed_dir}")


if __name__ == "__main__":
    main()
