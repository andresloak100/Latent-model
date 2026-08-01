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
import gc
import json
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np


def _rss_mb() -> float:
    """Peak resident size. A high-water mark: it never falls, so a flat
    reading is evidence of no growth while a rising one is not by itself
    evidence of a leak."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _mem_mb() -> dict:
    """Anonymous vs file-backed memory, which total RSS conflates.

    Reading tens of thousands of CIFs fills the page cache, and that cache is
    charged to the cgroup and counted in RSS -- so a build can sit pinned at
    the memory limit while its true footprint is a fraction of it, because the
    kernel reclaims cache instead of OOMing. Total RSS therefore cannot answer
    the leak question at all.

    ``RssAnon`` is the number that can: it is the unreclaimable part. If that
    grows linearly with structure count there is a real retention path; if it
    plateaus while RSS pins at the ceiling, the growth was cache.
    """
    out = {"rss": _rss_mb(), "anon": float("nan"), "file": float("nan")}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("RssAnon:"):
                out["anon"] = int(line.split()[1]) / 1024.0
            elif line.startswith("RssFile:"):
                out["file"] = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return out

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae.parsing import parse_structure, to_npz_dict, AllResiduesFiltered  # noqa: E402
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


def prefetch_cifs(ids, cache_dir: Path, jobs: int):
    """Download all CIFs concurrently (I/O bound) before parsing."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in ids if not (cache_dir / f"{p}.cif").exists()]
    if not todo:
        return
    print(f"[prefetch] downloading {len(todo)} CIFs with {jobs} workers...")
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(download_cif, p, cache_dir): p for p in todo}
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:
                print(f"  [prefetch] {futs[f]} failed: {e}")
            if done % 200 == 0:
                print(f"  [prefetch] {done}/{len(todo)}")


def read_id_list(path: Path):
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line.upper())
    return ids


def _cluster_split(clusters, keys, val_fraction, seed):
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


def exact_sequence_split(sequences: dict, val_fraction: float, seed: int):
    """O(N) leakage-control split: group *identical* sequences into clusters.

    Scales to thousands of structures. Catches the dominant PDB leakage source
    (the same protein deposited many times) but NOT near-duplicates -- a proper
    structural clustering (MMseqs2 / Foldseek) is the production solution and is
    noted as future work.
    """
    clusters = {}
    for k, seq in sequences.items():
        clusters.setdefault(seq, []).append(k)
    return _cluster_split(clusters, list(sequences.keys()), val_fraction, seed)


def similarity_split(sequences: dict, val_fraction: float, threshold: float, seed: int):
    """Single-linkage cluster by sequence similarity, then split clusters.

    O(N^2) -- suitable for a few hundred structures. For larger sets use
    ``exact_sequence_split``. Returns (train_keys, val_keys).
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
    return _cluster_split(clusters, keys, val_fraction, seed)


def read_journal(path: Path, filters: dict, processed_dir: Path):
    """Replay a previous run's per-structure records so a build can resume.

    The manifest is written once, at the very end, so an OOM or a preemption
    at 65% discards every parsed structure's record even though the .npz files
    survive on disk. At 17,000 structures that is hours. Each outcome is
    therefore journalled as it happens, and a rerun replays the journal instead
    of re-parsing.

    Two things are checked rather than assumed:

    * The filters must match. Resuming across a changed residue band or ligand
      floor would silently produce a corpus that is half one recipe and half
      another, and nothing downstream could detect it.
    * A ``kept`` entry whose .npz is missing is dropped and re-parsed, since a
      run killed between the journal write and the save would otherwise leave a
      manifest record pointing at a file that does not exist.
    """
    done, kept, rejected, sequences = {}, [], [], {}
    if not path.exists():
        return done, kept, rejected, sequences
    stale = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from a killed process
        if rec.get("_meta"):
            if rec["filters"] != filters:
                raise SystemExit(
                    f"[prepare] {path.name} was written with different filters:\n"
                    f"  journal: {rec['filters']}\n  now:     {filters}\n"
                    f"Resuming would mix two recipes in one corpus. Delete the "
                    f"journal (and the .npz files) to rebuild, or restore the "
                    f"original settings.")
            continue
        pid = rec["pdb_id"]
        if rec["status"] == "kept":
            if not (processed_dir / f"{rec['key']}.npz").exists():
                stale += 1
                continue
            kept.append(rec["record"])
            sequences[rec["key"]] = rec["sequence"]
        else:
            rejected.append(rec["record"])
        done[pid] = rec["status"]
    if stale:
        print(f"[prepare] {stale} journalled structures have no .npz and will "
              f"be re-parsed")
    return done, kept, rejected, sequences


def split_dataset(sequences, val_fraction, threshold, seed, method="auto"):
    """Choose a leakage-control split scalable to the dataset size."""
    n = len(sequences)
    if method == "auto":
        method = "similarity" if n <= 400 else "exact"
    if method == "similarity":
        return method, similarity_split(sequences, val_fraction, threshold, seed)
    return method, exact_sequence_split(sequences, val_fraction, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pdb-list", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--sim-threshold", type=float, default=0.4)
    ap.add_argument("--split-method", default="auto", choices=["auto", "similarity", "exact"])
    ap.add_argument("--multi-chain", action="store_true",
                    help="keep EVERY peptide chain meeting the length floor "
                         "(protein-protein complexes) instead of just one. "
                         "res_pos becomes global across chains, which the "
                         "direct decoder requires.")
    ap.add_argument("--keep-ligands", action="store_true",
                    help="keep non-water hetero groups (ligands, cofactors, "
                         "ions) as ordinal-addressed decoder groups. Needed for "
                         "protein-ligand systems; waters are still dropped.")
    ap.add_argument("--keep-modified-residues", action="store_true",
                    help="keep non-standard residues (SeMet, hydroxyproline, "
                         "D-amino acids, non-canonical residues) IN their "
                         "polymer chain at their sequence position. Without "
                         "this they are dropped; with --keep-ligands but "
                         "without this they are wrongly re-added as free "
                         "ligands and lose their peptide bonds.")
    ap.add_argument("--min-ligand-atoms", type=int, default=1,
                    help="drop kept-ligand groups smaller than this (e.g. 6 to "
                         "exclude lone ions). Only used with --keep-ligands.")
    ap.add_argument("--jobs", type=int, default=8, help="parallel download workers")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="ignore and overwrite the parse journal instead of "
                         "continuing a partial build")
    ap.add_argument("--gc-every", type=int, default=250,
                    help="force a cyclic collection and print RSS every N "
                         "structures. 0 disables both.")
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
        "multi_chain": args.multi_chain,
        "keep_ligands": args.keep_ligands,
        "min_ligand_atoms": args.min_ligand_atoms,
        "keep_modified_residues": args.keep_modified_residues,
    }}
    sequences = {}

    journal_path = processed_dir / "parse_journal.jsonl"
    done = {}
    if args.resume:
        done, manifest["kept"], manifest["rejected"], sequences = read_journal(
            journal_path, manifest["filters"], processed_dir)
        if done:
            print(f"[prepare] resuming: {len(done)} structures already recorded "
                  f"({len(manifest['kept'])} kept), {len(ids) - len(done)} to go")
    elif journal_path.exists():
        journal_path.unlink()

    journal = open(journal_path, "a", buffering=1)  # line-buffered: survives a kill
    if not done:
        journal.write(json.dumps({"_meta": True, "filters": manifest["filters"]}) + "\n")

    def record(pid, status, rec, key=None, sequence=None):
        journal.write(json.dumps(
            {"pdb_id": pid, "status": status, "record": rec,
             "key": key, "sequence": sequence}, default=str) + "\n")

    if args.jobs > 1:
        prefetch_cifs(ids, cache_dir, args.jobs)

    for i, pid in enumerate(ids):
        if pid in done:
            continue
        # Report memory as the loop runs. An OOM 4,600 structures into an
        # 8,000-structure build is otherwise diagnosed by inference: a linear
        # extrapolation from two points cannot tell a genuine leak from heap
        # fragmentation, and the two have different fixes. gemmi structures
        # are large C++ objects behind small Python handles, and CPython's
        # cyclic collector triggers on object COUNT, not bytes -- so a cycle
        # holding one can sit uncollected through thousands of allocations
        # while the process grows. Hence the periodic collect.
        if args.gc_every and i and i % args.gc_every == 0:
            gc.collect()
            m = _mem_mb()
            print(f"  [mem] {i}/{len(ids)} parsed, RSS {m['rss']:.0f} MB "
                  f"(anon {m['anon']:.0f}, file {m['file']:.0f}) -- "
                  f"{m['anon'] / max(i, 1):.3f} MB anon/structure")
        try:
            cif = download_cif(pid, cache_dir)
        except Exception as e:
            rec = {"pdb_id": pid, "reason": f"download_failed: {e}"}
            manifest["rejected"].append(rec); record(pid, "rejected", rec)
            print(f"  {pid}: DOWNLOAD FAILED ({e})")
            continue
        try:
            ps = parse_structure(str(cif), pdb_id=pid,
                                 multi_chain=args.multi_chain,
                                 keep_ligands=args.keep_ligands,
                                 min_ligand_atoms=args.min_ligand_atoms,
                                 keep_modified_residues=args.keep_modified_residues)
        except AllResiduesFiltered as e:
            rec = {"pdb_id": pid, "reason": "all_residues_filtered", "detail": str(e)}
            manifest["rejected"].append(rec); record(pid, "rejected", rec)
            print(f"  {pid}: all residues filtered")
            continue
        if ps is None:
            rec = {"pdb_id": pid, "reason": "no_peptide_chain"}
            manifest["rejected"].append(rec); record(pid, "rejected", rec)
            print(f"  {pid}: no peptide chain")
            continue
        nres, natoms = ps.n_residues, ps.n_atoms
        if nres < cfg.data.min_residues or nres > cfg.data.max_residues:
            rec = {"pdb_id": pid, "reason": f"residues_out_of_band({nres})", **ps.record}
            manifest["rejected"].append(rec); record(pid, "rejected", rec)
            print(f"  {pid}: {nres} res out of band [{cfg.data.min_residues},{cfg.data.max_residues}]")
            continue
        if natoms > cfg.data.max_atoms:
            rec = {"pdb_id": pid, "reason": f"too_many_atoms({natoms})", **ps.record}
            manifest["rejected"].append(rec); record(pid, "rejected", rec)
            print(f"  {pid}: {natoms} atoms > max {cfg.data.max_atoms}")
            continue

        # chain_id is "A,B,C" for multi-chain entries; commas are awkward in
        # filenames and in the split keys derived from them.
        key = f"{ps.pdb_id}_{ps.chain_id.replace(',', '-')}"
        np.savez(processed_dir / f"{key}.npz", **to_npz_dict(ps))
        sequences[key] = ps.sequence
        manifest["kept"].append(ps.record)
        record(pid, "kept", ps.record, key=key, sequence=ps.sequence)
        print(f"  {pid}: kept chain {ps.chain_id}  {nres} res  {natoms} atoms  {ps.record['n_bonds']} bonds")

    journal.close()

    # Split.
    method, (train_keys, val_keys) = split_dataset(
        sequences, args.val_fraction, args.sim_threshold, args.seed, args.split_method
    )
    splits = {"train": train_keys, "val": val_keys, "split_method": method,
              "sim_threshold": args.sim_threshold, "val_fraction": args.val_fraction}
    utils.save_json(splits, ROOT / cfg.data.splits_file)
    # Beside the data it describes, NOT at a fixed data/manifest.json. That
    # path is shared by every corpus, so building a second one silently
    # destroyed the first one's composition record -- which is the only place
    # the multi-chain / ligand-bearing fractions of a corpus are written down,
    # and therefore the only way to tell a data-volume result from a
    # composition shift.
    manifest_path = processed_dir / "manifest.json"
    utils.save_json(manifest, manifest_path)

    print(f"\n[prepare] kept {len(manifest['kept'])}, rejected {len(manifest['rejected'])}")
    print(f"[prepare] split ({method}): {len(train_keys)} train / {len(val_keys)} val")
    print(f"[prepare] manifest -> {manifest_path}")
    print(f"[prepare] processed -> {processed_dir}")


if __name__ == "__main__":
    main()
