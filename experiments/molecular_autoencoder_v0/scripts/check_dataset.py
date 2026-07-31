#!/usr/bin/env python3
"""Structural invariants for a built dataset. Run BEFORE training on it.

Why this exists
---------------
Six distinct representation defects reached a GPU queue before being caught,
across five rebuilds. Every one of them was found by a global invariant over
the whole corpus, and none by inspecting individual structures -- which only
ever finds the case you thought to look at.

The most expensive one is the reason for check 6. We perceived ZERO disulfide
bonds for the entire life of this project: 1BPI deposits three at ~2.0 A and
the ``|delta res_pos| <= 1`` adjacency rule excluded every cysteine pair 20+
residues apart. No other check here could have caught it, because a MISSING
bond lowers degrees and every other invariant fires on extra ones.

These checks are cheap, they run on the .npz files rather than the source
CIFs, and they are the gate a new corpus has to pass. MISATO and any other
external dataset arrive with their own preprocessing conventions; assume they
have their own surprises and make them prove otherwise here first.

Usage:
    python scripts/check_dataset.py --processed-dir data/processed_complex
    python scripts/check_dataset.py --processed-dir <dir> --quiet   # CI mode

Exit code is 1 if any HARD invariant fails, 0 otherwise. Soft findings are
reported but do not fail the run.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae import constants as C  # noqa: E402


# Maximum covalent degree per element. Generous: the point is to catch
# invented bonds (a carbon at degree 7), not to police borderline chemistry.
MAX_VALENCE = {
    "C": 4, "N": 4, "O": 3, "S": 6, "P": 5, "SE": 2, "B": 4, "SI": 4,
    "F": 1, "CL": 1, "BR": 1, "I": 1, "H": 1,
    # Metals: coordination is deliberately not perceived as bonding, so any
    # bond on a metal means the radii table has been changed or bypassed.
    "NA": 0, "MG": 0, "K": 0, "CA": 0, "ZN": 0, "FE": 0,
}
DEFAULT_MAX_VALENCE = 4


def check_structure(d):
    """Return (hard_failures, soft_findings, stats) for one structure."""
    hard, soft = [], []
    coords = np.asarray(d["coords"])
    bonds = np.asarray(d["bonds"]).reshape(-1, 2)
    res_pos = np.asarray(d["res_pos"])
    slot = np.asarray(d["slot_idx"]) if "slot_idx" in d else np.asarray(d["atom_name_idx"])
    resid = np.asarray(d["residue_idx"])
    names = [str(x) for x in d["atom_name"]]
    elems = [str(x).upper() for x in d["element_symbol"]]
    n = len(coords)

    # 1. Coordinates are finite. A NaN poisons every downstream aggregate
    #    silently -- an earlier diagnostic reported NaN means for a week.
    if not np.isfinite(coords).all():
        hard.append("non-finite coordinates")

    # 2. Bond indices are in range and canonically ordered (i < j), so
    #    deduplication and membership tests are meaningful.
    if bonds.size:
        if bonds.min() < 0 or bonds.max() >= n:
            hard.append("bond index out of range")
        elif not (bonds[:, 0] < bonds[:, 1]).all():
            hard.append("bond pair not ordered i<j")
        elif len(np.unique(bonds, axis=0)) != len(bonds):
            soft.append("duplicate bond pairs")

    # 3. Decoder addressing is injective. The direct decoder gathers slot
    #    (res_pos * N_SLOTS + slot_idx); a collision means two atoms silently
    #    share one predicted coordinate.
    addr = res_pos * C.N_SLOTS + slot
    if len(set(addr.tolist())) != n:
        hard.append("decoder address collision")

    # 4. No group holds more atoms than it has slots.
    per_group = collections.Counter(res_pos.tolist())
    if per_group and max(per_group.values()) > C.N_SLOTS:
        hard.append(f"group with {max(per_group.values())} atoms > {C.N_SLOTS} slots")

    # 5. Valence. Catches invented bonds: fused duplicate ligand copies and
    #    1,3 shortcuts both showed up here first.
    deg = np.zeros(n, dtype=int)
    for i, j in bonds:
        deg[i] += 1
        deg[j] += 1
    for k in range(n):
        cap = MAX_VALENCE.get(elems[k], DEFAULT_MAX_VALENCE)
        if deg[k] > cap:
            hard.append(f"{names[k]} ({elems[k]}) has degree {deg[k]} > {cap}")
            break              # one report per structure is enough

    # 6. Disulfides. THE missing-bond check, and the only one here that can
    #    catch a bond class we are failing to perceive at all. Every other
    #    invariant is blind to absence.
    sg = [k for k in range(n) if names[k] == "SG"]
    ss_close = 0
    for a_i, a in enumerate(sg):
        for b in sg[a_i + 1:]:
            if np.linalg.norm(coords[a] - coords[b]) < 2.5:
                ss_close += 1
    ss_bonded = sum(1 for i, j in bonds if i in set(sg) and j in set(sg))
    if ss_close and not ss_bonded:
        hard.append(f"{ss_close} SG-SG pairs within 2.5 A but 0 bonded")

    # 7. Orphan atoms. A ligand or residue bonded to nothing is usually a
    #    severed linkage rather than a genuine isolated species.
    isolated = int((deg == 0).sum())

    lig = resid == C.LIGAND_RESIDUE_IDX
    chain = np.asarray(d["chain_idx"]) if "chain_idx" in d else np.zeros(n, dtype=np.int64)
    anchor = sum(1 for i, j in bonds if lig[i] != lig[j])
    tree = sum(1 for i, j in bonds if lig[i] and lig[j] and chain[i] != chain[j])

    # 8. SEVERED covalent links -- a direct test, not an aggregate heuristic.
    #
    # An earlier version hard-failed a corpus with ligands but zero anchors.
    # That encodes a PDB assumption (about half of PDB ligand groups are
    # covalently attached -- glycans to ASN, heme C to CXXCH) which is simply
    # false for a corpus of non-covalent binders: MISATO's ligands are
    # separate molecules with no protein-ligand bond at all, and the check
    # fired on every build. A gate that cries wolf gets ignored, which is
    # worse than no gate.
    #
    # What we actually care about is a pair sitting at COVALENT range and not
    # bonded. Non-covalent binding sits at 2.7 A and up, so a genuine
    # non-covalent corpus scores zero here while a severed glycan link scores
    # one. This is source-agnostic.
    severed = 0
    if lig.any() and (~lig).any():
        li = np.where(lig)[0]
        pi = np.where(~lig)[0]
        bond_set = {(int(a), int(b)) for a, b in bonds}
        dm = np.linalg.norm(coords[li][:, None, :] - coords[pi][None, :, :], axis=-1)
        for a, b in zip(*np.where(dm < 1.9)):
            u, v = int(li[a]), int(pi[b])
            if (min(u, v), max(u, v)) not in bond_set:
                severed += 1
    if severed:
        hard.append(f"{severed} ligand-protein pairs within 1.9 A but not bonded")

    stats = {"n_atoms": n, "n_bonds": len(bonds), "n_ligand_atoms": int(lig.sum()),
             "disulfides": ss_bonded, "anchors": anchor, "trees": tree,
             "isolated_atoms": isolated,
             "monatomic_ions": int(sum(1 for g, c in per_group.items() if c == 1))}
    # An isolated atom inside a MULTI-atom group is the interesting case: a
    # lone ion is expected, an atom of a larger molecule bonded to nothing is
    # a severed linkage. This is how the Fe4S4 clusters surfaced -- the metal
    # radii that stop Zn-His coordination being read as covalent also
    # disconnect the genuine Fe-S bonds inside a cluster.
    orphans = collections.Counter(
        f"{names[k]}({elems[k]})" for k in range(n)
        if deg[k] == 0 and per_group[int(res_pos[k])] > 1)
    return hard, soft, stats, orphans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--quiet", action="store_true", help="only print the verdict")
    ap.add_argument("--max-report", type=int, default=10)
    args = ap.parse_args()

    files = sorted(Path(args.processed_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {args.processed_dir}")

    totals = collections.Counter()
    all_orphans = collections.Counter()
    failures, softs = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        hard, soft, stats, orphans = check_structure(d)
        all_orphans.update(orphans)
        for k, v in stats.items():
            totals[k] += v
        if hard:
            failures.append((f.stem, hard))
        if soft:
            softs.append((f.stem, soft))

    print(f"[check] {len(files)} structures from {args.processed_dir}")
    if not args.quiet:
        print(f"  atoms {totals['n_atoms']:,}   bonds {totals['n_bonds']:,}")
        print(f"  ligand atoms {totals['n_ligand_atoms']:,}   "
              f"anchors {totals['anchors']}   glycan/tree links {totals['trees']}")
        print(f"  disulfides {totals['disulfides']}")
        if totals["n_ligand_atoms"] and totals["anchors"] == 0:
            print("  note: ligands present, zero covalent anchors -- expected "
                  "for a non-covalent binder corpus, not an error")
        print(f"  isolated atoms {totals['isolated_atoms']}   "
              f"(of which monatomic ions, expected: {totals['monatomic_ions']})")
        if all_orphans:
            top = ", ".join(f"{k} x{v}" for k, v in all_orphans.most_common(6))
            print(f"  unbonded atoms inside MULTI-atom groups: "
                  f"{sum(all_orphans.values())}  [{top}]")

    if softs and not args.quiet:
        print(f"\n[check] {len(softs)} structures with soft findings")
        for name, items in softs[:args.max_report]:
            print(f"  {name}: {'; '.join(items)}")

    # Corpus-level checks. These cannot be evaluated per structure: a single
    # protein legitimately has no disulfide, but a corpus of hundreds with
    # zero means the bond class is not being perceived at all.
    corpus_fail = []
    if totals["disulfides"] == 0:
        corpus_fail.append(
            "NO disulfides anywhere in the corpus -- this is what was wrong "
            "for the entire project before struct_conn was read")
    # NOTE: zero anchors is NOT a failure. A corpus of non-covalent binders
    # (MISATO) legitimately has none. Severed anchors are caught per-structure
    # by the covalent-range test above, which is source-agnostic.

    if failures or corpus_fail:
        print(f"\n[check] FAILED: {len(failures)} structures violate a hard invariant")
        for name, items in failures[:args.max_report]:
            print(f"  {name}: {'; '.join(items)}")
        if len(failures) > args.max_report:
            print(f"  ... and {len(failures) - args.max_report} more")
        for msg in corpus_fail:
            print(f"  CORPUS: {msg}")
        raise SystemExit(1)

    print("\n[check] PASS -- all hard invariants hold")


if __name__ == "__main__":
    main()
