#!/usr/bin/env python3
"""Select the subset of a grown corpus that matches a reference's composition.

Arm sizing cannot be computed in advance. The parser decides what survives --
chains below the length floor are dropped, ligands below the atom floor are
dropped, structures outside the residue/atom band are rejected -- so the
composition of a *fetched* set and of the *built* corpus are different things,
and the mapping between them is only known after parsing.

So do not predict it. Over-draw every arm, parse, then select here against the
composition actually measured. The cells are the cross of the two axes that
were held fixed:

    multi-chain (n_kept_chains > 1)  x  ligand-bearing (n_ligand_atoms > 0)

Matching both matters because they are not interchangeable. A corpus matched
only on the multi-chain axis drifts on the ligand axis, and a data-scaling
ladder run across it is then comparing corpora that differ in TWO ways -- size,
which is the intervention, and ligand content, which is not. Any slope it
produces is unattributable.

Members of the reference corpus are pinned: they are the anchor rung and must
survive selection whatever the cell arithmetic says.

Usage:
    python scripts/match_composition.py \
        --manifest data/processed_complex_scaled/manifest.json \
        --reference-manifest data/manifest_complex_742.json \
        --n 8000 --out data/keys_complex_matched.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from molae import utils  # noqa: E402

CELLS = ("mc+lig", "mc+noLig", "mono+lig", "mono+noLig")


def cell_of(record) -> str:
    """The composition cell a structure falls in, as the MODEL sees it.

    Both predicates read what survived parsing, not what the PDB advertises:
    ``n_kept_chains`` excludes chains dropped below the length floor, and
    ``n_ligand_atoms`` excludes ligands dropped below the atom floor.
    """
    mc = "mc" if record.get("n_kept_chains", 1) > 1 else "mono"
    lig = "+lig" if record.get("n_ligand_atoms", 0) > 0 else "+noLig"
    return mc + lig


def key_of(record) -> str:
    return f"{record['pdb_id']}_{record['chain_id'].replace(',', '-')}"


def cell_counts(records):
    counts = {c: 0 for c in CELLS}
    for r in records:
        counts[cell_of(r)] += 1
    return counts


def cell_fractions(records):
    counts = cell_counts(records)
    total = sum(counts.values())
    if not total:
        raise SystemExit("[match] reference manifest has no kept structures")
    return {c: counts[c] / total for c in CELLS}


def feasible_total(available, fractions, requested):
    """Largest total that preserves the reference fractions exactly.

    A cell with too few members caps the whole corpus. Silently taking what is
    there instead would produce a corpus that is neither the requested size nor
    the reference composition, and nothing downstream would notice.
    """
    caps = [available[c] / fractions[c] for c in CELLS if fractions[c] > 0]
    return int(min([requested] + caps))


def select(records, reference_records, n, seed=0, pinned=()):
    """Keys forming an n-structure subset with the reference's cell fractions."""
    fractions = cell_fractions(reference_records)
    by_cell = {c: [] for c in CELLS}
    for r in records:
        by_cell[cell_of(r)].append(key_of(r))
    for c in CELLS:
        by_cell[c].sort()

    pinned = set(pinned)
    available = {c: len(by_cell[c]) for c in CELLS}
    total = feasible_total(available, fractions, n)

    rng = np.random.default_rng(seed)
    chosen, shortfalls = [], {}
    for c in CELLS:
        want = int(round(fractions[c] * total))
        pool = by_cell[c]
        must = [k for k in pool if k in pinned]
        rest = [k for k in pool if k not in pinned]
        take = max(0, want - len(must))
        if take > len(rest):
            shortfalls[c] = want - (len(must) + len(rest))
            take = len(rest)
        if take and take < len(rest):
            idx = sorted(rng.choice(len(rest), size=take, replace=False).tolist())
            rest = [rest[i] for i in idx]
        else:
            rest = rest[:take]
        chosen.extend(must + rest)
    return sorted(chosen), fractions, shortfalls, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest of the GROWN corpus")
    ap.add_argument("--reference-manifest", required=True,
                    help="manifest whose composition is being matched")
    ap.add_argument("--n", type=int, default=8000, help="target corpus size")
    ap.add_argument("--out", required=True, help="key list to write")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pin-reference", dest="pin_reference", action="store_false",
                    help="do not force reference members into the selection")
    args = ap.parse_args()

    def load(p):
        p = Path(p)
        return utils.load_json(p if p.is_absolute() else ROOT / p)

    grown = load(args.manifest)["kept"]
    reference = load(args.reference_manifest)["kept"]
    pinned = {key_of(r) for r in reference} if args.pin_reference else set()

    keys, fractions, shortfalls, total = select(
        grown, reference, args.n, args.seed, pinned)

    chosen_records = [r for r in grown if key_of(r) in set(keys)]
    got = cell_counts(chosen_records)
    have = cell_counts(grown)

    print(f"[match] grown corpus {len(grown)}, reference {len(reference)}")
    print(f"[match] target {args.n}, feasible {total}, selected {len(keys)}")
    print(f"{'cell':12s} {'reference':>10s} {'available':>10s} {'selected':>9s} {'got':>8s}")
    for c in CELLS:
        print(f"{c:12s} {fractions[c]:9.1%} {have[c]:10d} "
              f"{got[c]:9d} {got[c]/max(len(keys),1):8.1%}")
    if shortfalls:
        print(f"[match] SHORT in {shortfalls} -- the corpus is smaller than "
              f"requested because those cells ran out. Fetch more of the "
              f"corresponding arm rather than accepting a drifted composition.")
    n_pin = len(pinned & set(keys))
    print(f"[match] {n_pin}/{len(pinned)} reference structures retained")
    if pinned and n_pin < len(pinned):
        print(f"[match] WARNING: {len(pinned) - n_pin} reference structures are "
              f"absent from the grown corpus -- the anchor rung will be short")

    out = Path(args.out)
    (out if out.is_absolute() else ROOT / out).write_text("\n".join(keys) + "\n")
    print(f"[match] wrote {len(keys)} keys -> {args.out}")


if __name__ == "__main__":
    main()
