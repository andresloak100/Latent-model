#!/usr/bin/env python3
"""Fetch a reproducible list of PDB ids from RCSB, single-chain or complex.

Uses the RCSB Search API (HTTPS) to select X-ray protein structures in a
residue-length / resolution band, then deterministically subsamples to ``--n``
ids (seeded). This is how the scaling experiments build a real
few-thousand-structure dataset without hand-listing ids and without any bulk
auto-download (ids only; structures are fetched later by prepare_dataset.py).
ESM Atlas is intentionally not used.

Two modes, differing ONLY in the composition clause:

  ``--mode single`` (default, unchanged)
      exactly one protein entity AND exactly one polymer instance. This is the
      band every reference number was measured on (0.79 A all-atom).

  ``--mode complex``
      at least ``--min-instances`` polymer instances OR at least one
      non-polymer entity -- i.e. assemblies *or* ligand-bearing entries. Both
      are admitted because the complex corpus is what exercises the two
      representation paths that single chains never touch (global res_pos
      across chains, and ordinal-slot ligand groups), and a set restricted to
      multi-chain-only would drop every monomer-plus-cofactor system.

Preserving a split across a rebuild
-----------------------------------
``--keep-list`` pins ids that MUST appear in the output (they are emitted
first, are never subsampled away, and are kept even if the query no longer
returns them). Pass the ids of an existing corpus here when growing it, so the
old validation keys survive verbatim and a data-scaling curve measures data
VOLUME rather than a change of sample composition.

Usage:
    python scripts/fetch_rcsb_ids.py --n 2000 --out data/pdb_ids_scale.txt
    python scripts/fetch_rcsb_ids.py --mode complex --n 8000 \
        --max-res 400 --keep-list data/pdb_ids_complex.txt \
        --out data/pdb_ids_complex_scaled.txt
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def _terminal(attribute, operator, value):
    return {"type": "terminal", "service": "text",
            "parameters": {"attribute": attribute, "operator": operator, "value": value}}


def composition_nodes(mode: str, min_instances: int, require: str = "any"):
    """The only clause that differs between the single-chain and complex sets.

    ``require`` narrows the complex mode to one arm of the OR. That exists so a
    grown corpus can be composition-MATCHED to the one it extends: sorting by
    resolution pulls small high-resolution monomers to the front of the pool, so
    an unconstrained draw drifts toward ligand-bearing monomers and away from
    assemblies. Building the two arms separately and concatenating holds the
    multi-chain fraction fixed, which keeps a data-scaling result a statement
    about VOLUME rather than about an easier training mix.
    """
    if mode == "single":
        return [
            _terminal("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
            _terminal("rcsb_assembly_info.polymer_entity_instance_count", "equals", 1),
        ]
    assembly = _terminal("rcsb_assembly_info.polymer_entity_instance_count",
                         "greater_or_equal", min_instances)
    ligand = _terminal("rcsb_entry_info.nonpolymer_entity_count", "greater_or_equal", 1)
    protein = _terminal("rcsb_entry_info.polymer_entity_count_protein",
                        "greater_or_equal", 1)
    if require == "assembly":
        return [protein, assembly]
    if require == "ligand":
        # A ligand-bearing MONOMER: the arm the assembly query cannot reach.
        return [protein, ligand,
                _terminal("rcsb_assembly_info.polymer_entity_instance_count",
                          "less", min_instances)]
    return [protein, {"type": "group", "logical_operator": "or",
                      "nodes": [assembly, ligand]}]


def build_query(min_res, max_res, max_resolution, start, rows,
                mode="single", min_instances=2, require="any"):
    nodes = composition_nodes(mode, min_instances, require) + [
        # Per-ENTITY sequence length. For a complex this bounds each chain, not
        # the assembly total; prepare_dataset.py enforces the whole-structure
        # residue band and atom cap, so this only keeps the pool sane.
        _terminal("entity_poly.rcsb_sample_sequence_length", "range",
                  {"from": min_res, "to": max_res,
                   "include_lower": True, "include_upper": True}),
        _terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        _terminal("rcsb_entry_info.resolution_combined", "less_or_equal", max_resolution),
    ]
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }


def dedup(ids):
    """Order-preserving unique. Concatenated arms are NOT disjoint (see below)."""
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def merge_pinned(pool, pinned, n, seed):
    """Pinned ids first (verbatim, in order), then a seeded sample of the rest.

    A pinned id is kept even when the query did not return it -- that is the
    point: a rebuilt corpus that silently dropped members would invalidate the
    split derived from the old one.

    Both lists are de-duplicated. That is not defensive tidying: RCSB evaluates
    ``polymer_entity_instance_count`` per ASSEMBLY and an entry may deposit
    several, so an entry whose assembly 1 is a monomer and assembly 2 a dimer
    satisfies both ``>=2`` and ``<2``. The ``assembly`` and ``ligand`` arms
    therefore overlap (~2.5% of the complex pool), and concatenating them
    without this would train on those entries twice.
    """
    pinned = dedup(pinned)
    pinned_set = set(pinned)
    fresh = [i for i in dedup(pool) if i not in pinned_set]
    n_fresh = max(0, n - len(pinned))
    if len(fresh) > n_fresh:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(len(fresh), size=n_fresh, replace=False).tolist())
        fresh = [fresh[i] for i in keep]
    return list(pinned) + fresh


def read_ids(path: Path):
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line.upper())
    return ids


def fetch_page(query, timeout=60):
    url = SEARCH_URL + "?json=" + urllib.parse.quote(json.dumps(query))
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of ids to keep")
    ap.add_argument("--pool", type=int, default=8000,
                    help="how many top-resolution ids to draw the sample from")
    ap.add_argument("--mode", default="single", choices=["single", "complex"],
                    help="single: one protein entity, one polymer instance. "
                         "complex: >=min-instances polymer instances OR a "
                         "non-polymer entity (assemblies and/or ligands).")
    ap.add_argument("--min-instances", type=int, default=2,
                    help="complex mode: polymer instances counted as an assembly")
    ap.add_argument("--require", default="any", choices=["any", "assembly", "ligand"],
                    help="complex mode: narrow to one arm of the OR so a grown "
                         "corpus can be composition-matched (assembly = "
                         "multi-chain; ligand = ligand-bearing monomer).")
    ap.add_argument("--keep-list", default=None,
                    help="ids that must survive verbatim (emitted first, never "
                         "subsampled away). Pass the existing corpus when "
                         "growing it so its val split stays valid.")
    ap.add_argument("--min-res", type=int, default=30)
    ap.add_argument("--max-res", type=int, default=150)
    ap.add_argument("--max-resolution", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/pdb_ids_scale.txt")
    args = ap.parse_args()

    pinned = read_ids(Path(args.keep_list)) if args.keep_list else []
    if pinned:
        print(f"[fetch] pinning {len(pinned)} ids from {args.keep_list}")
    if len(pinned) > args.n:
        raise SystemExit(
            f"--n {args.n} is smaller than the {len(pinned)} pinned ids; the "
            f"pinned set cannot be subsampled, so raise --n."
        )

    ids, start, page = [], 0, 1000
    total = None
    while len(ids) < args.pool:
        q = build_query(args.min_res, args.max_res, args.max_resolution, start, page,
                        mode=args.mode, min_instances=args.min_instances,
                        require=args.require)
        d = fetch_page(q)
        total = d.get("total_count", 0)
        batch = [x["identifier"] for x in d.get("result_set", [])]
        if not batch:
            break
        ids.extend(batch)
        start += page
        if start >= total:
            break
    ids = ids[:args.pool]
    print(f"[fetch] matched {total} structures; drew a pool of {len(ids)}")

    ids = merge_pinned(ids, pinned, args.n, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arm = {"any": f">={args.min_instances} polymer instances OR >=1 non-polymer entity",
           "assembly": f">={args.min_instances} polymer instances",
           "ligand": f"<{args.min_instances} polymer instances AND >=1 non-polymer entity"}
    what = ("single-chain X-ray proteins" if args.mode == "single" else
            f"X-ray protein complexes ({arm[args.require]})")
    header = [
        f"# RCSB {what}, {args.min_res}-{args.max_res} res per entity, "
        f"resolution<={args.max_resolution}A. n={len(ids)} seed={args.seed}.",
    ]
    if pinned:
        header.append(f"# First {len(pinned)} ids are pinned from "
                      f"{Path(args.keep_list).name} and are listed in that order.")
    out.write_text("\n".join(header) + "\n" + "\n".join(ids) + "\n")
    print(f"[fetch] wrote {len(ids)} ids ({len(pinned)} pinned) -> {out}")


if __name__ == "__main__":
    main()
