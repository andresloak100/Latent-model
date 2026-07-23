#!/usr/bin/env python3
"""Fetch a reproducible list of small single-chain protein PDB ids from RCSB.

Uses the RCSB Search API (HTTPS) to select single-chain X-ray protein
structures in a residue-length / resolution band, then deterministically
subsamples to ``--n`` ids (seeded). This is how the scaling experiment builds
a real few-thousand-structure dataset without hand-listing ids and without any
bulk auto-download (ids only; structures are fetched later by
prepare_dataset.py). ESM Atlas is intentionally not used.

Usage:
    python scripts/fetch_rcsb_ids.py --n 2000 --out data/pdb_ids_scale.txt
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def build_query(min_res, max_res, max_resolution, start, rows):
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "equals", "value": 1}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_assembly_info.polymer_entity_instance_count",
                "operator": "equals", "value": 1}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "entity_poly.rcsb_sample_sequence_length",
                "operator": "range",
                "value": {"from": min_res, "to": max_res,
                          "include_lower": True, "include_upper": True}}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "exptl.method", "operator": "exact_match",
                "value": "X-RAY DIFFRACTION"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": max_resolution}},
        ]},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }


def fetch_page(query, timeout=60):
    url = SEARCH_URL + "?json=" + urllib.parse.quote(json.dumps(query))
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of ids to keep")
    ap.add_argument("--pool", type=int, default=8000,
                    help="how many top-resolution ids to draw the sample from")
    ap.add_argument("--min-res", type=int, default=30)
    ap.add_argument("--max-res", type=int, default=150)
    ap.add_argument("--max-resolution", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/pdb_ids_scale.txt")
    args = ap.parse_args()

    ids, start, page = [], 0, 1000
    total = None
    while len(ids) < args.pool:
        q = build_query(args.min_res, args.max_res, args.max_resolution, start, page)
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

    # Deterministic seeded subsample for diversity + reproducibility.
    rng = np.random.default_rng(args.seed)
    if len(ids) > args.n:
        keep = sorted(rng.choice(len(ids), size=args.n, replace=False).tolist())
        ids = [ids[i] for i in keep]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# RCSB single-chain X-ray proteins, {args.min_res}-{args.max_res} res, "
              f"resolution<={args.max_resolution}A. n={len(ids)} seed={args.seed}.\n")
    out.write_text(header + "\n".join(ids) + "\n")
    print(f"[fetch] wrote {len(ids)} ids -> {out}")


if __name__ == "__main__":
    main()
