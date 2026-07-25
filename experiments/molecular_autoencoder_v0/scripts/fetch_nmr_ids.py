#!/usr/bin/env python3
"""Fetch NMR entries — free conformational ensembles for the latent-suitability gate.

Solution-NMR entries deposit many models of the *same* molecule, so one entry is
a conformational ensemble with identical atom composition and zero MD compute.
That is the distribution latent diffusion will actually run on, and it is what
``scripts/latent_suitability.py`` consumes.

Mirrors fetch_rcsb_ids.py (ids only, no bulk auto-download) but selects
``SOLUTION NMR`` entries (the model-count floor is applied locally), and also
downloads the mmCIF files, since the whole point is the multi-model content that
the normal prepare_dataset.py path discards.

Usage:
    python scripts/fetch_nmr_ids.py --n 40 --out data/nmr_ids.txt --download
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_URL = "https://files.rcsb.org/download/{}.cif"


def build_query(min_res, max_res, start, rows):
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
                "value": "SOLUTION NMR"}},
            # NOTE: rcsb_nmr_ensemble.conformers_submitted_total_number is NOT
            # a searchable attribute (the API rejects it with a 400), so the
            # model-count requirement cannot be pushed into the query. It is
            # enforced locally instead -- latent_suitability.load_ensemble drops
            # entries with fewer than `min_models` deposited models. In practice
            # almost every solution-NMR entry deposits 10-20.
        ]},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "results_content_type": ["experimental"],
            # Sort by ID, NOT by atom count. Sorting ascending by
            # deposited_atom_count (copied from the X-ray fetcher) skims the
            # very smallest entries in the PDB: it produced a cohort averaging
            # 19 residues / 135 atoms against a training distribution of 78
            # residues / 607 atoms, with half the cohort below the training
            # minimum of 20 residues. Reconstruction numbers from that cohort
            # are a domain-shift measurement, not an ensemble measurement.
            "sort": [{"sort_by": "rcsb_entry_container_identifiers.entry_id",
                      "direction": "asc"}],
        },
    }


def fetch_page(query, timeout=60):
    url = SEARCH_URL + "?json=" + urllib.parse.quote(json.dumps(query))
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/nmr_ids.txt")
    ap.add_argument("--raw-dir", default="data/raw_nmr")
    # Defaults deliberately match the TRAINING distribution (20-200 residues,
    # mean 78). Evaluating on 19-residue peptides measures domain shift, not
    # conformational-ensemble compression.
    ap.add_argument("--min-residues", type=int, default=40)
    ap.add_argument("--max-residues", type=int, default=200)
    ap.add_argument("--min-models", type=int, default=8,
                help="enforced locally by load_ensemble, not in the query")
    ap.add_argument("--pool", type=int, default=400, help="candidates to sample from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    ids, start = [], 0
    while len(ids) < args.pool:
        page = fetch_page(build_query(args.min_residues, args.max_residues, start, 100))
        got = [r["identifier"] for r in page.get("result_set", [])]
        if not got:
            break
        ids.extend(got)
        start += len(got)

    ids = sorted(set(ids))
    rng = np.random.default_rng(args.seed)          # deterministic subsample
    if len(ids) > args.n:
        ids = [ids[i] for i in sorted(rng.choice(len(ids), args.n, replace=False))]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + "\n")
    print(f"[fetch_nmr_ids] {len(ids)} ids -> {out}")

    if args.download:
        raw = Path(args.raw_dir)
        raw.mkdir(parents=True, exist_ok=True)
        for i, pid in enumerate(ids, 1):
            dest = raw / f"{pid}.cif"
            if dest.exists():
                continue
            try:
                with urllib.request.urlopen(DOWNLOAD_URL.format(pid), timeout=60) as r:
                    dest.write_bytes(r.read())
            except Exception as exc:                            # noqa: BLE001
                print(f"  [skip] {pid}: {exc}")
                continue
            if i % 10 == 0:
                print(f"  downloaded {i}/{len(ids)}")
        print(f"[fetch_nmr_ids] structures -> {raw}")


if __name__ == "__main__":
    main()
