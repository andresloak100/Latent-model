"""Composition matching decides whether a data ladder measures one thing.

A corpus matched on the multi-chain axis alone still drifts on the ligand
axis, and then every rung differs from the baseline in two ways -- size, the
intervention, and ligand content, which is not. The slope becomes
unattributable, and nothing in the result table shows it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from match_composition import (  # noqa: E402
    CELLS, cell_counts, cell_fractions, cell_of, feasible_total, key_of, select,
)


def _rec(pdb_id, chains=1, lig=0, chain_id=None):
    return {"pdb_id": pdb_id, "chain_id": chain_id or "A",
            "n_kept_chains": chains, "n_ligand_atoms": lig}


def _corpus(n_per_cell):
    out, i = [], 0
    for cell, n in n_per_cell.items():
        chains = 2 if cell.startswith("mc") else 1
        lig = 9 if cell.endswith("+lig") else 0
        for _ in range(n):
            i += 1
            out.append(_rec(f"X{i:04d}", chains, lig))
    return out


def test_cell_of_reads_what_survived_parsing():
    assert cell_of(_rec("A", chains=2, lig=9)) == "mc+lig"
    assert cell_of(_rec("A", chains=2, lig=0)) == "mc+noLig"
    assert cell_of(_rec("A", chains=1, lig=9)) == "mono+lig"
    assert cell_of(_rec("A", chains=1, lig=0)) == "mono+noLig"


def test_cell_of_defaults_are_the_conservative_ones():
    """A record missing the fields must not be counted as multi-chain."""
    assert cell_of({}) == "mono+noLig"


def test_key_matches_the_npz_filename():
    assert key_of(_rec("2WFJ", chain_id="A,B")) == "2WFJ_A-B"


def test_reference_fractions_reproduce_the_742_corpus():
    """The real cells: 227 / 310 / 95 / 110."""
    ref = _corpus({"mc+lig": 227, "mc+noLig": 310, "mono+lig": 95, "mono+noLig": 110})
    f = cell_fractions(ref)
    assert round(f["mc+lig"], 3) == 0.306
    assert round(f["mc+noLig"], 3) == 0.418
    assert round(f["mono+lig"], 3) == 0.128
    assert round(f["mono+noLig"], 3) == 0.148
    assert round(f["mc+lig"] + f["mc+noLig"], 3) == 0.724, "multi-chain total"
    assert round(f["mc+lig"] + f["mono+lig"], 3) == 0.434, "ligand total"


def test_selection_reproduces_the_reference_composition():
    ref = _corpus({"mc+lig": 227, "mc+noLig": 310, "mono+lig": 95, "mono+noLig": 110})
    grown = _corpus({"mc+lig": 3000, "mc+noLig": 3000, "mono+lig": 3000,
                     "mono+noLig": 3000})
    keys, fractions, short, total = select(grown, ref, n=4000, seed=0)
    assert not short
    assert total == 4000
    chosen = [r for r in grown if key_of(r) in set(keys)]
    got = cell_counts(chosen)
    for c in CELLS:
        assert abs(got[c] / len(keys) - fractions[c]) < 0.01, c


def test_a_short_cell_caps_the_corpus_rather_than_drifting():
    """Taking what's there would give neither the size nor the composition."""
    ref = _corpus({"mc+lig": 25, "mc+noLig": 25, "mono+lig": 25, "mono+noLig": 25})
    grown = _corpus({"mc+lig": 1000, "mc+noLig": 1000, "mono+lig": 1000,
                     "mono+noLig": 40})
    keys, fractions, short, total = select(grown, ref, n=4000, seed=0)
    assert total == 160, "capped by the 40-member cell at 25%"
    chosen = [r for r in grown if key_of(r) in set(keys)]
    got = cell_counts(chosen)
    for c in CELLS:
        assert abs(got[c] / len(keys) - 0.25) < 0.02, c


def test_feasible_total_is_the_binding_cell():
    fractions = {"mc+lig": 0.3, "mc+noLig": 0.4, "mono+lig": 0.15, "mono+noLig": 0.15}
    available = {"mc+lig": 900, "mc+noLig": 4000, "mono+lig": 4000, "mono+noLig": 4000}
    assert feasible_total(available, fractions, 10_000) == 3000
    assert feasible_total(available, fractions, 500) == 500, "request can bind first"


def test_reference_members_are_retained():
    ref = _corpus({"mc+lig": 20, "mc+noLig": 20, "mono+lig": 20, "mono+noLig": 20})
    grown = ref + _corpus({"mc+lig": 500, "mc+noLig": 500, "mono+lig": 500,
                           "mono+noLig": 500})
    pinned = {key_of(r) for r in ref}
    keys, _, _, _ = select(grown, ref, n=400, seed=0, pinned=pinned)
    assert pinned <= set(keys), "the anchor rung must survive selection"


def test_pinning_survives_even_when_a_cell_is_tight():
    ref = _corpus({"mc+lig": 10, "mc+noLig": 10, "mono+lig": 10, "mono+noLig": 10})
    grown = ref[:]
    pinned = {key_of(r) for r in ref}
    keys, _, _, _ = select(grown, ref, n=40, seed=0, pinned=pinned)
    assert pinned <= set(keys)
    assert len(keys) == 40


def test_selection_is_deterministic_and_seed_sensitive():
    ref = _corpus({"mc+lig": 25, "mc+noLig": 25, "mono+lig": 25, "mono+noLig": 25})
    grown = _corpus({"mc+lig": 400, "mc+noLig": 400, "mono+lig": 400, "mono+noLig": 400})
    a = select(grown, ref, n=200, seed=0)[0]
    b = select(grown, ref, n=200, seed=0)[0]
    c = select(grown, ref, n=200, seed=1)[0]
    assert a == b
    assert a != c


def test_no_duplicate_keys_in_the_selection():
    ref = _corpus({"mc+lig": 25, "mc+noLig": 25, "mono+lig": 25, "mono+noLig": 25})
    grown = _corpus({"mc+lig": 300, "mc+noLig": 300, "mono+lig": 300, "mono+noLig": 300})
    keys = select(grown, ref, n=400, seed=0)[0]
    assert len(keys) == len(set(keys))
