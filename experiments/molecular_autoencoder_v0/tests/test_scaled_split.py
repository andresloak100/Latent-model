"""Growing a corpus can invalidate the ladder it was grown for, silently.

The failure that has no symptom: a 10x larger draw from the PDB pulls in
structures whose sequence is identical to a validation structure under a
different id. Held-out RMSD then improves with corpus size because the model
is memorising val's homologs -- which is indistinguishable, in the result
table, from the data-scaling effect we are trying to measure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_scaled_split import (  # noqa: E402
    exact_leaks, jaccard, kmers, manifest_key, near_duplicates, sequences_of,
)


def _rec(pdb_id, chain_id, *seqs):
    return {"pdb_id": pdb_id, "chain_id": chain_id, "per_chain_sequence": list(seqs)}


def test_manifest_key_matches_prepare_datasets_filename():
    """prepare_dataset writes f'{pdb}_{chain.replace(",", "-")}.npz'."""
    assert manifest_key(_rec("2WFJ", "A,B")) == "2WFJ_A-B"
    assert manifest_key(_rec("1ABC", "A")) == "1ABC_A"


def test_sequences_of_handles_list_and_string_and_empty():
    assert sequences_of(_rec("X", "A", "MKV", "GGG")) == {"MKV", "GGG"}
    assert sequences_of({"per_chain_sequence": "MKV"}) == {"MKV"}
    assert sequences_of({"per_chain_sequence": []}) == set()
    assert sequences_of({}) == set()
    assert sequences_of({"per_chain_sequence": ["", "MKV"]}) == {"MKV"}


def test_exact_leak_caught_under_a_different_pdb_id():
    """The whole point: same protein, different deposition, different key."""
    val_seqs = {"MKVLAAGIVGYW"}
    candidates = {
        "9ZZZ_A": {"MKVLAAGIVGYW"},   # redeposited val protein
        "8YYY_A": {"QQQQPPPPRRRR"},   # genuinely new
    }
    assert exact_leaks(candidates, val_seqs) == {"9ZZZ_A"}


def test_one_shared_chain_of_many_is_enough_to_leak():
    """A complex sharing ONE chain with val still hands over that chain."""
    val_seqs = {"MKVLAAGIVGYW"}
    candidates = {"7XXX_A-B": {"QQQQPPPPRRRR", "MKVLAAGIVGYW"}}
    assert exact_leaks(candidates, val_seqs) == {"7XXX_A-B"}


def test_no_leak_when_nothing_is_shared():
    assert exact_leaks({"1AAA_A": {"QQQQ"}}, {"MKVL"}) == set()


def test_empty_sequences_never_count_as_a_leak():
    """A record with no sequence must not match every val structure."""
    assert exact_leaks({"1AAA_A": set()}, {"MKVL"}) == set()
    assert exact_leaks({"1AAA_A": {"MKVL"}}, set()) == set()


def test_kmers_and_jaccard_basics():
    assert kmers("ABCDE", 3) == {"ABC", "BCD", "CDE"}
    assert kmers("AB", 5) == {"AB"}, "short sequences fall back to the whole string"
    assert kmers("", 3) == set()
    assert jaccard({"A", "B"}, {"A", "B"}) == 1.0
    assert jaccard({"A"}, {"B"}) == 0.0
    assert jaccard(set(), {"A"}) == 0.0


def test_near_duplicate_found_for_a_single_point_mutation():
    val = "MKVLAAGIVGYWMKVLAAGIVGYWMKVLAAGIVGYW"
    mutant = val[:20] + "P" + val[21:]
    hits = near_duplicates({"9ZZZ_A": {mutant}}, {val}, threshold=0.9)
    assert [k for k, _ in hits] == ["9ZZZ_A"]
    assert hits[0][1] > 0.95


def test_unrelated_sequence_is_not_a_near_duplicate():
    val = "MKVLAAGIVGYWMKVLAAGIVGYWMKVLAAGIVGYW"
    other = "QWERTYIPASDFQWERTYIPASDFQWERTYIPASDF"
    assert near_duplicates({"8YYY_A": {other}}, {val}, threshold=0.9) == []


def test_near_duplicates_are_reported_worst_first():
    val = "MKVLAAGIVGYWMKVLAAGIVGYWMKVLAAGIVGYW"
    near = val[:30] + "PPPPPP"
    hits = near_duplicates({"A_A": {near}, "B_B": {val}}, {val}, threshold=0.8)
    assert [k for k, _ in hits] == ["B_B", "A_A"]
    assert hits[0][1] == 1.0


def test_prefilter_does_not_drop_a_true_near_duplicate():
    """A loose prefilter may over-admit; it must never under-admit."""
    val = "MKVLAAGIVGYW" * 5
    mutant = val[:30] + "P" + val[31:]
    strict = near_duplicates({"X_A": {mutant}}, {val}, threshold=0.9, prefilter=0.0)
    default = near_duplicates({"X_A": {mutant}}, {val}, threshold=0.9)
    assert [k for k, _ in strict] == [k for k, _ in default] == ["X_A"]


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_identical_sequence_clears_every_threshold(threshold):
    val = "MKVLAAGIVGYW" * 4
    assert near_duplicates({"X_A": {val}}, {val}, threshold=threshold)


def test_reference_train_is_exempt_so_the_anchor_survives():
    """Per-chain exclusion is STRICTER than the split the baseline used.

    The reference corpus clustered on the concatenated structure sequence; a
    complex whose chain A matches a val complex's chain A passes that rule and
    fails this one. Deleting such a structure from the reference training set
    would mean the anchor rung is no longer the baseline's training set.
    """
    val_seqs = {"MKVL"}
    candidates = {"5OLD_A-B": {"MKVL", "QQQQ"}, "9RED_A": {"MKVL"}}
    exempt = {"5OLD_A-B"}
    all_leaks = exact_leaks(candidates, val_seqs)
    assert all_leaks == {"5OLD_A-B", "9RED_A"}
    assert all_leaks - exempt == {"9RED_A"}, "only the NEW leak is excluded"
    assert all_leaks & exempt == {"5OLD_A-B"}, "the baseline's own leakage is reported"
