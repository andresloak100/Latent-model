"""A 17,000-structure parse must survive being killed.

The manifest was written once, at the very end, so an OOM at 65% discarded
every record even though the .npz files were on disk. The journal makes each
outcome durable as it happens; these tests pin the two ways a resume can be
worse than starting over -- silently mixing two recipes into one corpus, and
recording a structure whose file never got written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_dataset import _mem_mb, read_journal  # noqa: E402

FILTERS = {"min_residues": 20, "max_residues": 400, "max_atoms": 3000,
           "multi_chain": True, "keep_ligands": True, "min_ligand_atoms": 6,
           "keep_modified_residues": True}


def _write(tmp_path, lines, npz_keys=()):
    d = tmp_path / "processed"
    d.mkdir(exist_ok=True)
    for k in npz_keys:
        (d / f"{k}.npz").write_bytes(b"x")
    p = d / "parse_journal.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return p, d


def _kept(pid, key, seq="MKV"):
    return {"pdb_id": pid, "status": "kept", "key": key, "sequence": seq,
            "record": {"pdb_id": pid, "chain_id": key.split("_", 1)[1],
                       "n_kept_chains": 2, "n_ligand_atoms": 9}}


def _rejected(pid, reason):
    return {"pdb_id": pid, "status": "rejected", "key": None, "sequence": None,
            "record": {"pdb_id": pid, "reason": reason}}


def test_missing_journal_is_a_fresh_start(tmp_path):
    done, kept, rejected, seqs = read_journal(
        tmp_path / "nope.jsonl", FILTERS, tmp_path)
    assert done == {} and kept == [] and rejected == [] and seqs == {}


def test_replayed_records_reproduce_the_manifest(tmp_path):
    p, d = _write(tmp_path,
                  [{"_meta": True, "filters": FILTERS},
                   _kept("2WFJ", "2WFJ_A-B"), _rejected("1ABC", "no_peptide_chain")],
                  npz_keys=["2WFJ_A-B"])
    done, kept, rejected, seqs = read_journal(p, FILTERS, d)
    assert done == {"2WFJ": "kept", "1ABC": "rejected"}
    assert [r["pdb_id"] for r in kept] == ["2WFJ"]
    assert [r["pdb_id"] for r in rejected] == ["1ABC"]
    assert seqs == {"2WFJ_A-B": "MKV"}


def test_changed_filters_abort_rather_than_mix_recipes(tmp_path):
    """Half one residue band and half another, undetectable downstream."""
    p, d = _write(tmp_path, [{"_meta": True, "filters": FILTERS}, _kept("2WFJ", "2WFJ_A-B")],
                  npz_keys=["2WFJ_A-B"])
    changed = dict(FILTERS, max_atoms=1200)
    with pytest.raises(SystemExit) as e:
        read_journal(p, changed, d)
    assert "different filters" in str(e.value)


def test_kept_record_without_its_npz_is_reparsed(tmp_path, capsys):
    """Killed between the journal write and the save."""
    p, d = _write(tmp_path,
                  [{"_meta": True, "filters": FILTERS},
                   _kept("2WFJ", "2WFJ_A-B"), _kept("9GON", "9GON_A")],
                  npz_keys=["2WFJ_A-B"])
    done, kept, rejected, seqs = read_journal(p, FILTERS, d)
    assert "9GON" not in done, "must be re-parsed, not trusted"
    assert [r["pdb_id"] for r in kept] == ["2WFJ"]
    assert "9GON_A" not in seqs
    assert "re-parsed" in capsys.readouterr().out


def test_a_torn_final_line_does_not_abort_the_resume(tmp_path):
    """A killed process can leave a half-written line."""
    p, d = _write(tmp_path, [{"_meta": True, "filters": FILTERS}, _kept("2WFJ", "2WFJ_A-B")],
                  npz_keys=["2WFJ_A-B"])
    with open(p, "a") as f:
        f.write('{"pdb_id": "9XYZ", "status": "ke')
    done, kept, _, _ = read_journal(p, FILTERS, d)
    assert done == {"2WFJ": "kept"}
    assert len(kept) == 1


def test_rejected_records_do_not_need_an_npz(tmp_path):
    p, d = _write(tmp_path, [{"_meta": True, "filters": FILTERS},
                             _rejected("1ABC", "too_many_atoms(9001)")])
    done, kept, rejected, _ = read_journal(p, FILTERS, d)
    assert done == {"1ABC": "rejected"}
    assert len(rejected) == 1


def test_mem_reports_anon_separately_from_total():
    """Total RSS conflates the leak question with reclaimable page cache."""
    m = _mem_mb()
    assert m["rss"] > 0
    assert m["anon"] == m["anon"], "RssAnon should be readable on Linux"
    assert m["anon"] <= m["rss"] + 1e-6
    assert m["file"] >= 0
