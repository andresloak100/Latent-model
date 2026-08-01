"""A data-scaling ladder is only a data-scaling ladder if nothing else moves.

Two things silently break that, and both are ordering problems rather than
anything visible in a result table: rungs that are not nested (so each rung is
a different sample, not a superset), and a smallest rung that is a random draw
of the right size rather than the actual training set a prior run used (so the
ladder has no anchor to check itself against).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_data_scaling import load_pinned, pin_to_front  # noqa: E402


def _rng():
    return np.random.default_rng(0)


def test_pinned_keys_come_first_in_order():
    pool = [f"K{i:03d}" for i in range(100)]
    pinned = ["K042", "K007", "K099"]
    out = pin_to_front(pool, pinned, _rng())
    assert out[:3] == pinned


def test_the_pinned_rung_is_exactly_the_prior_training_set():
    """rung n=len(pinned) must BE the prior set, not a same-size sample."""
    pool = [f"K{i:03d}" for i in range(100)]
    pinned = [f"K{i:03d}" for i in range(0, 60, 2)]  # 30 keys
    out = pin_to_front(pool, pinned, _rng())
    assert sorted(out[:len(pinned)]) == sorted(pinned)


def test_pinning_preserves_the_pool_as_a_set():
    pool = [f"K{i:03d}" for i in range(100)]
    out = pin_to_front(pool, ["K005", "K061"], _rng())
    assert sorted(out) == sorted(pool)
    assert len(out) == len(set(out))


def test_rungs_stay_nested_after_pinning():
    """pool[:n] is how rungs are cut; every rung must contain the smaller one."""
    pool = [f"K{i:03d}" for i in range(100)]
    out = pin_to_front(pool, ["K090", "K091", "K092"], _rng())
    for small, big in ((3, 10), (10, 40), (40, 100)):
        assert set(out[:small]) <= set(out[:big])


def test_missing_pinned_keys_are_dropped_not_invented(capsys):
    pool = [f"K{i:03d}" for i in range(10)]
    out = pin_to_front(pool, ["K001", "NOPE", "K002"], _rng())
    assert "NOPE" not in out
    assert out[:2] == ["K001", "K002"]
    assert sorted(out) == sorted(pool)
    assert "WARNING" in capsys.readouterr().out


def test_all_pinned_keys_missing_is_fatal():
    """Silently falling back to a random pool would produce a bogus anchor."""
    with pytest.raises(SystemExit):
        pin_to_front([f"K{i:03d}" for i in range(10)], ["A", "B"], _rng())


def test_duplicate_pins_are_collapsed():
    pool = [f"K{i:03d}" for i in range(10)]
    out = pin_to_front(pool, ["K003", "K003", "K004"], _rng())
    assert out[:2] == ["K003", "K004"]
    assert len(out) == len(set(out))


def test_no_pins_leaves_the_pool_untouched():
    pool = [f"K{i:03d}" for i in range(50)]
    assert pin_to_front(list(pool), [], _rng()) == pool
    assert pin_to_front(list(pool), None, _rng()) == pool


def test_load_pinned_reads_the_train_list_from_a_splits_json(tmp_path):
    import json
    p = tmp_path / "splits_complex.json"
    p.write_text(json.dumps({"train": ["A_A", "B_B"], "val": ["V_V"]}))
    assert load_pinned(str(p)) == ["A_A", "B_B"]


def test_load_pinned_reads_a_plain_key_list(tmp_path):
    p = tmp_path / "keys.txt"
    p.write_text("# prior training set\nA_A\n\nB_B\n")
    assert load_pinned(str(p)) == ["A_A", "B_B"]


def test_load_pinned_of_nothing_is_empty():
    assert load_pinned(None) == []
    assert load_pinned("") == []
