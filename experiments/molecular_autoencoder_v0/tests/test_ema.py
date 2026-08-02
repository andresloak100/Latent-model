"""Weight EMA and the last-N held-out summary.

The point of both is the same: a single final checkpoint is a draw. These
tests pin the two properties that make the fix trustworthy -- the average
actually converges to the weights it is averaging, and swapping it in for an
evaluation leaves the training weights untouched.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train import ModelEMA, val_summary  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 2)
        self.register_buffer("counter", torch.zeros((), dtype=torch.long))


def test_ema_converges_to_a_constant_weight():
    m = Tiny()
    with torch.no_grad():
        m.lin.weight.fill_(1.0)
    ema = ModelEMA(m, decay=0.9)
    for _ in range(500):
        ema.update(m)
    assert torch.allclose(ema.shadow["lin.weight"], torch.ones(2, 3), atol=1e-5)


def test_ema_lags_a_moving_weight():
    """The whole purpose: the average must not track the last step exactly."""
    m = Tiny()
    with torch.no_grad():
        m.lin.weight.fill_(0.0)
    ema = ModelEMA(m, decay=0.99)
    for _ in range(200):        # warm the average up at 0.0
        ema.update(m)
    with torch.no_grad():
        m.lin.weight.fill_(10.0)
    ema.update(m)
    assert float(ema.shadow["lin.weight"].mean()) < 1.0


def test_warmup_beats_pure_decay_early():
    """Without warmup a 0.999 average is ~all initialisation for 1000 steps."""
    m = Tiny()
    with torch.no_grad():
        m.lin.weight.fill_(1.0)
    ema = ModelEMA(m, decay=0.999)
    ema.shadow["lin.weight"].zero_()      # pretend init was 0
    for _ in range(20):
        ema.update(m)
    naive = 1.0 - 0.999 ** 20             # what pure decay would have reached
    assert float(ema.shadow["lin.weight"].mean()) > naive * 5


def test_as_weights_restores_training_weights_exactly():
    m = Tiny()
    ema = ModelEMA(m, decay=0.9)
    with torch.no_grad():
        m.lin.weight.fill_(5.0)           # diverge the live weights from shadow
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with ema.as_weights(m):
        assert not torch.allclose(m.lin.weight, before["lin.weight"])
    after = m.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), k


def test_integer_buffers_are_copied_not_averaged():
    m = Tiny()
    ema = ModelEMA(m, decay=0.9)
    with torch.no_grad():
        m.counter.fill_(7)
    ema.update(m)
    assert ema.shadow["counter"].dtype == torch.long
    assert int(ema.shadow["counter"]) == 7


def test_state_dict_round_trip_survives_resume():
    m = Tiny()
    ema = ModelEMA(m, decay=0.9)
    with torch.no_grad():
        m.lin.weight.fill_(3.0)
    for _ in range(50):
        ema.update(m)
    saved = ema.state_dict()

    fresh = ModelEMA(Tiny(), decay=0.9)
    fresh.load_state_dict(saved)
    assert fresh.n == ema.n
    assert torch.allclose(fresh.shadow["lin.weight"], ema.shadow["lin.weight"])


def test_val_summary_reports_spread_not_just_the_final_draw():
    log = [{"epoch": i, "val_rmsd": v}
           for i, v in enumerate([9.0] + [6.0, 5.0, 7.0, 5.5, 6.5])]
    s = val_summary(log, last=5)["val_rmsd"]
    assert s["n"] == 5                      # the 9.0 is outside the window
    assert s["final"] == pytest.approx(6.5)
    assert s["mean"] == pytest.approx(6.0)
    assert s["min"] == pytest.approx(5.0)
    assert s["max"] == pytest.approx(7.0)
    assert s["sd"] > 0.5


def test_val_summary_ignores_nan_and_missing_rows():
    log = [{"epoch": 0, "rmsd": 1.0},                      # no val at all
           {"epoch": 1, "val_rmsd": float("nan")},
           {"epoch": 2, "val_rmsd": 4.0},
           {"epoch": 3, "val_rmsd": 6.0}]
    s = val_summary(log, last=10)["val_rmsd"]
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(5.0)


def test_val_summary_is_empty_without_evaluations():
    assert val_summary([{"epoch": 0, "rmsd": 1.0}]) == {}


def test_val_summary_single_eval_has_zero_sd():
    s = val_summary([{"epoch": 0, "val_rmsd": 5.0}])["val_rmsd"]
    assert s["n"] == 1 and s["sd"] == 0.0
