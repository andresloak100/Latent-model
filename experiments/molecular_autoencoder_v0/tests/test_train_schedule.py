"""Regression: the held-out and train-log schedules must be independent.

The val eval used to live inside the ``log_every`` branch, so an epoch had to
be a multiple of BOTH counters to record a held-out point. For the data-ladder
arms the two counters were coprime-ish (208/417 and 42/85), which reduced the
within-run scaling curve to its two endpoints without any error or warning --
exactly the kind of silent loss a test is for.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train import log_eval_schedule  # noqa: E402


def _counts(epochs, log_every, eval_every, has_val=True):
    logs = evals = 0
    for e in range(epochs):
        do_log, do_eval = log_eval_schedule(e, epochs, log_every, eval_every, has_val)
        logs += do_log
        evals += do_eval
    return logs, evals


def test_misaligned_schedules_still_give_a_full_val_curve():
    """The three ladder rungs, at their real epoch/log/eval settings."""
    for epochs, log_every, eval_every in [(8345, 208, 417),   # n=450
                                          (4400, 110, 220),   # n=878
                                          (1704, 42, 85)]:    # n=2272
        _, evals = _counts(epochs, log_every, eval_every)
        # Every eval_every-th epoch, plus the final one if it is not already a
        # multiple. The old nested form gave 1-2 points for the coprime pairs.
        expected = len(set(list(range(0, epochs, eval_every)) + [epochs - 1]))
        assert evals == expected
        assert evals > 10, (epochs, log_every, eval_every, evals)


def test_log_schedule_is_unaffected_by_eval_every():
    logs_a, _ = _counts(1000, 100, 7)
    logs_b, _ = _counts(1000, 100, 999)
    logs_c, _ = _counts(1000, 100, 0)
    assert logs_a == logs_b == logs_c == 11   # 0..900 (10) + the final epoch 999
    logs_d, _ = _counts(1001, 100, 7)
    assert logs_d == 11                        # 0..1000, final epoch already a multiple


def test_no_val_loader_means_no_eval():
    _, evals = _counts(500, 50, 50, has_val=False)
    assert evals == 0


def test_eval_every_zero_disables_eval():
    _, evals = _counts(500, 50, 0)
    assert evals == 0


def test_final_epoch_always_logs_and_evals():
    # epochs-1 = 999 divides neither counter
    do_log, do_eval = log_eval_schedule(999, 1000, 208, 417, True)
    assert do_log and do_eval
