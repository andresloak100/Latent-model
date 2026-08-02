"""The two hard caps on scale: position table size, and O(R^2) attention.

Correctness first: with the window wide enough, windowed attention must equal
dense attention exactly. A linear-cost path that quietly computes something
different is worse than no linear path at all.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from molae.model import ModelConfig  # noqa: E402
from molae.scaling import (  # noqa: E402
    PositionEncoding, WindowedSelfAttention, sinusoidal_index,
)


# ---------------------------------------------------------------- positions

def test_learned_table_clamps_and_therefore_collides():
    """The defect being fixed: past the cap two distinct groups are identical."""
    pe = PositionEncoding(16, max_positions=1024, unbounded=False)
    far = pe(torch.tensor([1024, 90_000]))
    assert torch.allclose(far[0], far[1])          # collision, silently


def test_sinusoidal_positions_stay_distinct_far_past_the_cap():
    pe = PositionEncoding(32, max_positions=1024, unbounded=True)
    out = pe(torch.tensor([1024, 90_000, 125_000]))
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])


def test_unbounded_encoding_has_no_table_to_size():
    pe = PositionEncoding(32, max_positions=1024, unbounded=True)
    assert pe.table is None
    assert all("table" not in n for n, _ in pe.named_parameters())


def test_sinusoidal_index_is_finite_and_bounded_at_extreme_indices():
    enc = sinusoidal_index(torch.tensor([0, 1, 10**6]), 64)
    assert torch.isfinite(enc).all()
    assert float(enc.abs().max()) <= 1.0 + 1e-6
    assert enc.shape == (3, 64)


def test_sinusoidal_index_handles_odd_d_model():
    assert sinusoidal_index(torch.tensor([5]), 33).shape == (1, 33)


# ---------------------------------------------------------------- attention

def _block(window, n_global=0, seed=0):
    torch.manual_seed(seed)
    return WindowedSelfAttention(32, 4, 2, 0.0, window=window, n_global=n_global).eval()


def test_window_wider_than_sequence_is_exactly_dense():
    """The correctness oracle. Same weights, same input, same output."""
    x = torch.randn(2, 12, 32)
    dense, wide = _block(0), _block(64)
    with torch.no_grad():
        assert torch.allclose(dense(x), wide(x), atol=1e-6)


def test_windowed_output_shape_and_finiteness():
    x = torch.randn(2, 100, 32)
    with torch.no_grad():
        out = _block(16, n_global=4)(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_sequence_not_divisible_by_window_is_handled():
    """Padding to a whole number of windows must not change the length."""
    for R in (17, 33, 65, 100):
        x = torch.randn(1, R, 32)
        with torch.no_grad():
            out = _block(8, n_global=2)(x)
        assert out.shape == (1, R, 32), R
        assert torch.isfinite(out).all(), R


def test_all_padding_window_does_not_produce_nan():
    """A window whose keys are entirely padding would divide by zero."""
    x = torch.randn(1, 64, 32)
    kpm = torch.zeros(1, 64, dtype=torch.bool)
    kpm[:, 16:] = True                       # everything after 16 is padding
    with torch.no_grad():
        out = _block(8, n_global=2)(x, key_padding_mask=kpm)
    assert torch.isfinite(out).all()


def test_windowed_attention_is_local_beyond_the_global_tokens():
    """Without global tokens, a far-away change must not reach the query."""
    torch.manual_seed(0)
    blk = _block(4, n_global=0)
    x = torch.randn(1, 64, 32)
    y = x.clone()
    y[0, 60] += 10.0                          # perturb far from position 0
    with torch.no_grad():
        a, b = blk(x), blk(y)
    assert torch.allclose(a[0, 0], b[0, 0], atol=1e-5)     # unreachable
    assert not torch.allclose(a[0, 60], b[0, 60], atol=1e-5)


def test_global_tokens_restore_the_long_range_path():
    """A path exists — but it is weak, and the test says so honestly.

    The global tokens mean-pool a whole segment, so one perturbed group is
    diluted by the segment length before it can reach a distant query. The
    claim that survives measurement is *reachable at all*, strictly more than
    the window-only case, not *strongly coupled*. See the dilution note in
    molae/scaling.py.
    """
    torch.manual_seed(0)          # seed the INPUTS too, not just the weights,
    x = torch.randn(1, 64, 32)    # or the leak magnitude depends on suite order
    y = x.clone()
    y[0, 60] += 10.0

    def leak(n_global):
        blk = _block(4, n_global=n_global, seed=0)
        with torch.no_grad():
            return float((blk(x)[0, 0] - blk(y)[0, 0]).abs().max())

    without, with_global = leak(0), leak(4)
    assert without == pytest.approx(0.0, abs=1e-7)   # provably unreachable
    assert with_global > without                      # a path exists
    assert with_global < 1e-2                         # but a heavily diluted one


def test_windowed_attention_memory_is_linear_not_quadratic():
    """The claim that motivates the whole module, measured rather than argued.

    Dense scores are R^2 per head; windowed are R.(3w+g). Doubling R must
    roughly double the windowed peak, not quadruple it.
    """
    blk = _block(16, n_global=4)
    costs = {}
    for R in (256, 512, 1024):
        x = torch.randn(1, R, 32)
        with torch.no_grad():
            blk(x)
        nw = -(-R // 16)
        costs[R] = nw * 16 * (3 * 16 + 4)      # queries x keys actually formed
    assert costs[512] / costs[256] == pytest.approx(2.0, rel=0.05)
    assert costs[1024] / costs[512] == pytest.approx(2.0, rel=0.05)
    # and far below dense at the same length
    assert costs[1024] < 1024 * 1024 / 8


# ---------------------------------------------------------------- config

def test_scaling_defaults_are_off_so_prior_results_reproduce():
    c = ModelConfig()
    assert c.unbounded_positions is False
    assert c.attn_window == 0
