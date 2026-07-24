"""The vectorized loss must match the per-sample reference loop exactly."""

import time

import torch

from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn
from molae.losses import LossComputer, LossWeights, batched_losses
from molae import utils


def _batch(sizes):
    return collate_fn([sample_from_arrays(make_synthetic_ala(n)) for n in sizes])


def test_matches_loop_uniform_sizes():
    utils.set_seed(0)
    batch = _batch([5, 5, 5])
    preds = torch.randn_like(batch["coords"])
    fast, _ = LossComputer(LossWeights(), vectorized=True)(preds, batch)
    slow, _ = LossComputer(LossWeights(), vectorized=False)(preds, batch)
    assert torch.allclose(fast, slow, atol=1e-4), f"{fast.item()} vs {slow.item()}"


def test_matches_loop_ragged_sizes_with_padding():
    """The important case: different structure sizes -> padding must not leak."""
    utils.set_seed(1)
    batch = _batch([3, 7, 4, 6])
    preds = torch.randn_like(batch["coords"])
    fast_t, fast_c = LossComputer(LossWeights(), vectorized=True)(preds, batch)
    slow_t, slow_c = LossComputer(LossWeights(), vectorized=False)(preds, batch)
    for k in ("coord", "distance", "bond", "clash", "chirality"):
        assert abs(fast_c[k] - slow_c[k]) < 1e-4, f"{k}: {fast_c[k]} vs {slow_c[k]}"
    assert torch.allclose(fast_t, slow_t, atol=1e-4)


def test_per_term_agreement_on_realistic_predictions():
    """Predictions close to the target (the regime training actually spends
    most of its time in) — hinge terms are near their kinks here."""
    utils.set_seed(2)
    batch = _batch([4, 8])
    preds = batch["coords"] + 0.3 * torch.randn_like(batch["coords"])
    _, fast_c = LossComputer(LossWeights(), vectorized=True)(preds, batch)
    _, slow_c = LossComputer(LossWeights(), vectorized=False)(preds, batch)
    for k in ("coord", "distance", "bond", "clash", "chirality"):
        assert abs(fast_c[k] - slow_c[k]) < 1e-4, f"{k}: {fast_c[k]} vs {slow_c[k]}"


def test_gradients_match():
    utils.set_seed(3)
    batch = _batch([4, 6])
    p1 = (batch["coords"] + 0.5 * torch.randn_like(batch["coords"])).requires_grad_(True)
    p2 = p1.detach().clone().requires_grad_(True)
    LossComputer(LossWeights(), vectorized=True)(p1, batch)[0].backward()
    LossComputer(LossWeights(), vectorized=False)(p2, batch)[0].backward()
    assert torch.isfinite(p1.grad).all()
    assert torch.allclose(p1.grad, p2.grad, atol=1e-4)


def test_padding_does_not_affect_loss():
    """Corrupting padded slots must leave every term unchanged."""
    utils.set_seed(4)
    batch = _batch([3, 9])
    preds = torch.randn_like(batch["coords"])
    _, before = LossComputer(LossWeights(), vectorized=True)(preds, batch)
    n0 = int(batch["n_atoms"][0])
    preds2 = preds.clone()
    preds2[0, n0:] += 500.0                       # garbage in the padded region
    _, after = LossComputer(LossWeights(), vectorized=True)(preds2, batch)
    for k in before:
        assert abs(before[k] - after[k]) < 1e-4, f"padding leaked into {k}"


def test_vectorized_is_not_slower():
    """Sanity: the batched path should be at least as fast as the loop."""
    utils.set_seed(5)
    batch = _batch([8] * 12)
    preds = torch.randn_like(batch["coords"])
    fast_fn = LossComputer(LossWeights(), vectorized=True)
    slow_fn = LossComputer(LossWeights(), vectorized=False)
    fast_fn(preds, batch); slow_fn(preds, batch)          # warm-up
    t0 = time.perf_counter(); [fast_fn(preds, batch) for _ in range(5)]
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); [slow_fn(preds, batch) for _ in range(5)]
    t_slow = time.perf_counter() - t0
    print(f"\nvectorized {t_fast:.3f}s vs loop {t_slow:.3f}s ({t_slow/t_fast:.1f}x)")
    assert t_fast <= t_slow * 1.2
