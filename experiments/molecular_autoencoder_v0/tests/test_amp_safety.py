"""Losses and alignment must survive mixed precision (AMP).

Regression for a crash seen on the cluster: under AMP the Kabsch solve failed
with `lu_factor_cusolver ... 'Half'` (CPU equivalent: `lu_cpu not implemented
for 'BFloat16'`). Cause: `torch.matmul` is an autocast-to-half op, so casting
the *inputs* to float32 was not enough — the matmul re-cast them, and the
following `det` ran an LU factorisation in half precision. The fix disables
autocast for the whole rigid-transform block.
"""

import torch

from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn
from molae.alignment import kabsch_align_torch, aligned_rmsd_torch
from molae.losses import LossComputer, LossWeights


def _batch(sizes):
    return collate_fn([sample_from_arrays(make_synthetic_ala(n)) for n in sizes])


def test_kabsch_runs_under_autocast():
    pred = torch.randn(3, 20, 3)
    target = torch.randn(3, 20, 3)
    mask = torch.ones(3, 20)
    with torch.autocast(device_type="cpu", enabled=True):
        out = kabsch_align_torch(pred, target, mask)
    assert out.shape == pred.shape
    assert torch.isfinite(out).all()


def test_aligned_rmsd_under_autocast():
    pred = torch.randn(2, 15, 3)
    mask = torch.ones(2, 15)
    with torch.autocast(device_type="cpu", enabled=True):
        r = aligned_rmsd_torch(pred, pred.clone(), mask)
    assert torch.isfinite(r).all()
    assert (r < 1e-2).all()          # identical inputs -> ~0 RMSD


def test_full_loss_backward_under_autocast():
    """The exact path that crashed on the cluster: loss + backward under AMP."""
    batch = _batch([4, 6])
    preds = (batch["coords"] + 0.4 * torch.randn_like(batch["coords"])).requires_grad_(True)
    with torch.autocast(device_type="cpu", enabled=True):
        loss, comp = LossComputer(LossWeights())(preds, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(preds.grad).all()
    for k, v in comp.items():
        assert v == v, f"{k} is NaN under autocast"


def test_vectorized_loss_under_autocast():
    batch = _batch([5, 3])
    preds = (batch["coords"] + 0.4 * torch.randn_like(batch["coords"])).requires_grad_(True)
    with torch.autocast(device_type="cpu", enabled=True):
        loss, _ = LossComputer(LossWeights(), vectorized=True)(preds, batch)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(preds.grad).all()
