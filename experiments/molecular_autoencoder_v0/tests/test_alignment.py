"""Alignment invariance tests."""

import numpy as np
import torch

from molae.alignment import (
    kabsch_rmsd_numpy, kabsch_transform_numpy,
    kabsch_align_torch, aligned_rmsd_torch,
)


def _random_rotation(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_rmsd_invariant_to_rigid_transform():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    R = _random_rotation(1)
    t = rng.normal(size=(3,))
    y = x @ R.T + t
    assert kabsch_rmsd_numpy(x, y) < 1e-5


def test_transform_recovers_target():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 3))
    R = _random_rotation(3)
    y = x @ R.T + np.array([5.0, -2.0, 1.0])
    aligned = kabsch_transform_numpy(x, y)
    assert np.allclose(aligned, y, atol=1e-4)


def test_known_perturbation_rmsd():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 3))
    y = x.copy()
    y[:, 0] += 1.0  # pure translation -> RMSD 0 after alignment
    assert kabsch_rmsd_numpy(x, y) < 1e-5
    y2 = x.copy()
    y2 += rng.normal(scale=0.1, size=x.shape)  # small noise
    assert 0.0 < kabsch_rmsd_numpy(x, y2) < 0.3


def test_torch_matches_numpy():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(25, 3)).astype(np.float32)
    R = _random_rotation(6).astype(np.float32)
    y = (x @ R.T + np.array([1, 2, 3], dtype=np.float32))
    pt = torch.tensor(x).unsqueeze(0)
    tt = torch.tensor(y).unsqueeze(0)
    mask = torch.ones(1, x.shape[0])
    r = aligned_rmsd_torch(pt, tt, mask).item()
    assert r < 1e-3


def test_torch_alignment_differentiable():
    x = torch.randn(1, 20, 3, requires_grad=True)
    y = torch.randn(1, 20, 3)
    mask = torch.ones(1, 20)
    loss = aligned_rmsd_torch(x, y, mask).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
