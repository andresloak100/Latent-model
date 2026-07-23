"""Kabsch superposition (rigid alignment) for evaluation and losses.

The autoencoder is *not* rotation/translation equivariant by construction
(see model.py). Global rigid transforms are therefore removed at two points:
  * inputs are centred (translation) before encoding;
  * predictions are Kabsch-aligned to the target (rotation + translation)
    before any coordinate-based loss or metric.

`kabsch_align_torch` is differentiable (uses ``torch.linalg.svd``) so it can
sit inside the training loss. `kabsch_rmsd_numpy` mirrors it for metrics.
"""

from __future__ import annotations

import numpy as np
import torch


def kabsch_align_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Align ``pred`` onto ``target`` per batch element under ``mask``.

    Args:
        pred:   (B, N, 3) predicted coordinates.
        target: (B, N, 3) reference coordinates.
        mask:   (B, N) 1.0 for real atoms, 0.0 for padding.
    Returns:
        (B, N, 3) copy of ``pred`` rigidly aligned to ``target``.
    """
    w = mask.unsqueeze(-1)                       # (B, N, 1)
    n = w.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1, 1)

    pred_c = (pred * w).sum(dim=1, keepdim=True) / n
    tgt_c = (target * w).sum(dim=1, keepdim=True) / n
    p = (pred - pred_c) * w
    q = (target - tgt_c) * w

    # Covariance matrix H = p^T q, (B, 3, 3).
    h = torch.einsum("bni,bnj->bij", p, q)
    u, _, vh = torch.linalg.svd(h)
    # Correct for reflection so we get a proper rotation (det = +1).
    d = torch.sign(torch.linalg.det(torch.matmul(u, vh)))
    diag = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0).repeat(pred.shape[0], 1, 1)
    diag[:, 2, 2] = d
    rot = torch.matmul(torch.matmul(u, diag), vh)  # (B, 3, 3)

    aligned = torch.matmul(pred - pred_c, rot) + tgt_c
    return aligned * w


def aligned_rmsd_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True,
) -> torch.Tensor:
    """Per-batch aligned RMSD, (B,). Differentiable."""
    if align:
        pred = kabsch_align_torch(pred, target, mask)
    w = mask
    n = w.sum(dim=1).clamp_min(1.0)
    sq = ((pred - target) ** 2).sum(dim=-1) * w  # (B, N)
    return torch.sqrt(sq.sum(dim=1) / n + 1e-8)


def kabsch_rmsd_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """Aligned RMSD between two (N, 3) arrays (no padding). For metrics."""
    assert pred.shape == target.shape and pred.ndim == 2
    pc = pred - pred.mean(axis=0, keepdims=True)
    tc = target - target.mean(axis=0, keepdims=True)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    diag = np.diag([1.0, 1.0, d])
    rot = u @ diag @ vt
    aligned = pc @ rot
    return float(np.sqrt(((aligned - tc) ** 2).sum(axis=1).mean()))


def kabsch_transform_numpy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``pred`` rigidly aligned onto ``target`` (both (N, 3))."""
    pc_mean = pred.mean(axis=0, keepdims=True)
    tc_mean = target.mean(axis=0, keepdims=True)
    pc = pred - pc_mean
    tc = target - tc_mean
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return pc @ rot + tc_mean
