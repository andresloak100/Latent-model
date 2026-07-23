"""Differentiable reconstruction losses.

Each term operates on a single structure's real atoms (N, 3); the
``LossComputer`` sums them over a batch (proteins have different sizes, so a
per-sample loop is clearer and correct than padded batched ops). All terms
are rotation/translation invariant: the coordinate term Kabsch-aligns first,
the rest are internal-geometry functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .alignment import kabsch_align_torch


def coord_loss_single(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared per-atom deviation after rigid alignment. (scalar)"""
    mask = torch.ones(pred.shape[0], device=pred.device, dtype=pred.dtype).unsqueeze(0)
    aligned = kabsch_align_torch(pred.unsqueeze(0), target.unsqueeze(0), mask)[0]
    return ((aligned - target) ** 2).sum(dim=-1).mean()


def distance_loss_single(pred, target, max_atoms=1200):
    n = pred.shape[0]
    if n > max_atoms:
        idx = torch.linspace(0, n - 1, max_atoms, device=pred.device).long()
        pred, target = pred[idx], target[idx]
    dp = torch.cdist(pred, pred)
    dt = torch.cdist(target, target)
    iu = torch.triu_indices(pred.shape[0], pred.shape[0], offset=1, device=pred.device)
    return F.smooth_l1_loss(dp[iu[0], iu[1]], dt[iu[0], iu[1]], beta=1.0)


def bond_loss_single(pred, target, bonds):
    if bonds.shape[0] == 0:
        return pred.new_zeros(())
    lp = torch.linalg.norm(pred[bonds[:, 0]] - pred[bonds[:, 1]], dim=1)
    lt = torch.linalg.norm(target[bonds[:, 0]] - target[bonds[:, 1]], dim=1)
    return F.smooth_l1_loss(lp, lt, beta=0.1)


def clash_loss_single(pred, bonded_mask, clash_dist=1.5, max_atoms=1200):
    """Hinge penalty on non-bonded atom pairs closer than ``clash_dist``."""
    n = pred.shape[0]
    if n > max_atoms:  # subsample for tractability on large proteins
        idx = torch.linspace(0, n - 1, max_atoms, device=pred.device).long()
        pred = pred[idx]
        bonded_mask = bonded_mask[idx][:, idx]
    d = torch.cdist(pred, pred)
    iu = torch.triu_indices(pred.shape[0], pred.shape[0], offset=1, device=pred.device)
    dv = d[iu[0], iu[1]]
    bm = bonded_mask[iu[0], iu[1]]
    penalty = F.relu(clash_dist - dv) * (1.0 - bm)
    denom = (1.0 - bm).sum().clamp_min(1.0)
    return penalty.sum() / denom


def chirality_loss_single(pred, target, centers, margin=1.0):
    """Push predicted signed volumes to match the target sign (hinge)."""
    if centers.shape[0] == 0:
        return pred.new_zeros(())

    def signed(coords):
        ca = coords[centers[:, 0]]
        v1 = coords[centers[:, 1]] - ca
        v2 = coords[centers[:, 2]] - ca
        v3 = coords[centers[:, 3]] - ca
        return (torch.cross(v1, v2, dim=-1) * v3).sum(dim=-1)

    sp = signed(pred)
    st = signed(target)
    return F.relu(margin - torch.sign(st) * sp).mean()


@dataclass
class LossWeights:
    coord: float = 1.0
    distance: float = 1.0
    bond: float = 1.0
    clash: float = 0.5
    chirality: float = 0.2


class LossComputer:
    def __init__(self, weights: LossWeights, clash_dist: float = 1.5):
        self.w = weights
        self.clash_dist = clash_dist

    def __call__(self, preds, batch):
        """preds: (B, Nmax, 3) padded. batch: dict from collate_fn."""
        device = preds.device
        totals = {k: torch.zeros((), device=device)
                  for k in ("coord", "distance", "bond", "clash", "chirality")}
        B = preds.shape[0]
        for i in range(B):
            n = int(batch["n_atoms"][i])
            p = preds[i, :n]
            t = batch["coords"][i, :n].to(device)
            bonds = batch["bonds"][i].to(device)
            centers = batch["chirality_centers"][i].to(device)
            bonded_mask = batch["bonded_mask"][i].to(device)
            totals["coord"] = totals["coord"] + coord_loss_single(p, t)
            totals["distance"] = totals["distance"] + distance_loss_single(p, t)
            totals["bond"] = totals["bond"] + bond_loss_single(p, t, bonds)
            totals["clash"] = totals["clash"] + clash_loss_single(p, bonded_mask, self.clash_dist)
            totals["chirality"] = totals["chirality"] + chirality_loss_single(p, t, centers)
        for k in totals:
            totals[k] = totals[k] / max(B, 1)
        total = (
            self.w.coord * totals["coord"]
            + self.w.distance * totals["distance"]
            + self.w.bond * totals["bond"]
            + self.w.clash * totals["clash"]
            + self.w.chirality * totals["chirality"]
        )
        comp = {k: float(v.detach()) for k, v in totals.items()}
        comp["total"] = float(total.detach())
        return total, comp
