"""Geometry-aware reconstruction metrics (all operate on NumPy, no grad).

Every metric is invariant to global rotation/translation: coordinate RMSDs
are Kabsch-aligned first; distance, bond, clash, chirality and contact
metrics are functions of internal geometry only.

None of these metrics claims physical/energetic accuracy. They quantify how
well *geometry* is reconstructed, nothing more (see README caveats).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .alignment import kabsch_rmsd_numpy


BACKBONE_IDX_SET = {C.ATOM_NAME_TO_IDX[a] for a in C.BACKBONE_ATOMS}
_CA = C.ATOM_NAME_TO_IDX["CA"]
_N = C.ATOM_NAME_TO_IDX["N"]
_Cc = C.ATOM_NAME_TO_IDX["C"]
_CB = C.ATOM_NAME_TO_IDX["CB"]


@dataclass
class TopologyInfo:
    backbone_mask: np.ndarray     # (N,) bool
    ca_per_residue: np.ndarray    # (L,) atom index of CA for each residue (-1 if none)
    chirality_centers: np.ndarray # (K, 4) indices [CA, N, C, CB]
    bonds: np.ndarray             # (M, 2)
    radii: np.ndarray             # (N,) covalent radii


def build_topology_info(
    atom_name_idx: np.ndarray,
    res_pos: np.ndarray,
    bonds: np.ndarray,
    element_symbol,
) -> TopologyInfo:
    n = atom_name_idx.shape[0]
    backbone_mask = np.isin(atom_name_idx, list(BACKBONE_IDX_SET))

    n_res = int(res_pos.max()) + 1 if n else 0
    ca_per_residue = np.full(n_res, -1, dtype=np.int64)
    centers = []
    for r in range(n_res):
        sel = np.where(res_pos == r)[0]
        names = {int(atom_name_idx[i]): i for i in sel}
        if _CA in names:
            ca_per_residue[r] = names[_CA]
        if all(k in names for k in (_CA, _N, _Cc, _CB)):
            centers.append([names[_CA], names[_N], names[_Cc], names[_CB]])
    chirality_centers = (
        np.array(centers, dtype=np.int64) if centers else np.zeros((0, 4), dtype=np.int64)
    )
    radii = np.array(
        [C.COVALENT_RADII.get(e.upper(), C.DEFAULT_COVALENT_RADIUS) for e in element_symbol],
        dtype=np.float32,
    )
    return TopologyInfo(backbone_mask, ca_per_residue, chirality_centers, bonds, radii)


def _signed_volumes(coords: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    ca = coords[centers[:, 0]]
    v1 = coords[centers[:, 1]] - ca
    v2 = coords[centers[:, 2]] - ca
    v3 = coords[centers[:, 3]] - ca
    return np.einsum("ij,ij->i", np.cross(v1, v2), v3)


def all_atom_rmsd(pred, target):
    return kabsch_rmsd_numpy(pred, target)


def backbone_rmsd(pred, target, topo: TopologyInfo):
    m = topo.backbone_mask
    if m.sum() < 3:
        return float("nan")
    return kabsch_rmsd_numpy(pred[m], target[m])


def pairwise_distance_error(pred, target, max_atoms=1500, seed=0):
    n = pred.shape[0]
    idx = np.arange(n)
    if n > max_atoms:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_atoms, replace=False))
    p, t = pred[idx], target[idx]
    dp = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    dt = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1)
    iu = np.triu_indices(idx.shape[0], k=1)
    diff = np.abs(dp[iu] - dt[iu])
    return float(diff.mean())


def bond_length_error(pred, target, bonds):
    if bonds.shape[0] == 0:
        return float("nan")
    lp = np.linalg.norm(pred[bonds[:, 0]] - pred[bonds[:, 1]], axis=1)
    lt = np.linalg.norm(target[bonds[:, 0]] - target[bonds[:, 1]], axis=1)
    return float(np.abs(lp - lt).mean())


def chirality_violation_rate(pred, target, topo: TopologyInfo):
    centers = topo.chirality_centers
    if centers.shape[0] == 0:
        return float("nan")
    sp = _signed_volumes(pred, centers)
    st = _signed_volumes(target, centers)
    flipped = np.sign(sp) != np.sign(st)
    return float(flipped.mean())


def clash_metrics(coords, bonds, radii, clash_dist=1.2, max_atoms=2000, seed=0):
    """Non-bonded heavy-atom pairs closer than ``clash_dist`` angstrom."""
    n = coords.shape[0]
    idx = np.arange(n)
    if n > max_atoms:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_atoms, replace=False))
    sub = coords[idx]
    remap = -np.ones(n, dtype=np.int64)
    remap[idx] = np.arange(idx.shape[0])
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    iu = np.triu_indices(idx.shape[0], k=1)
    dvals = d[iu]
    # Build set of bonded pairs restricted to the sampled subset.
    bonded = set()
    for a, b in bonds:
        ra, rb = remap[a], remap[b]
        if ra >= 0 and rb >= 0:
            bonded.add((min(ra, rb), max(ra, rb)))
    pair_a, pair_b = iu
    is_clash = dvals < clash_dist
    n_clash = 0
    for k in np.where(is_clash)[0]:
        pa, pb = int(pair_a[k]), int(pair_b[k])
        if (pa, pb) not in bonded:
            n_clash += 1
    per_1000 = 1000.0 * n_clash / max(n, 1)
    return {"clash_count": int(n_clash), "clashes_per_1000_atoms": float(per_1000)}


def contact_map_recovery(pred, target, topo: TopologyInfo, cutoff=8.0, seq_sep=3):
    ca = topo.ca_per_residue
    ca = ca[ca >= 0]
    if ca.shape[0] < seq_sep + 2:
        return {"contact_f1": float("nan"), "contact_precision": float("nan"),
                "contact_recall": float("nan")}
    cp, ct = pred[ca], target[ca]
    dp = np.linalg.norm(cp[:, None, :] - cp[None, :, :], axis=-1)
    dt = np.linalg.norm(ct[:, None, :] - ct[None, :, :], axis=-1)
    L = ca.shape[0]
    ii, jj = np.triu_indices(L, k=seq_sep)
    gt = dt[ii, jj] < cutoff
    pr = dp[ii, jj] < cutoff
    tp = np.sum(gt & pr)
    fp = np.sum(~gt & pr)
    fn = np.sum(gt & ~pr)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"contact_f1": float(f1), "contact_precision": float(prec),
            "contact_recall": float(rec)}


def compute_all_metrics(pred, target, topo: TopologyInfo, latent_floats=None):
    """Full metric dict for one reconstructed structure."""
    n = pred.shape[0]
    out = {
        "all_atom_rmsd": all_atom_rmsd(pred, target),
        "backbone_rmsd": backbone_rmsd(pred, target, topo),
        "pairwise_distance_error": pairwise_distance_error(pred, target),
        "bond_length_error": bond_length_error(pred, target, topo.bonds),
        "chirality_violation_rate": chirality_violation_rate(pred, target, topo),
        "n_atoms": int(n),
    }
    out.update(clash_metrics(pred, topo.bonds, topo.radii))
    out.update(contact_map_recovery(pred, target, topo))
    if latent_floats is not None:
        out["compression_ratio"] = float(n * 3) / float(latent_floats)
        out["latent_floats"] = int(latent_floats)
        out["input_coord_floats"] = int(n * 3)
    return out
