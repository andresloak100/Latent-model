"""Symmetry-corrected RMSD: don't penalise a swap between identical atoms.

Plain RMSD assumes atom i of the prediction corresponds to atom i of the
target. For automorphic atoms that assumption is arbitrary. Benzene's six
carbons are interchangeable; a reconstruction that places all six correctly
but "labels" them rotated by one position is a *perfect* reconstruction, and
identity-matched RMSD scores it as badly wrong.

This matters more here than in a protein-only codec, because graph addressing
assigns ranks within a symmetry class by a canonical tie-break that carries no
physical meaning. Reporting only identity-matched RMSD would charge the model
for a choice we made for it.

Both numbers are reported. Identity-matched is the strict one and stays the
headline; the gap between them is itself informative -- a large gap means the
model is placing the right shape with the wrong labels, which is a very
different failure from placing the wrong shape.
"""

from __future__ import annotations

import numpy as np

from .alignment import kabsch_rmsd_numpy, kabsch_transform_numpy

try:                                        # optional; a greedy fallback exists
    from scipy.optimize import linear_sum_assignment
except Exception:                           # pragma: no cover
    linear_sum_assignment = None


def _match_within_class(pred, target, idx):
    """Best assignment of predicted to target atoms inside one class."""
    if len(idx) == 1:
        return idx
    cost = np.linalg.norm(pred[idx][:, None, :] - target[idx][None, :, :], axis=-1)
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        out = idx.copy()
        out[c] = idx[r]
        return out
    # Greedy fallback: repeatedly take the cheapest remaining pair.
    out = idx.copy()
    taken_r, taken_c = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in order:
        if r in taken_r or c in taken_c:
            continue
        out[c] = idx[r]
        taken_r.add(int(r))
        taken_c.add(int(c))
    return out


def symmetry_corrected_rmsd(pred, target, classes, n_rounds: int = 2):
    """RMSD after re-matching atoms within each symmetry class.

    Align, re-assign inside each class, re-align. Two rounds: the assignment
    depends on the superposition and the superposition on the assignment, and
    in practice it settles immediately for real structures.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    classes = np.asarray(classes)
    perm = np.arange(len(pred))
    best = kabsch_rmsd_numpy(pred, target)

    for _ in range(n_rounds):
        aligned = kabsch_transform_numpy(pred[perm], target)
        new = perm.copy()
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            if len(idx) > 1:
                new[idx] = perm[_match_within_class(aligned, target, idx)]
        r = kabsch_rmsd_numpy(pred[new], target)
        if r < best - 1e-9:
            best, perm = r, new
        else:
            break
    return float(best)


def rmsd_pair(pred, target, classes):
    """``(identity_matched, symmetry_corrected)`` -- report both."""
    strict = float(kabsch_rmsd_numpy(np.asarray(pred, dtype=np.float64),
                                     np.asarray(target, dtype=np.float64)))
    if classes is None:
        return strict, strict
    return strict, symmetry_corrected_rmsd(pred, target, classes)
