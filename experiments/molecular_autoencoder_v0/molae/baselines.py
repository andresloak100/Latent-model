"""Classical compression baselines for a fair comparison.

The learned autoencoder compresses a whole protein into a fixed (L x d) latent
regardless of atom count. Classical linear methods (PCA) need a fixed input
dimensionality, so we build a *fixed-length backbone cohort*: every structure
is cropped to its first ``n_res`` residues, restricted to backbone atoms
(N, CA, C, O), centred, and Kabsch-aligned to a common reference. Each becomes
a vector in R^(12*n_res). This is the only clean way to run PCA across
variable-size proteins, and it is applied identically to the learned model in
the comparison so the numbers are apples-to-apples.

Baselines:
  * identity      -- returns the input (RMSD 0, compression 1x): trivial upper bound.
  * mean_shape    -- predicts the train-set mean backbone (0 components): trivial lower bound.
  * pca(k)        -- PCA to k components; latent budget = k floats per structure.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from . import constants as C
from .alignment import kabsch_rmsd_numpy, kabsch_transform_numpy

_BB = [C.ATOM_NAME_TO_IDX[a] for a in C.BACKBONE_ATOMS]


def structure_backbone(coords, atom_name_idx, res_pos, n_res):
    """Return (4*n_res, 3) backbone coords for the first ``n_res`` residues.

    Returns ``None`` if any of the first ``n_res`` residues lacks a full
    backbone (N, CA, C, O).
    """
    out = np.zeros((4 * n_res, 3), dtype=np.float64)
    for r in range(n_res):
        sel = np.where(res_pos == r)[0]
        names = {int(atom_name_idx[i]): i for i in sel}
        for a_i, aname in enumerate(_BB):
            if aname not in names:
                return None
            out[4 * r + a_i] = coords[names[aname]]
    return out


def build_backbone_cohort(structures, n_res, ref=None):
    """structures: list of dicts (coords, atom_name_idx, res_pos, pdb_id).

    Returns (X, ids, ref) where X is (S, 12*n_res) aligned+flattened, ids the
    kept pdb ids, ref the reference backbone used for alignment. Pass ``ref``
    to align a second cohort into the SAME frame as a first one (required so a
    PCA fit on a train cohort transfers to a held-out cohort).
    """
    raw, ids = [], []
    for s in structures:
        bb = structure_backbone(s["coords"], s["atom_name_idx"], s["res_pos"], n_res)
        if bb is None:
            continue
        bb = bb - bb.mean(axis=0, keepdims=True)
        raw.append(bb)
        ids.append(s["pdb_id"])
    if not raw:
        return np.zeros((0, 12 * n_res)), [], ref
    if ref is None:
        ref = raw[0]
    aligned = [kabsch_transform_numpy(bb, ref) for bb in raw]
    X = np.stack([a.reshape(-1) for a in aligned], axis=0)
    return X, ids, ref


def _rmsd_rows(recon, truth, n_atoms):
    rmsds = []
    for i in range(recon.shape[0]):
        p = recon[i].reshape(n_atoms, 3)
        t = truth[i].reshape(n_atoms, 3)
        rmsds.append(kabsch_rmsd_numpy(p, t))
    return np.array(rmsds)


def evaluate_baselines(X_train, X_test, n_res, k_list):
    """Return dict: method -> {'mean_backbone_rmsd', 'latent_floats', 'compression_ratio'}."""
    n_atoms = 4 * n_res
    input_floats = X_train.shape[1]  # 12 * n_res
    results = {}

    # identity (trivial, no compression)
    results["identity"] = {
        "mean_backbone_rmsd": 0.0,
        "latent_floats": input_floats,
        "compression_ratio": 1.0,
    }

    # mean shape (0 components)
    mean = X_train.mean(axis=0, keepdims=True)
    recon = np.repeat(mean, X_test.shape[0], axis=0)
    rmsd = _rmsd_rows(recon, X_test, n_atoms)
    results["mean_shape"] = {
        "mean_backbone_rmsd": float(rmsd.mean()),
        "latent_floats": 0,
        "compression_ratio": float("inf"),
    }

    # PCA at each k
    max_k = min(X_train.shape[0], X_train.shape[1])
    for k in k_list:
        if k >= max_k or k <= 0:
            continue
        pca = PCA(n_components=k, random_state=0)
        pca.fit(X_train)
        z = pca.transform(X_test)
        recon = pca.inverse_transform(z)
        rmsd = _rmsd_rows(recon, X_test, n_atoms)
        results[f"pca_k{k}"] = {
            "mean_backbone_rmsd": float(rmsd.mean()),
            "latent_floats": int(k),
            "compression_ratio": float(input_floats) / float(k),
        }
    return results
