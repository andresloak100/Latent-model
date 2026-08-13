"""GENERIC autoencoder metrics. Nothing here knows what a molecule is.

Deliberately absent: bond lengths, angles, clashes, chirality, Ramachandran, contact maps,
secondary structure, aligned RMSD. Those are chemistry, and the brief excludes them from this
study because they would measure the prior rather than the scaling.

`rmsd_unaligned` is included only as a readable unit for `mse`. It is NOT aligned, because the
model is asked to reproduce the input as given; superposing first would credit it for a rotation
it did not have to learn.
"""

from __future__ import annotations

import numpy as np
import torch


def masked_mse(rec, xyz, mask):
    """Mean squared error PER COORDINATE over real atoms."""
    m = mask[..., None].to(rec.dtype)
    return float(((rec - xyz) ** 2 * m).sum() / (m.sum() * 3).clamp_min(1))


def rmsd_unaligned(rec, xyz, mask):
    m = mask[..., None].to(rec.dtype)
    per_atom = ((rec - xyz) ** 2 * m).sum(-1)
    return float(torch.sqrt(per_atom.sum() / m[..., 0].sum().clamp_min(1)))


def latent_spectrum(Z: np.ndarray) -> np.ndarray:
    """Eigenvalues of the latent covariance, descending. Z is (n_samples, latent_size)."""
    Z = Z - Z.mean(0, keepdims=True)
    C = (Z.T @ Z) / max(len(Z) - 1, 1)
    w = np.linalg.eigvalsh(C)[::-1]
    return np.clip(w, 0.0, None)


def effective_rank(Z: np.ndarray) -> float:
    """Participation ratio of the latent covariance spectrum: (sum w)^2 / sum w^2.

    THE MOST INFORMATIVE SINGLE NUMBER IN THIS STUDY. A nominal latent of 512 whose effective
    rank is 30 means the model is not using the capacity it was given, and every point to the
    right of that on the latent-size axis is measuring nothing. It is invisible unless plotted
    against the nominal size.
    """
    w = latent_spectrum(Z)
    s = w.sum()
    return float(s * s / (w * w).sum()) if s > 0 else 0.0


def active_dims(Z: np.ndarray, frac: float = 0.01) -> int:
    """Count of latent dimensions whose variance exceeds `frac` of the largest."""
    v = Z.var(0)
    return int((v > frac * v.max()).sum()) if v.max() > 0 else 0


def interpolation_error(model, batch, device="cpu"):
    """Encode two structures, decode the midpoint of their latents, compare with the midpoint
    of their reconstructions.

    Measures whether the latent is locally linear -- a smooth latent decodes the average to
    roughly the average. A latent that fails this is a poor target for anything that has to move
    through it continuously.
    """
    model.eval()
    with torch.no_grad():
        b = batch.to(device)
        z = model.encode(b.xyz, b.elem, b.mask)
        half = len(z) // 2
        if half == 0:
            return float("nan")
        za, zb = z[:half], z[half:2 * half]
        elem, mask = b.elem[:half], b.mask[:half]
        mid_dec, _ = model.decode(0.5 * (za + zb), elem, mask)
        rec_a, _ = model.decode(za, elem, mask)
        rec_b, _ = model.decode(zb, b.elem[half:2 * half], b.mask[half:2 * half])
        return masked_mse(mid_dec, 0.5 * (rec_a + rec_b), mask)


def linear_probe(Z: np.ndarray, y: np.ndarray, n_train: int | None = None) -> float:
    """R^2 of a least-squares linear map from the frozen latent onto a scalar, held out.

    A LINEAR model only -- the point is what the representation exposes without further
    learning. `y` must be a generic property of the point set (extent, atom count, spread),
    never a chemical label.
    """
    n_train = n_train or int(0.7 * len(Z))
    Xtr = np.c_[Z[:n_train], np.ones(n_train)]
    Xte = np.c_[Z[n_train:], np.ones(len(Z) - n_train)]
    w = np.linalg.lstsq(Xtr, y[:n_train], rcond=None)[0]
    pred = Xte @ w
    resid = ((y[n_train:] - pred) ** 2).sum()
    total = ((y[n_train:] - y[n_train:].mean()) ** 2).sum()
    return float(1.0 - resid / total) if total > 0 else float("nan")
