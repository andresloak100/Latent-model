"""Data. A synthetic generator with KNOWN intrinsic dimension, and a loader interface for the
real corpus.

WHY A SYNTHETIC SET WITH A KNOWN ANSWER. A scaling sweep that cannot recover a known result is
not evidence about an unknown one. `SyntheticStructures` draws each structure from a generator
with exactly `intrinsic_dim` degrees of freedom, so the reconstruction curve MUST saturate once
the latent size reaches that number. Running the sweep on it first is a test of the harness, not
of the science, and it costs minutes on a CPU.

The real loader is deliberately a thin interface. It is filled in wherever the corpus lives; the
only requirements this study places on it are stated in `StructureBatch` and in the split rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class StructureBatch:
    """xyz (B,N,3) float32 in ANGSTROM, elem (B,N) long, mask (B,N) bool -- True = real atom."""
    xyz: torch.Tensor
    elem: torch.Tensor
    mask: torch.Tensor

    def to(self, device):
        return StructureBatch(self.xyz.to(device), self.elem.to(device), self.mask.to(device))


def center_and_scale(xyz, mask, scale: float = 10.0):
    """Subtract the per-structure centroid, divide by a FIXED constant.

    Per-structure rescaling would silently remove size information from the input and make the
    reconstruction task easier in a way that varies with the structure. A fixed divisor is a
    units change and nothing else.
    """
    m = mask[..., None].to(xyz.dtype)
    c = (xyz * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
    return (xyz - c) * m / scale


def random_rotations(n, device=None, generator=None):
    """Uniform SO(3) via QR of a Gaussian, sign-fixed. Used as AUGMENTATION.

    This is the generic route to rotation invariance -- learn it from data -- as opposed to
    building an equivariant architecture, which the brief excludes.
    """
    a = torch.randn(n, 3, 3, device=device, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    flip = torch.where(torch.det(q) < 0, -1.0, 1.0)
    q[:, :, 0] = q[:, :, 0] * flip[:, None]
    return q


class SyntheticStructures:
    """Structures with EXACTLY `intrinsic_dim` degrees of freedom, plus optional noise.

    Each structure is `mean_shape + B @ code`, where `B` is a fixed random (3N, k) basis shared
    across the dataset and `code ~ N(0, I_k)`. The best achievable reconstruction with a latent
    of size L is therefore known: perfect for L >= k, and limited by the discarded variance
    below it. That makes the saturation point a checkable prediction.
    """

    def __init__(self, n_structures=2048, n_atoms=64, intrinsic_dim=16, n_elements=4,
                 noise=0.0, seed=0):
        g = np.random.default_rng(seed)
        self.k = intrinsic_dim
        self.n_atoms = n_atoms
        self.noise = noise
        self.mean = g.normal(0, 3.0, size=(n_atoms, 3))
        basis = g.normal(0, 1.0, size=(n_atoms * 3, intrinsic_dim))
        self.basis = np.linalg.qr(basis)[0] * 6.0
        self.codes = g.normal(0, 1.0, size=(n_structures, intrinsic_dim))
        self.elem = g.integers(0, n_elements, size=(n_atoms,))
        self._noise_rng = np.random.default_rng(seed + 1)

    def __len__(self):
        return len(self.codes)

    def batch(self, idx) -> StructureBatch:
        c = self.codes[idx]
        x = self.mean[None] + (c @ self.basis.T).reshape(len(idx), self.n_atoms, 3)
        if self.noise:
            x = x + self._noise_rng.normal(0, self.noise, size=x.shape)
        return StructureBatch(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(np.tile(self.elem, (len(idx), 1)), dtype=torch.long),
            torch.ones(len(idx), self.n_atoms, dtype=torch.bool),
        )

    def split(self, frac_train=0.8):
        n = int(len(self) * frac_train)
        return np.arange(n), np.arange(n, len(self))


class RealCorpus:
    """Interface for the prepared structure corpus. Not implemented here -- no data on this host.

    REQUIREMENTS THIS STUDY PLACES ON IT, and they are not negotiable for the result to mean
    anything:

      1. `split()` must partition BY SEQUENCE HOMOLOGY at a stated identity threshold, not at
         random. A random split puts near-duplicates on both sides and reconstruction error
         reads far better than it is.
      2. `batch()` must return atoms in a fixed, reproducible order.
      3. Element ids must come from a fixed vocabulary recorded alongside the results.
      4. The number of structures in each side, and the threshold, must be reported.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "RealCorpus is an interface. Implement batch()/split() where the corpus lives, "
            "honouring the four requirements in this docstring, and record the identity "
            "threshold and the two split sizes in the results file."
        )
