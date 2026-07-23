"""The invariant encoder's latent must not change under global rotation/translation."""

import numpy as np
import torch

from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn
from molae.model import ModelConfig
from molae.model_equivariant import (
    InvariantEncoder, InvariantAutoencoder, invariant_geometry_features,
)
from molae.losses import LossComputer, LossWeights
from molae.alignment import aligned_rmsd_torch
from molae import utils


def _rand_rigid(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = rng.normal(size=(3,)) * 5.0
    return torch.tensor(q, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)


def _cfg():
    return ModelConfig(d_model=32, n_heads=2, n_latent_tokens=8, latent_dim=4,
                       enc_self_layers=1, dec_self_layers=1)


def test_geometry_features_are_invariant():
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(5))])
    R, t = _rand_rigid(0)
    f1 = invariant_geometry_features(batch["coords"], batch["mask"], k=8)
    rot = batch["coords"] @ R.T + t
    f2 = invariant_geometry_features(rot, batch["mask"], k=8)
    assert torch.allclose(f1, f2, atol=1e-3)


def test_encoder_latent_is_invariant():
    torch.manual_seed(0)
    enc = InvariantEncoder(_cfg()).eval()
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(6)),
                        sample_from_arrays(make_synthetic_ala(4))])
    with torch.no_grad():
        z1 = enc(batch, batch["coords"])
        for seed in (1, 2, 3):
            R, t = _rand_rigid(seed)
            rot = batch["coords"] @ R.T + t
            z2 = enc(batch, rot)
            assert torch.allclose(z1, z2, atol=1e-3), f"not invariant under rigid {seed}"


def test_invariant_autoencoder_forward_and_shapes():
    model = InvariantAutoencoder(_cfg())
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(6)),
                        sample_from_arrays(make_synthetic_ala(4))])
    preds, z = model(batch)
    assert preds.shape == (2, 30, 3)      # padded to 6 residues * 5 atoms
    assert z.shape == (2, 8, 4)


def test_invariant_autoencoder_overfits_single_structure():
    utils.set_seed(0)
    model = InvariantAutoencoder(ModelConfig(d_model=64, n_heads=4, n_latent_tokens=16,
                                             latent_dim=8, enc_self_layers=2, dec_self_layers=2))
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(4))])
    loss_fn = LossComputer(LossWeights())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(400):
        preds, _ = model(batch)
        loss, _ = loss_fn(preds, batch)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds, _ = model(batch)
        rmsd = aligned_rmsd_torch(preds, batch["coords"], batch["mask"]).item()
    assert rmsd < 2.0, f"invariant AE failed to overfit: RMSD={rmsd:.3f}"
