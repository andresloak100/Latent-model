"""RoPE / LLM-style encoder: translation-invariant coord code, runs, overfits."""

import torch

from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn
from molae.model import ModelConfig
from molae.model_rope import RoPEAutoencoder, fourier_coord_pe
from molae.model_equivariant import make_autoencoder
from molae.losses import LossComputer, LossWeights
from molae.alignment import aligned_rmsd_torch
from molae import utils


def _cfg(**kw):
    d = dict(d_model=48, n_heads=2, n_latent_tokens=8, latent_dim=4,
             enc_self_layers=1, dec_self_layers=1, rope_freqs=16)
    d.update(kw)
    return ModelConfig(**d)


def test_coord_pe_translation_invariant():
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(5))])
    pe1 = fourier_coord_pe(batch["coords"], batch["mask"], n_freqs=16, spherical=False)
    shifted = batch["coords"] + torch.tensor([3.0, -2.0, 1.5])
    pe2 = fourier_coord_pe(shifted, batch["mask"], n_freqs=16, spherical=False)
    assert torch.allclose(pe1, pe2, atol=1e-3)


def test_factory_builds_rope():
    model = make_autoencoder(_cfg(encoder_type="rope"))
    assert isinstance(model, RoPEAutoencoder)


def test_rope_forward_and_shapes():
    for spherical in (False, True):
        model = RoPEAutoencoder(_cfg(rope_spherical=spherical))
        batch = collate_fn([sample_from_arrays(make_synthetic_ala(6)),
                            sample_from_arrays(make_synthetic_ala(4))])
        preds, z = model(batch)
        assert preds.shape == (2, 30, 3)
        assert z.shape == (2, 8, 4)


def test_rope_overfits_single_structure():
    utils.set_seed(0)
    model = RoPEAutoencoder(ModelConfig(d_model=64, n_heads=4, n_latent_tokens=16,
                                        latent_dim=8, enc_self_layers=2,
                                        dec_self_layers=2, rope_freqs=64))
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
    assert rmsd < 2.0, f"rope AE failed to overfit: RMSD={rmsd:.3f}"
