"""Flow-matching (diffusion) autoencoder: mechanics + overfit gate."""

import torch

from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn
from molae.model import ModelConfig
from molae.flow import FlowMatchingAutoencoder, zero_com, timestep_embedding
from molae.model_equivariant import make_autoencoder
from molae.alignment import aligned_rmsd_torch
from molae import utils


def _cfg(**kw):
    d = dict(d_model=48, n_heads=2, n_latent_tokens=8, latent_dim=8,
             enc_self_layers=1, dec_self_layers=1,
             objective="flowmatch", encoder_type="perresidue", flow_steps=20)
    d.update(kw)
    return ModelConfig(**d)


def test_zero_com_centres():
    x = torch.randn(2, 10, 3) + 5.0
    mask = torch.ones(2, 10)
    xc = zero_com(x, mask)
    assert torch.allclose(xc.mean(dim=1), torch.zeros(2, 3), atol=1e-5)


def test_timestep_embedding_shape():
    emb = timestep_embedding(torch.rand(4), 32)
    assert emb.shape == (4, 32)


def test_factory_builds_flow():
    assert isinstance(make_autoencoder(_cfg()), FlowMatchingAutoencoder)


def test_training_loss_backprops():
    model = make_autoencoder(_cfg())
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(5))])
    loss, comp = model.training_loss(batch)
    loss.backward()
    assert "flow_mse" in comp
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_reconstruct_shape_and_zero_com():
    model = make_autoencoder(_cfg())
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(6)),
                        sample_from_arrays(make_synthetic_ala(4))])
    with torch.no_grad():
        x = model.reconstruct(batch, n_steps=10)
    assert x.shape == (2, 30, 3)


def test_flow_overfits_single_structure():
    utils.set_seed(0)
    model = make_autoencoder(_cfg(d_model=96, n_heads=4, latent_dim=16,
                                  enc_self_layers=2, dec_self_layers=2, flow_steps=50))
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(5))])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(800):
        loss, _ = model.training_loss(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        x = model.reconstruct(batch, n_steps=100)
        rmsd = aligned_rmsd_torch(x, batch["coords"], batch["mask"]).item()
    assert rmsd < 3.0, f"flow AE failed to overfit: RMSD={rmsd:.3f}"
