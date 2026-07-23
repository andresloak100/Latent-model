"""End-to-end: gradients flow, the model overfits, checkpoints round-trip."""

import torch

from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn
from molae.model import MolecularAutoencoder, ModelConfig
from molae.losses import LossComputer, LossWeights
from molae.alignment import aligned_rmsd_torch
from molae import utils


def _tiny_model():
    cfg = ModelConfig(d_model=64, n_heads=4, n_latent_tokens=16, latent_dim=8,
                      enc_self_layers=2, dec_self_layers=2, coord_scale=10.0)
    return MolecularAutoencoder(cfg), cfg


def test_gradients_finite_and_nonzero():
    model, _ = _tiny_model()
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(4))])
    loss_fn = LossComputer(LossWeights())
    preds, _ = model(batch)
    loss, _ = loss_fn(preds, batch)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_overfits_single_structure():
    utils.set_seed(0)
    model, _ = _tiny_model()
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(4))])
    loss_fn = LossComputer(LossWeights())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(500):
        preds, _ = model(batch)
        loss, _ = loss_fn(preds, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        preds, _ = model(batch)
        rmsd = aligned_rmsd_torch(preds, batch["coords"], batch["mask"]).item()
    assert rmsd < 1.5, f"model failed to overfit: RMSD={rmsd:.3f}"


def test_checkpoint_roundtrip(tmp_path):
    model, cfg = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(3))])
    with torch.no_grad():
        z_before = model.encode(batch)
    path = tmp_path / "ckpt.pt"
    utils.save_checkpoint(path, model, opt, epoch=7)

    model2, _ = _tiny_model()
    ckpt = utils.load_checkpoint(path, model2, None)
    assert ckpt["epoch"] == 7
    model2.eval()
    with torch.no_grad():
        z_after = model2.encode(batch)
    assert torch.allclose(z_before, z_after, atol=1e-6)
