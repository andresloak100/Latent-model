"""Per-residue latent AE: latent scales with #residues, pooling is correct, overfits."""

import torch

from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn
from molae.model import ModelConfig
from molae.model_perresidue import (
    PerResidueAutoencoder, scatter_mean_residues, residue_mask, n_residues,
)
from molae.model_equivariant import make_autoencoder
from molae.losses import LossComputer, LossWeights
from molae.alignment import aligned_rmsd_torch
from molae import utils


def _cfg(**kw):
    d = dict(d_model=48, n_heads=2, n_latent_tokens=8, latent_dim=4,
             enc_self_layers=1, dec_self_layers=1)
    d.update(kw)
    return ModelConfig(**d)


def test_scatter_mean_matches_manual():
    # two residues: atoms [0,0,1] -> means per residue
    tokens = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]])  # (1,3,2)
    res_pos = torch.tensor([[0, 0, 1]])
    mask = torch.ones(1, 3)
    rf, rm = scatter_mean_residues(tokens, res_pos, mask, R=2)
    assert torch.allclose(rf[0, 0], torch.tensor([2.0, 2.0]))   # mean of atoms 0,1
    assert torch.allclose(rf[0, 1], torch.tensor([5.0, 5.0]))   # atom 2
    assert rm.tolist() == [[1.0, 1.0]]


def test_pooling_ignores_padding():
    tokens = torch.tensor([[[2.0], [4.0], [999.0]]])   # last atom is padding
    res_pos = torch.tensor([[0, 0, 0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    rf, rm = scatter_mean_residues(tokens, res_pos, mask, R=1)
    assert torch.allclose(rf[0, 0], torch.tensor([3.0]))        # padding excluded


def test_factory_builds_perresidue():
    assert isinstance(make_autoencoder(_cfg(encoder_type="perresidue")),
                      PerResidueAutoencoder)


def test_latent_scales_with_residues():
    model = PerResidueAutoencoder(_cfg())
    for nres in (4, 9):
        batch = collate_fn([sample_from_arrays(make_synthetic_ala(nres))])
        z = model.encode(batch)
        assert z.shape == (1, nres, 4), f"expected {nres} residue latents, got {z.shape}"


def test_perresidue_forward_and_padding_mask():
    model = PerResidueAutoencoder(_cfg())
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(6)),
                        sample_from_arrays(make_synthetic_ala(4))])
    preds, z = model(batch)
    assert preds.shape == (2, 30, 3)
    assert z.shape == (2, 6, 4)   # R = max residues in batch


def test_perresidue_overfits_single_structure():
    utils.set_seed(0)
    model = PerResidueAutoencoder(ModelConfig(d_model=64, n_heads=4, n_latent_tokens=16,
                                              latent_dim=8, enc_self_layers=2, dec_self_layers=2))
    batch = collate_fn([sample_from_arrays(make_synthetic_ala(5))])
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
    assert rmsd < 2.0, f"per-residue AE failed to overfit: RMSD={rmsd:.3f}"
