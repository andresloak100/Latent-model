"""Masking / batching / variable-size tests."""

import torch

from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn
from molae.model import MolecularAutoencoder, ModelConfig


def _samples():
    a = sample_from_arrays(make_synthetic_ala(n_res=3))   # 15 atoms
    b = sample_from_arrays(make_synthetic_ala(n_res=5))   # 25 atoms
    return [a, b]


def test_collate_padding_and_mask():
    batch = collate_fn(_samples())
    assert batch["coords"].shape == (2, 25, 3)
    assert batch["mask"].shape == (2, 25)
    assert batch["mask"][0].sum().item() == 15   # first structure has 15 real atoms
    assert batch["mask"][1].sum().item() == 25
    # padded region is zero
    assert torch.allclose(batch["coords"][0, 15:], torch.zeros(10, 3))
    assert batch["n_atoms"].tolist() == [15, 25]


def test_model_variable_size_forward():
    cfg = ModelConfig(d_model=32, n_heads=2, n_latent_tokens=8, latent_dim=4,
                      enc_self_layers=1, dec_self_layers=1)
    model = MolecularAutoencoder(cfg)
    batch = collate_fn(_samples())
    preds, z = model(batch)
    assert preds.shape == (2, 25, 3)
    assert z.shape == (2, 8, 4)      # latent independent of atom count


def test_padding_does_not_leak_into_latent():
    """Changing only the padded region must not change the encoded latent."""
    cfg = ModelConfig(d_model=32, n_heads=2, n_latent_tokens=8, latent_dim=4,
                      enc_self_layers=1, dec_self_layers=1)
    model = MolecularAutoencoder(cfg)
    model.eval()
    batch = collate_fn(_samples())
    with torch.no_grad():
        z1 = model.encode(batch)
        # corrupt the padded atoms of the first (shorter) structure
        batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch2["coords"][0, 15:] += 100.0
        batch2["element_idx"][0, 15:] = 2
        z2 = model.encode(batch2)
    assert torch.allclose(z1[0], z2[0], atol=1e-5)


def test_random_rotate_preserves_geometry_and_is_proper():
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
    from train import random_rotate
    coords = torch.randn(4, 20, 3)
    rot = random_rotate(coords)
    assert rot.shape == coords.shape
    # rotation preserves pairwise distances (rigid)
    d0 = torch.cdist(coords, coords)
    d1 = torch.cdist(rot, rot)
    assert torch.allclose(d0, d1, atol=1e-4)
    # centroid-relative norms preserved
    assert torch.allclose(coords.norm(dim=-1), rot.norm(dim=-1), atol=1e-4)
