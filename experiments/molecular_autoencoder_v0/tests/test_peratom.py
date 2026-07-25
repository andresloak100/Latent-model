"""Per-atom latent autoencoder: routing, generality, and the compression cap."""

import pytest
import torch

from molae.synthetic import make_synthetic_ala
from molae.dataset import sample_from_arrays, collate_fn
from molae.model import ModelConfig
from molae.model_equivariant import make_autoencoder
from molae.model_peratom import PerAtomAutoencoder


def _cfg(latent_dim=2, encoder_type="peratom"):
    return ModelConfig(d_model=32, n_heads=2, latent_dim=latent_dim,
                       enc_self_layers=1, dec_self_layers=1,
                       encoder_type=encoder_type)


def _batch():
    a = sample_from_arrays(make_synthetic_ala(n_res=3))   # 15 atoms
    b = sample_from_arrays(make_synthetic_ala(n_res=5))   # 25 atoms
    return collate_fn([a, b])


@pytest.mark.parametrize("etype", ["peratom", "peratom_elem"])
def test_factory_and_shapes(etype):
    model = make_autoencoder(_cfg(encoder_type=etype))
    assert isinstance(model, PerAtomAutoencoder)
    batch = _batch()
    preds, z = model(batch)
    assert preds.shape == (2, 25, 3)
    assert z.shape == (2, 25, 2)          # one latent per atom, dim 2


def test_element_only_arm_uses_no_protein_identity():
    """peratom_elem must not read residue type / atom name / residue index.

    Those three inputs are the protein assumptions. Scrambling them must leave
    the output bit-identical, which is what makes this arm transferable to
    ligands, solvent and arbitrary molecules.
    """
    model = make_autoencoder(_cfg(encoder_type="peratom_elem")).eval()
    batch = _batch()
    with torch.no_grad():
        out1, _ = model(batch)
        scrambled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        scrambled["residue_idx"] = torch.zeros_like(scrambled["residue_idx"])
        scrambled["atom_name_idx"] = torch.zeros_like(scrambled["atom_name_idx"])
        scrambled["res_pos"] = torch.zeros_like(scrambled["res_pos"])
        out2, _ = model(scrambled)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_protein_featurised_arm_does_read_identity():
    """Control for the test above: the plain `peratom` arm should NOT be blind
    to identity, otherwise the two arms would not be testing different things."""
    model = make_autoencoder(_cfg(encoder_type="peratom")).eval()
    batch = _batch()
    with torch.no_grad():
        out1, _ = model(batch)
        scrambled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        scrambled["atom_name_idx"] = torch.zeros_like(scrambled["atom_name_idx"])
        out2, _ = model(scrambled)
    assert not torch.allclose(out1, out2, atol=1e-6)


def test_permutation_equivariance():
    """Permuting the atom list permutes the outputs identically.

    This is what removes the need for a canonical atom ordering -- the property
    the per-residue slot bank had to buy with (res_pos, atom_name) indexing.
    """
    model = make_autoencoder(_cfg(encoder_type="peratom_elem")).eval()
    single = collate_fn([sample_from_arrays(make_synthetic_ala(n_res=4))])
    n = int(single["n_atoms"][0])
    perm = torch.randperm(n)
    permuted = {k: (v[:, perm] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == n else v)
                for k, v in single.items()}
    with torch.no_grad():
        out, _ = model(single)
        out_p, _ = model(permuted)
    assert torch.allclose(out[:, perm], out_p, atol=1e-5)


def test_padding_does_not_leak_into_latent():
    model = make_autoencoder(_cfg()).eval()
    batch = _batch()
    with torch.no_grad():
        z1 = model.encode(batch)
        corrupted = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        corrupted["coords"][0, 15:] += 100.0
        corrupted["element_idx"][0, 15:] = 2
        z2 = model.encode(corrupted)
    assert torch.allclose(z1[0, :15], z2[0, :15], atol=1e-5)


def test_latent_size_scales_with_atoms_not_residues():
    model = make_autoencoder(_cfg(latent_dim=2))
    assert model.latent_floats == 2
    assert model.latent_floats_for_atoms(600) == 1200
    # Asking for a residue-based size must fail loudly rather than silently
    # under-report the latent (the bug that inflated per-residue compression).
    with pytest.raises(TypeError):
        model.latent_floats_for(78)


@pytest.mark.parametrize("d,expected", [(1, 3.0), (2, 1.5), (3, 1.0)])
def test_compression_ceiling_is_three_over_d(d, expected):
    """Per-atom compression is exactly 3/d, independent of system size.

    Pins the ceiling documented in model_peratom.py: d>=3 is not compression at
    all, so only latent_dim 1-2 gives a meaningful experiment.
    """
    model = make_autoencoder(_cfg(latent_dim=d))
    for n_atoms in (50, 607, 5000):
        ratio = (3 * n_atoms) / model.latent_floats_for_atoms(n_atoms)
        assert ratio == pytest.approx(expected)
