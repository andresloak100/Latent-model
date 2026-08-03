"""Atom tokens -> arbitrary L shared latents -> atom outputs.

The requirements this arm exists to satisfy are all falsifiable, so each one
gets a test that would fail if the property were quietly lost:

  * one token per atom through the encoder (no pooling to groups)
  * L independent of atom count, at runtime
  * anchors stored inside the latent, counted in its budget
  * NO input geometry reaching the decoder -- the property that makes the
    latent worth diffusing in, and the easiest one to break by accident
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from molae.model import ModelConfig  # noqa: E402
from molae.model_atomlatent import (  # noqa: E402
    AtomLatentAutoencoder, topk_latents,
)
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.dataset import collate_fn  # noqa: E402


def _sample(n_res=12, apr=8, scale=5.0):
    rp = []
    for r in range(n_res):
        rp += [r] * apr
    n = len(rp)
    return {
        "n_atoms": n, "pdb_id": "X", "chain_id": "A",
        "element_symbol": ["C"] * n,
        "coords": torch.randn(n, 3) * scale,
        "res_pos": torch.tensor(rp),
        "chain_idx": torch.zeros(n, dtype=torch.long),
        "residue_idx": torch.zeros(n, dtype=torch.long),
        "element_idx": torch.zeros(n, dtype=torch.long),
        "atom_name_idx": torch.zeros(n, dtype=torch.long),
        "slot_idx": torch.arange(n) % apr,
        "res_seq": torch.tensor(rp),
        "bonds": torch.zeros((0, 2), dtype=torch.long),
        "chirality_centers": torch.zeros((0, 4), dtype=torch.long),
    }


def _cfg(local=False, L=32, d=16):
    return ModelConfig(d_model=64, n_heads=4, latent_dim=d, n_latent_tokens=L,
                       enc_self_layers=1, dec_cross_layers=2, dec_local_layers=1,
                       encoder_type="atomlatent", dec_local_cross=local,
                       dec_local_k=8, use_slot_emb=True, max_chains=4,
                       attn_window=32)


# ------------------------------------------------- latent count independence

@pytest.mark.parametrize("L", [16, 32, 64, 128, 256])
def test_latent_count_is_a_free_runtime_argument(L):
    m = AtomLatentAutoencoder(_cfg())
    z = m.encode(collate_fn([_sample()]), n_latents=L)
    assert z.shape[1] == L


def test_same_model_same_L_across_different_atom_counts():
    """The property no previous arm had: L does not follow the structure."""
    m = AtomLatentAutoencoder(_cfg(L=32))
    for n_res in (4, 12, 40):
        z = m.encode(collate_fn([_sample(n_res=n_res)]))
        assert z.shape[1] == 32, n_res


def test_reported_latent_budget_does_not_depend_on_structure_size():
    m = AtomLatentAutoencoder(_cfg(L=32, d=16))
    assert m.latent_floats_for(10) == m.latent_floats_for(100_000) == 32 * 16


# ------------------------------------------------------------- no atom pooling

def test_encoder_keeps_one_token_per_atom():
    """Perturbing ONE atom must change the latent.

    Mean-pooling 8 atoms into a group would wash out a single-atom change;
    keeping a token per atom does not.
    """
    torch.manual_seed(0)
    m = AtomLatentAutoencoder(_cfg()).eval()
    s = _sample()
    b1 = collate_fn([s])
    s2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
    s2["coords"][3] += 2.0
    b2 = collate_fn([s2])
    with torch.no_grad():
        assert not torch.allclose(m.encode(b1), m.encode(b2), atol=1e-5)


# ------------------------------------------------------------------- anchors

def test_anchors_live_inside_the_latent_and_are_real_positions():
    torch.manual_seed(0)
    m = AtomLatentAutoencoder(_cfg(d=16)).eval()
    batch = collate_fn([_sample(scale=10.0)])
    with torch.no_grad():
        z = m.encode(batch)
    anchors = z[..., :3] * m.cfg.coord_scale
    coords = batch["coords"]
    # Anchors are attention-weighted centroids of real atoms, so they sit
    # inside the structure's bounding box rather than anywhere in R^3.
    assert bool((anchors.amax() <= coords.amax() + 1e-3).all())
    assert bool((anchors.amin() >= coords.amin() - 1e-3).all())


def test_anchor_channels_are_counted_in_the_latent_budget():
    """d channels total, 3 of them anchor -- not d + 3 smuggled through."""
    for d in (8, 16, 32):
        m = AtomLatentAutoencoder(_cfg(d=d))
        assert m.encode(collate_fn([_sample()])).shape[-1] == d


def test_topk_latents_picks_the_nearest_anchors():
    coords = torch.tensor([[[0.0, 0.0, 0.0]]])
    anchors = torch.tensor([[[9.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]]])
    assert topk_latents(coords, anchors, 2)[0, 0].tolist() == [1, 2]


# --------------------------------------------------- NO content skips (key)

def test_decoder_never_sees_input_coordinates():
    """The requirement that makes the latent worth diffusing in.

    Decode the SAME latent against a batch whose coordinates have been
    replaced with garbage. If any geometry reached the decoder outside the
    latent, the output would move.
    """
    torch.manual_seed(0)
    for local in (False, True):
        m = AtomLatentAutoencoder(_cfg(local=local)).eval()
        batch = collate_fn([_sample()])
        with torch.no_grad():
            z = m.encode(batch)
            a = m.decode(z, batch)
            trashed = dict(batch)
            trashed["coords"] = torch.randn_like(batch["coords"]) * 50
            b = m.decode(z, trashed)
        assert torch.allclose(a, b, atol=1e-6), f"geometry leaked (local={local})"


def test_encoder_output_depends_on_coordinates():
    """Control for the test above: the ENCODER must see geometry."""
    torch.manual_seed(0)
    m = AtomLatentAutoencoder(_cfg()).eval()
    batch = collate_fn([_sample()])
    trashed = dict(batch)
    trashed["coords"] = torch.randn_like(batch["coords"]) * 50
    with torch.no_grad():
        assert not torch.allclose(m.encode(batch), m.encode(trashed), atol=1e-4)


# ------------------------------------------------------------------- shapes

@pytest.mark.parametrize("local", [False, True])
def test_forward_shapes_and_gradients(local):
    m = AtomLatentAutoencoder(_cfg(local=local))
    batch = collate_fn([_sample(), _sample(n_res=6)])
    out, z = m(batch)
    assert out.shape == batch["coords"].shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_local_arm_adds_a_refinement_over_the_global_pass():
    """Arm C must actually differ from arm B, or the A/B is empty."""
    torch.manual_seed(0)
    m = AtomLatentAutoencoder(_cfg(local=True)).eval()
    batch = collate_fn([_sample()])
    with torch.no_grad():
        fine, _ = m(batch)
    assert not torch.allclose(fine, m.last_coarse, atol=1e-6)


def test_registered_in_the_model_factory():
    m = make_autoencoder(_cfg())
    assert isinstance(m, AtomLatentAutoencoder)
    assert m.per_residue is False


def test_padding_is_excluded_from_anchor_pooling():
    """Ragged batches must not drag anchors toward the origin via padding."""
    torch.manual_seed(0)
    m = AtomLatentAutoencoder(_cfg()).eval()
    big = _sample(n_res=20, scale=10.0)
    with torch.no_grad():
        solo = m.encode(collate_fn([big]))
        padded = m.encode(collate_fn([big, _sample(n_res=3, scale=10.0)]))[:1]
    assert torch.allclose(solo, padded, atol=1e-4)
