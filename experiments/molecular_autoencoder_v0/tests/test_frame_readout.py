"""Frame readout: rigid per-residue placement instead of 14 absolute coords.

The plain head emits a residue's placement once per atom, at the same output
scale as its 1.5A bond lengths, so placement error deforms internal geometry.
These tests pin the properties that make the frame version a fair swap: the
rotation really is a rotation, placement really is rigid, and the flag is off
by default.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from molae.model import ModelConfig  # noqa: E402
from molae.model_direct import DirectAutoencoder, rot_from_6d  # noqa: E402
from molae.dataset import collate_fn  # noqa: E402


def _sample(n_res=5, atoms_per_res=4):
    res_pos, chain_idx = [], []
    for r in range(n_res):
        res_pos += [r] * atoms_per_res
        chain_idx += [0] * atoms_per_res
    n = len(res_pos)
    return {
        "n_atoms": n, "pdb_id": "TEST", "chain_id": "A",
        "element_symbol": ["C"] * n,
        "coords": torch.randn(n, 3),
        "res_pos": torch.tensor(res_pos),
        "chain_idx": torch.tensor(chain_idx),
        "residue_idx": torch.zeros(n, dtype=torch.long),
        "element_idx": torch.zeros(n, dtype=torch.long),
        "atom_name_idx": torch.zeros(n, dtype=torch.long),
        "slot_idx": torch.arange(n) % atoms_per_res,
        "res_seq": torch.tensor(res_pos),
        "bonds": torch.zeros((0, 2), dtype=torch.long),
        "chirality_centers": torch.zeros((0, 4), dtype=torch.long),
    }


def _cfg(frames=False):
    return ModelConfig(d_model=32, n_heads=2, latent_dim=4, enc_self_layers=1,
                       dec_self_layers=1, max_res_pos=64, max_chains=4,
                       use_slot_emb=True, encoder_type="direct",
                       dec_frames=frames, local_scale=5.0)


def test_rot_from_6d_returns_proper_rotations():
    torch.manual_seed(0)
    R = rot_from_6d(torch.randn(7, 6))
    eye = torch.eye(3).expand(7, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(7), atol=1e-5)


def test_rot_from_6d_is_insensitive_to_input_magnitude():
    """Gram-Schmidt normalises, so only the DIRECTIONS carry information."""
    x = torch.randn(4, 6)
    assert torch.allclose(rot_from_6d(x), rot_from_6d(x * 7.5), atol=1e-5)


def test_rot_from_6d_recovers_a_known_rotation():
    # Two orthonormal columns of a 90-degree rotation about z.
    x = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])
    expected = torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    assert torch.allclose(rot_from_6d(x), expected, atol=1e-6)


def test_rotation_head_starts_at_identity():
    """A frame model must begin as a pure translate-the-local-cloud model."""
    m = DirectAutoencoder(_cfg(True))
    h = torch.randn(2, 5, 32)
    R = rot_from_6d(m.decoder.rot_head(h))
    assert torch.allclose(R, torch.eye(3).expand(2, 5, 3, 3), atol=1e-5)


def test_flag_defaults_off_and_adds_no_modules():
    assert ModelConfig().dec_frames is False
    m = DirectAutoencoder(_cfg(False))
    assert not hasattr(m.decoder, "trans_head")


def test_frame_model_runs_and_is_finite():
    torch.manual_seed(0)
    m = DirectAutoencoder(_cfg(True))
    batch = collate_fn([_sample()])
    out, z = m(batch)
    assert out.shape == batch["coords"].shape
    assert torch.isfinite(out).all()


def test_placement_is_rigid_within_a_residue():
    """The property the plain head cannot offer.

    Changing only the translation output must move every atom of a residue by
    the SAME vector, leaving all intra-residue distances untouched. In the
    plain head placement is re-emitted per atom, so nothing enforces this.
    """
    torch.manual_seed(0)
    m = DirectAutoencoder(_cfg(True)).eval()
    batch = collate_fn([_sample()])
    with torch.no_grad():
        before, _ = m(batch)
        m.decoder.trans_head.bias.add_(torch.tensor([1.0, -2.0, 0.5]))
        after, _ = m(batch)
    shift = after - before
    # Same shift for every atom, so every pairwise distance is preserved.
    assert torch.allclose(shift, shift[0, 0].expand_as(shift), atol=1e-4)
    d0 = torch.cdist(before[0], before[0])
    d1 = torch.cdist(after[0], after[0])
    assert torch.allclose(d0, d1, atol=1e-4)


def test_local_and_placement_scales_are_independent():
    """Two output scales, separately controllable -- the whole point.

    The plain head has one: it must express a ~50A placement and a 1.5A bond
    length through the same linear layer at the same scale.
    """
    torch.manual_seed(0)
    batch = collate_fn([_sample()])

    def coords_at(local_scale, coord_scale):
        # Decoder only, from a FIXED latent: coord_scale also normalises the
        # encoder's input, so a full forward pass would confound the two.
        torch.manual_seed(0)
        cfg = _cfg(True)
        cfg.local_scale, cfg.coord_scale = local_scale, coord_scale
        m = DirectAutoencoder(cfg).eval()
        z = torch.full((1, 5, cfg.latent_dim), 0.3)
        with torch.no_grad():
            return m.decode(z, batch)[0]

    base = coords_at(5.0, 10.0)
    # Doubling local_scale doubles the residue's internal spread...
    wide = coords_at(10.0, 10.0)
    spread = lambda c: float(torch.cdist(c[:4], c[:4]).max())   # noqa: E731
    assert spread(wide) == pytest.approx(2 * spread(base), rel=1e-3)
    # ...while leaving the residue's placement alone.
    far = coords_at(5.0, 20.0)
    assert spread(far) == pytest.approx(spread(base), rel=1e-3)
    assert float(far.mean(0).norm()) > float(base.mean(0).norm())


def test_frames_and_chain_awareness_compose():
    cfg = _cfg(True)
    cfg.dec_chain_aware = True
    m = DirectAutoencoder(cfg)
    out, _ = m(collate_fn([_sample()]))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("frames", [False, True])
def test_gradients_reach_the_decoder(frames):
    torch.manual_seed(0)
    m = DirectAutoencoder(_cfg(frames))
    out, _ = m(collate_fn([_sample()]))
    out.square().mean().backward()
    grads = [p.grad for p in m.decoder.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
