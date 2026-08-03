"""Farthest-point anchor seeding: coverage by construction, not by hope.

Attention-pooled anchors are weighted centroids, and at initialisation the
attention is uniform, so every anchor lands on the centroid of all atoms --
measured spread/Rg = 0.000, mean atom-to-nearest-anchor = 0.998 Rg. The
k-nearest routing is degenerate from step 0 and the trained cells still sit at
0.015-0.021, so it does not recover on its own.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from molae.model import ModelConfig  # noqa: E402
from molae.model_atomlatent import (  # noqa: E402
    AtomLatentAutoencoder, farthest_point_anchors,
)


def _cloud(B=2, N=400, pad_from=None, scale=10.0):
    torch.manual_seed(0)
    coords = torch.randn(B, N, 3) * scale
    mask = torch.ones(B, N)
    if pad_from is not None:
        mask[1, pad_from:] = 0
    return coords, mask


@pytest.mark.parametrize("L", [8, 32, 128])
def test_anchors_cover_the_structure(L):
    coords, mask = _cloud()
    a = farthest_point_anchors(coords, mask, L)
    rg = torch.linalg.norm(coords[0] - coords[0].mean(0), dim=-1).mean()
    spread = torch.linalg.norm(a[0] - a[0].mean(0), dim=-1).mean()
    assert float(spread / rg) > 0.5, "anchors collapsed, which is the bug"


def test_coverage_improves_with_more_anchors():
    """More latents must mean atoms are closer to one -- otherwise L is inert."""
    coords, mask = _cloud()
    d = []
    for L in (16, 64, 128):
        a = farthest_point_anchors(coords, mask, L)
        d.append(float(torch.cdist(coords[0], a[0]).min(1).values.mean()))
    assert d[0] > d[1] > d[2]


def test_padding_is_never_selected():
    coords, mask = _cloud(pad_from=300)
    a = farthest_point_anchors(coords, mask, 64)
    assert bool((torch.cdist(a[1], coords[1, 300:]).min() > 1e-6).all())


def test_anchors_are_real_atom_positions():
    """Direct norm, not cdist: cdist expands ||a||^2+||b||^2-2a.b, which
    cancels badly at coord scale ~10 and reports 1.6e-2 for an EXACT match."""
    coords, mask = _cloud()
    a = farthest_point_anchors(coords, mask, 32)
    worst = max(float((a[0][i] - coords[0]).norm(dim=-1).min()) for i in range(32))
    assert worst < 1e-5


def test_deterministic():
    coords, mask = _cloud()
    a = farthest_point_anchors(coords, mask, 32)
    b = farthest_point_anchors(coords, mask, 32)
    assert torch.equal(a, b)


def _cfg(fps):
    return ModelConfig(d_model=64, n_heads=4, latent_dim=16, n_latent_tokens=32,
                       enc_self_layers=1, dec_cross_layers=2, dec_local_layers=1,
                       encoder_type="atomlatent", dec_local_cross=True,
                       attn_window=32, fps_anchors=fps)


def _sample(n_res=20, apr=8):
    rp = []
    for r in range(n_res):
        rp += [r] * apr
    n = len(rp)
    return {"n_atoms": n, "pdb_id": "X", "chain_id": "A",
            "element_symbol": ["C"] * n, "coords": torch.randn(n, 3) * 10,
            "res_pos": torch.tensor(rp), "chain_idx": torch.zeros(n, dtype=torch.long),
            "residue_idx": torch.zeros(n, dtype=torch.long),
            "element_idx": torch.zeros(n, dtype=torch.long),
            "atom_name_idx": torch.zeros(n, dtype=torch.long),
            "slot_idx": torch.arange(n) % apr, "res_seq": torch.tensor(rp),
            "bonds": torch.zeros((0, 2), dtype=torch.long),
            "chirality_centers": torch.zeros((0, 4), dtype=torch.long)}


def test_model_anchors_are_spread_at_init_with_fps_and_collapsed_without():
    """The A/B, end to end through the real model."""
    from molae.dataset import collate_fn
    batch = collate_fn([_sample()])
    coords = batch["coords"][0]
    rg = float(torch.linalg.norm(coords - coords.mean(0), dim=-1).mean())
    out = {}
    for fps in (False, True):
        torch.manual_seed(0)
        m = AtomLatentAutoencoder(_cfg(fps)).eval()
        with torch.no_grad():
            z = m.encode(batch)
        a = z[0, :, :3] * m.cfg.coord_scale
        out[fps] = float(torch.linalg.norm(a - a.mean(0), dim=-1).mean()) / rg
    assert out[False] < 0.05, "expected the attention-pooled anchors to collapse"
    assert out[True] > 0.5, "fps anchors must cover the structure"


def test_fps_still_lets_attention_contribute():
    """Seeded, not frozen: the learned residual must still be wired in."""
    m = AtomLatentAutoencoder(_cfg(True))
    assert m.encoder.pool.anchor_scale is not None
    assert m.encoder.pool.anchor_scale.requires_grad
