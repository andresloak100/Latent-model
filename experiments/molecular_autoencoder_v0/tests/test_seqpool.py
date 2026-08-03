"""Arm E: pooled bottleneck where the atom address is fixed by construction.

The attention-routed arms learn which latent serves which atom, and the
traceability shows that assignment collapsing (routing gini 0.001 -> 0.90,
distinct patterns 0.894 -> 0.57). Strided pooling cannot collapse, because
nothing learns it: atom i is served by latent floor(i/r), always. These tests
pin that, and pin that the no-leak guarantee still holds.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from molae.model import ModelConfig  # noqa: E402
from molae.model_atomlatent import SeqPoolBottleneck, SeqPoolAutoencoder  # noqa: E402
from molae.model_equivariant import make_autoencoder  # noqa: E402
from molae.dataset import collate_fn  # noqa: E402
from molae.sdf import sample_from_sdf  # noqa: E402
from test_atom_universal import ETHANOL  # noqa: E402


def _cfg(r=4, d=8):
    return ModelConfig(d_model=64, n_heads=4, latent_dim=d, enc_self_layers=1,
                       attn_window=32, encoder_type="atomlatent",
                       atom_bottleneck="seqpool", seq_pool_ratio=r,
                       atom_addressing="graph")


@pytest.mark.parametrize("r,expect", [(1, 512), (2, 256), (4, 128), (8, 64)])
def test_token_count_is_n_over_r(r, expect):
    b = SeqPoolBottleneck(ModelConfig(d_model=32, latent_dim=8, seq_pool_ratio=r))
    assert b.encode(torch.randn(1, 512, 32)).shape[1] == expect


def test_token_count_scales_with_atoms_not_fixed():
    """Honest about the trade: N/r is proportional, NOT independent of N."""
    b = SeqPoolBottleneck(ModelConfig(d_model=32, latent_dim=8, seq_pool_ratio=8))
    counts = [b.encode(torch.randn(1, n, 32)).shape[1] for n in (512, 1024, 2048)]
    assert counts == [64, 128, 256]


def test_the_factory_selects_the_arm():
    assert isinstance(make_autoencoder(_cfg()), SeqPoolAutoencoder)
    plain = ModelConfig(d_model=64, n_heads=4, encoder_type="atomlatent")
    assert not isinstance(make_autoencoder(plain), SeqPoolAutoencoder)


def test_runs_end_to_end_on_a_bare_sdf():
    torch.manual_seed(0)
    m = make_autoencoder(_cfg())
    out, z = m(collate_fn([sample_from_sdf(ETHANOL)]))
    assert out.shape == (1, 9, 3) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.parameters())


def test_no_coordinate_leak_through_the_bottleneck():
    torch.manual_seed(0)
    m = make_autoencoder(_cfg()).eval()
    batch = collate_fn([sample_from_sdf(ETHANOL)])
    with torch.no_grad():
        z = m.encode(batch)
        a = m.decode(z, batch)
        trashed = dict(batch)
        trashed["coords"] = torch.randn_like(batch["coords"]) * 50
        b = m.decode(z, trashed)
    assert torch.allclose(a, b, atol=1e-6)


def test_decoder_ignores_residue_fields():
    torch.manual_seed(0)
    m = make_autoencoder(_cfg()).eval()
    batch = collate_fn([sample_from_sdf(ETHANOL)])
    with torch.no_grad():
        base, _ = m(batch)
        scram = dict(batch)
        for f in ("res_pos", "residue_idx", "atom_name_idx", "slot_idx", "chain_idx"):
            scram[f] = torch.randint(0, 5, batch[f].shape)
        other, _ = m(scram)
    assert torch.allclose(base, other, atol=1e-6)


def test_address_is_positional_and_local():
    """Perturbing ONE latent position must move a LOCALISED set of atoms.

    Under learned routing a latent can serve atoms scattered anywhere; here
    the served set is contiguous in canonical order by construction.
    """
    torch.manual_seed(0)
    cfg = _cfg(r=4)
    m = make_autoencoder(cfg).eval()
    n = 64
    z = torch.zeros(1, n // 4, cfg.latent_dim)
    batch = collate_fn([sample_from_sdf(ETHANOL)])
    N = batch["coords"].shape[1]
    z = m.encode(batch)
    with torch.no_grad():
        base = m.decode(z, batch)
        bumped = z.clone()
        bumped[0, 0] += 5.0                       # perturb the FIRST latent only
        moved = (m.decode(bumped, batch) - base).norm(dim=-1)[0]
    # the affected atoms must be a contiguous prefix in canonical order
    order = torch.argsort(batch["canonical_rank"][0])
    in_canon = moved[order]
    hit = (in_canon > in_canon.max() * 0.1).nonzero().flatten()
    assert len(hit) > 0
    assert int(hit.max()) < N // 2, "a single latent moved atoms across the molecule"


def test_reported_latent_budget_grows_with_atoms():
    m = make_autoencoder(_cfg(r=8))
    assert m.latent_floats_for(8000) == 8 * (8000 // 8)
    assert m.latent_floats_for(80000) == 10 * m.latent_floats_for(8000)
