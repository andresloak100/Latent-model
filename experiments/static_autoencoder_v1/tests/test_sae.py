"""Tests for the scaling harness. Each one pins a defect the self-test actually found."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae.data import SyntheticStructures, center_and_scale, random_rotations
from sae.metrics import (active_dims, effective_rank, latent_spectrum, linear_probe,
                         masked_mse, rmsd_unaligned)
from sae.model import AEConfig, StaticAutoencoder, config_for_params, n_params
from sae.scaling import fit_scaling, power_law
from sae.train import TrainConfig, run


# ---- model -----------------------------------------------------------------------------

def test_shapes_and_masking():
    cfg = AEConfig(d_model=32, n_heads=2, bottleneck="latent_queries",
                   n_latent_tokens=2, latent_width=8, max_atoms=64)
    m = StaticAutoencoder(cfg)
    x, e = torch.randn(3, 20, 3), torch.randint(0, 16, (3, 20))
    mask = torch.ones(3, 20, dtype=torch.bool); mask[2, 15:] = False
    rec, z, logits = m(x, e, mask)
    assert rec.shape == (3, 20, 3)
    assert z.shape == (3, 2, 8) and cfg.latent_size == 16
    assert logits is None


def test_predict_composition_head():
    cfg = AEConfig(d_model=32, n_heads=2, latent_width=8, predict_composition=True, n_elements=7)
    _, _, logits = StaticAutoencoder(cfg)(torch.randn(2, 10, 3),
                                          torch.randint(0, 7, (2, 10)),
                                          torch.ones(2, 10, dtype=torch.bool))
    assert logits.shape == (2, 10, 7)


def test_param_solver_hits_target():
    for target in (300_000, 1_000_000):
        c = config_for_params(target, AEConfig(latent_width=64))
        n = n_params(StaticAutoencoder(c))
        assert abs(n - target) / target < 0.15, (target, n)
        assert c.d_model % c.n_heads == 0


def test_latent_is_the_only_path_from_input_to_output():
    """Perturbing the input must not change the output except through the bottleneck."""
    cfg = AEConfig(d_model=32, n_heads=2, latent_width=4, max_atoms=64)
    m = StaticAutoencoder(cfg).eval()
    e = torch.randint(0, 16, (1, 12)); mask = torch.ones(1, 12, dtype=torch.bool)
    with torch.no_grad():
        z = m.encode(torch.randn(1, 12, 3), e, mask)
        a, _ = m.decode(z, e, mask)
        b, _ = m.decode(z, e, mask)
        assert torch.allclose(a, b)                      # decode is a function of z alone
        c, _ = m.decode(z + 1.0, e, mask)
        assert not torch.allclose(a, c)                  # and it actually depends on z


# ---- data ------------------------------------------------------------------------------

def test_rotations_are_proper():
    R = random_rotations(32)
    assert torch.allclose(torch.det(R), torch.ones(32), atol=1e-4)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(32, 3, 3), atol=1e-4)


def test_centering_is_exact_and_respects_the_mask():
    ds = SyntheticStructures(n_structures=8, n_atoms=10, intrinsic_dim=3, seed=1)
    b = ds.batch(np.arange(8))
    b.mask[:, 7:] = False
    x = center_and_scale(b.xyz, b.mask, scale=10.0)
    m = b.mask[..., None].float()
    assert torch.allclose((x * m).sum(1), torch.zeros(8, 3), atol=1e-4)
    assert torch.allclose(x[~b.mask], torch.zeros(3), atol=0)   # padding stays zero


def test_synthetic_has_the_intrinsic_dimension_it_claims():
    """If this fails, every saturation reading from the self-test is meaningless."""
    ds = SyntheticStructures(n_structures=600, n_atoms=20, intrinsic_dim=5, seed=2)
    Z = ds.batch(np.arange(600)).xyz.reshape(600, -1).numpy()
    w = latent_spectrum(Z)
    assert (w[:5] > 1e-6).all()
    assert w[5:].sum() / w.sum() < 1e-6, "variance leaked outside the declared basis"
    assert 4.0 < effective_rank(Z) < 6.0


def test_real_corpus_refuses_to_pretend():
    from sae.data import RealCorpus
    with pytest.raises(NotImplementedError):
        RealCorpus()


# ---- metrics ---------------------------------------------------------------------------

def test_masked_mse_ignores_padding():
    rec, xyz = torch.zeros(1, 4, 3), torch.zeros(1, 4, 3)
    xyz[0, 3] = 1e6
    mask = torch.tensor([[True, True, True, False]])
    assert masked_mse(rec, xyz, mask) == 0.0
    assert rmsd_unaligned(rec, xyz, mask) == 0.0


def test_effective_rank_and_active_dims_on_a_known_spectrum():
    rng = np.random.default_rng(0)
    Z = np.zeros((4000, 20))
    Z[:, :4] = rng.normal(size=(4000, 4))
    assert 3.5 < effective_rank(Z) < 4.5
    assert active_dims(Z) == 4


def test_linear_probe_recovers_a_linear_target_and_not_a_scrambled_one():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(800, 12))
    y = Z @ rng.normal(size=12)
    assert linear_probe(Z, y) > 0.99
    assert linear_probe(Z, rng.permutation(y)) < 0.2


# ---- scaling fit ------------------------------------------------------------------------

def test_fit_recovers_known_alpha_and_floor():
    N = np.array([5e5, 2e6, 8e6, 3.2e7, 1.28e8])
    f = fit_scaling(N, power_law(N, 50.0, 0.35, 0.01), n_boot=200)
    assert f.converged and abs(f.alpha - 0.35) < 0.05 and abs(f.L_inf - 0.01) < 0.01


def test_fit_flags_the_degenerate_case_instead_of_reporting_a_wrong_floor():
    """Truth (alpha=0.9, L_inf=0.40) on an already-saturated grid: the power-law term is below
    the noise, A and L_inf trade off freely, and the fit used to report L_inf = 0.0000 with a
    tight interval. It must now refuse rather than answer."""
    rng = np.random.default_rng(0)
    N = np.array([5e5, 2e6, 8e6, 3.2e7, 1.28e8])
    L = power_law(N, 5.0, 0.9, 0.40) * (1 + rng.normal(0, 0.02, len(N)))
    f = fit_scaling(N, L, n_boot=300)
    assert f.degenerate and "not separately identifiable" in f.note.lower().replace("  ", " ")
    assert abs(f.L_extrap_10x - 0.40) < 0.05


def test_fit_refuses_too_few_points():
    f = fit_scaling([1e5, 1e6, 1e7], [1.0, 0.5, 0.3])
    assert not f.converged and "not identifiable" in f.note


# ---- the convergence guard ----------------------------------------------------------------

@pytest.mark.parametrize("steps,expected", [(200, "stuck"), (2500, "still_improving")])
def test_guard_separates_stuck_from_undertrained(steps, expected):
    """A model that never learns and a model still improving are BOTH flat-ish, and both would
    be read as saturation. The guard must name which."""
    ds = SyntheticStructures(n_structures=384, n_atoms=12, intrinsic_dim=5, n_elements=4, seed=0)
    cfg = config_for_params(120_000, replace(AEConfig(max_atoms=1024), latent_width=16))
    r = run(cfg, TrainConfig(steps=steps, lr=1e-3, seed=0, eval_batches=3), ds)
    assert r.status == expected, (r.status, r.frac_var_explained, r.tail_improvement)
    assert not r.converged


# ---- the merged bottleneck ---------------------------------------------------------------

def test_mean_pool_is_the_default_and_ignores_padding():
    """A plain h.mean(1) would halve the latent of a half-padded structure, which reads as a
    smaller molecule rather than a shorter array."""
    cfg = AEConfig(d_model=32, n_heads=2, latent_width=8, max_atoms=64)
    assert cfg.bottleneck == "mean_pool"
    m = StaticAutoencoder(cfg).eval()
    x, e = torch.randn(1, 10, 3), torch.randint(0, 16, (1, 10))
    full = torch.ones(1, 10, dtype=torch.bool)
    xp = torch.cat([x, torch.randn(1, 6, 3) * 100], 1)
    ep = torch.cat([e, torch.randint(0, 16, (1, 6))], 1)
    part = torch.cat([full, torch.zeros(1, 6, dtype=torch.bool)], 1)
    with torch.no_grad():
        assert torch.allclose(m.encode(x, e, full), m.encode(xp, ep, part), atol=1e-5)


def test_mean_pool_rejects_multiple_latent_tokens():
    with pytest.raises(ValueError):
        AEConfig(bottleneck="mean_pool", n_latent_tokens=4)
