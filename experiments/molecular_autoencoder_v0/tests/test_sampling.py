"""Sampling and scoring a decoded latent trajectory.

The load-bearing tests are the two failure modes that LOOK LIKE SUCCESS:

  * a collapsed sampler emitting one frame T times scores perfectly on
    physics and on temporal continuity, and only `diversity` catches it;
  * independent samples that happen to be individually plausible are not a
    trajectory, and only the spread-over-step ratio separates them from one.

Both are the same trap as a codec that averages conformers to a mean
structure, which is why the codec gate needed `ensemble_resolution`.
"""

import numpy as np
import torch

from molae.dataset import sample_from_arrays, collate_fn
from molae.latent_video import LatentVideoConfig, LatentVideoDiffusion
from molae.model import ModelConfig
from molae.model_direct import DirectAutoencoder
from molae.sampling import (decode_segment, temporal_continuity, diversity,
                            physics, sample_and_score)
from molae.synthetic import make_synthetic_ala


def _setup(n_res=5, latent_dim=4):
    d = make_synthetic_ala(n_res=n_res)
    sample = sample_from_arrays(d)
    batch = collate_fn([sample])
    cfg = ModelConfig(d_model=32, n_heads=2, latent_dim=latent_dim,
                      enc_self_layers=1, dec_self_layers=1, encoder_type="direct")
    codec = DirectAutoencoder(cfg)
    codec.eval()
    vcfg = LatentVideoConfig(latent_dim=latent_dim, d_model=32, n_heads=4, depth=2,
                             ff_mult=2)
    diff = LatentVideoDiffusion(vcfg)
    return sample, batch, codec, diff


# --- decoding --------------------------------------------------------------

def test_decode_returns_one_frame_per_latent_step():
    sample, batch, codec, _ = _setup(n_res=5)
    R = int(batch["res_pos"].max().item()) + 1
    z = torch.randn(6, R, 4)
    frames = decode_segment(codec, z, batch)
    assert len(frames) == 6
    assert all(f.shape == (sample["n_atoms"], 3) for f in frames)


def test_decode_does_not_leave_the_codec_in_eval():
    """The codec is frozen for this stage but the caller's mode is theirs."""
    _, batch, codec, _ = _setup()
    codec.train()
    R = int(batch["res_pos"].max().item()) + 1
    decode_segment(codec, torch.randn(3, R, 4), batch)
    assert codec.training


# --- the failures that look like success -----------------------------------

def test_collapsed_sampler_scores_perfectly_on_physics_and_continuity():
    """A model emitting ONE frame T times is physically perfect and
    temporally perfect. This is why diversity is measured separately."""
    rng = np.random.default_rng(0)
    one = rng.normal(0, 5, (40, 3))
    frames = [one.copy() for _ in range(8)]
    cont = temporal_continuity(frames)
    div = diversity(frames)
    assert cont["mean_step"] < 1e-9, "collapse looks perfectly continuous"
    assert div["spread"] < 1e-9, "diversity is the only check that catches it"


def test_diversity_separates_a_path_from_independent_samples():
    """Independent draws can each be plausible without being a trajectory.
    A path has spread >> step; independent samples have spread ~= step.

    Frames must DEFORM, not translate: these metrics are Kabsch-aligned, so
    rigid motion is removed by construction and a translating rigid body
    registers as no motion at all. That is correct -- what matters in a
    trajectory is internal conformational change, not tumbling or diffusion.
    """
    rng = np.random.default_rng(1)
    base = rng.normal(0, 5, (40, 3))
    direction = rng.normal(0, 1, base.shape)
    path = [base + 0.35 * t * direction for t in range(10)]     # one coherent mode
    indep = [base + rng.normal(0, 1.6, base.shape) for _ in range(10)]
    assert diversity(path)["spread_over_step"] > 2.5
    assert diversity(indep)["spread_over_step"] < 2.0


def test_jump_ratio_catches_a_single_teleport_the_mean_hides():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 5, (40, 3))
    direction = rng.normal(0, 1, base.shape)
    frames = [base + 0.15 * t * direction for t in range(8)]
    frames[5] = frames[5] + rng.normal(0, 6.0, base.shape)     # one teleport
    c = temporal_continuity(frames)
    assert c["jump_ratio"] > 3.0, "a single jump must not be averaged away"


def test_smooth_trajectory_has_a_low_jump_ratio():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 5, (40, 3))
    direction = rng.normal(0, 1, base.shape)
    frames = [base + 0.3 * t * direction for t in range(8)]
    assert temporal_continuity(frames)["jump_ratio"] < 1.5


def test_rigid_motion_is_not_counted_as_conformational_change():
    """Kabsch-aligned by design: a molecule that tumbles and translates
    without deforming has not moved conformationally, and a trajectory metric
    should say so rather than reporting the diffusion."""
    rng = np.random.default_rng(4)
    base = rng.normal(0, 5, (40, 3))
    q, _ = np.linalg.qr(rng.normal(0, 1, (3, 3)))
    frames = [base @ np.linalg.matrix_power(q, t) + np.array([3.0 * t, 0, 0])
              for t in range(6)]
    assert temporal_continuity(frames)["mean_step"] < 1e-6
    assert diversity(frames)["spread"] < 1e-6


# --- physics ---------------------------------------------------------------

def test_physics_returns_the_three_chemistry_metrics():
    sample, batch, codec, _ = _setup()
    R = int(batch["res_pos"].max().item()) + 1
    frames = decode_segment(codec, torch.randn(4, R, 4), batch)
    p = physics(frames, sample)
    for k in ("bond_length_error", "chirality_violation_rate",
              "clashes_per_1000_atoms"):
        assert k in p and np.isfinite(p[k])


def test_the_template_scores_well_on_its_own_physics():
    """Control: the template's true coordinates must not look broken, or the
    physics scoring is measuring the metric rather than the sample."""
    sample, _, _, _ = _setup(n_res=6)
    ref = np.asarray(sample["coords"], dtype=np.float64)
    p = physics([ref, ref], sample)
    assert p["bond_length_error"] < 1e-6
    assert p["chirality_violation_rate"] < 1e-9


# --- end to end ------------------------------------------------------------

def test_sample_and_score_runs_and_reports_every_axis():
    sample, batch, codec, diff = _setup(n_res=5)
    g = torch.Generator().manual_seed(0)
    frames, m = sample_and_score(diff, codec, sample, batch, n_frames=5, steps=3,
                                 generator=g, device=torch.device("cpu"))
    assert len(frames) == 5
    for k in ("bond_length_error", "mean_step", "jump_ratio", "spread",
              "spread_over_step", "n_frames", "n_atoms"):
        assert k in m


def test_sampling_is_reproducible_for_a_seed():
    sample, batch, codec, diff = _setup()
    a, _ = sample_and_score(diff, codec, sample, batch, n_frames=4, steps=3,
                            generator=torch.Generator().manual_seed(5),
                            device=torch.device("cpu"))
    b, _ = sample_and_score(diff, codec, sample, batch, n_frames=4, steps=3,
                            generator=torch.Generator().manual_seed(5),
                            device=torch.device("cpu"))
    assert all(np.allclose(x, y) for x, y in zip(a, b))
