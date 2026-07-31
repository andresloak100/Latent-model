"""Curriculum wiring in the training loop.

The machinery in molae/curriculum.py is only useful if train.py actually
draws from it. These test the wiring: that a tiered config produces a mixture
that tracks the schedule, and that validation stays experimental-only.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from molae.curriculum import Curriculum, mixture_indices  # noqa: E402


def test_epoch_mixture_tracks_progress_through_the_run():
    """What the loop does each epoch: progress -> weights -> indices."""
    cur = Curriculum.from_spec({"hq_start": 0.1, "hq_end": 1.0,
                                "ramp_start": 0.5, "ramp_end": 1.0})
    tiers = {"predicted": np.arange(0, 5000),
             "experimental": np.arange(5000, 5200)}
    rng = np.random.default_rng(0)
    epochs = 100
    fracs = []
    for epoch in (0, 25, 50, 75, 99):
        progress = epoch / (epochs - 1)
        idx = mixture_indices(cur, tiers, progress, 2000, rng)
        fracs.append((idx >= 5000).mean())
    # flat through the pretraining phase, then rises to all-experimental
    assert fracs[0] < 0.15 and fracs[1] < 0.15 and fracs[2] < 0.15
    assert fracs[3] > fracs[2]
    assert fracs[4] > 0.95


def test_epoch_size_is_stable_across_the_schedule():
    """The epoch must not shrink as the high tier takes over -- otherwise
    steps-per-epoch drifts and the step budget means different things at
    different points in the run."""
    cur = Curriculum.from_spec({"hq_start": 0.1, "hq_end": 1.0})
    tiers = {"predicted": np.arange(0, 5000),
             "experimental": np.arange(5000, 5050)}
    rng = np.random.default_rng(0)
    sizes = [len(mixture_indices(cur, tiers, p, 3000, rng))
             for p in (0.0, 0.5, 1.0)]
    assert sizes == [3000, 3000, 3000]


def test_single_source_config_is_unaffected():
    """An empty tiers dict must leave training exactly as it was."""
    from molae.config import ExperimentConfig
    cfg = ExperimentConfig()
    assert cfg.data.tiers == {}
    assert cfg.data.curriculum == {}


def test_tiered_config_round_trips_through_yaml(tmp_path):
    from molae.config import ExperimentConfig
    cfg = ExperimentConfig()
    cfg.data.tiers = {"predicted": "data/processed_csm",
                      "experimental": "data/processed_big"}
    cfg.data.curriculum = {"hq_start": 0.1, "ramp_start": 0.9}
    p = tmp_path / "c.yaml"
    cfg.save(p)
    back = ExperimentConfig.from_yaml(str(p))
    assert back.data.tiers == cfg.data.tiers
    assert back.data.curriculum == cfg.data.curriculum
    cur = Curriculum.from_spec(back.data.curriculum)
    assert abs(cur.weights_at(0.5)["experimental"] - 0.1) < 1e-9
