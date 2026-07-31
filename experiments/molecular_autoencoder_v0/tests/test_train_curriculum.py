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


# --- deterministic train eval ---------------------------------------------

def test_heldout_rmsd_is_deterministic_unlike_quick_rmsd():
    """The reason a fixed train slice was added.

    quick_rmsd samples 4 random batches, so repeated calls on an unchanged
    model disagree. heldout_rmsd walks a loader in order and does not. A
    train/val gap cannot be read off a metric whose noise exceeds the gap.
    """
    import torch
    from torch.utils.data import DataLoader
    from train import quick_rmsd, heldout_rmsd
    from molae.dataset import collate_fn, sample_from_arrays
    from molae.synthetic import make_synthetic_ala
    from molae.model import ModelConfig
    from molae.model_direct import DirectAutoencoder

    samples = [sample_from_arrays(make_synthetic_ala(n_res=4 + (i % 5)))
               for i in range(24)]

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i): return samples[i]

    cfg = ModelConfig(d_model=32, n_heads=2, latent_dim=4, enc_self_layers=1,
                      dec_self_layers=1, encoder_type="direct")
    model = DirectAutoencoder(cfg)
    model.eval()
    loader = DataLoader(DS(), batch_size=4, shuffle=True, collate_fn=collate_fn)
    fixed = DataLoader(DS(), batch_size=4, shuffle=False, collate_fn=collate_fn)

    a = heldout_rmsd(model, fixed, torch.device("cpu"))
    b = heldout_rmsd(model, fixed, torch.device("cpu"))
    assert abs(a - b) < 1e-9, "deterministic eval disagreed with itself"

    torch.manual_seed(0)
    q = [quick_rmsd(model, loader, torch.device("cpu"), max_batches=2)
         for _ in range(6)]
    assert max(q) - min(q) > 1e-6, "sampled eval was suspiciously stable"
