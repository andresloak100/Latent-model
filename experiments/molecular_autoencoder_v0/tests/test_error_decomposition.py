"""Error decomposition: internal geometry vs assembly placement."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from error_decomposition import parts_of  # noqa: E402
from molae.alignment import kabsch_rmsd_numpy  # noqa: E402


def test_multi_chain_parts_are_the_chains():
    chain = np.array([0] * 50 + [1] * 60)
    parts = parts_of(chain)
    assert len(parts) == 2 and len(parts[0]) == 50 and len(parts[1]) == 60


def test_single_chain_is_split_into_halves():
    """A two-domain protein has the placement problem with ONE chain, which
    is exactly what a chain-count test cannot see."""
    parts = parts_of(np.zeros(100, dtype=int))
    assert len(parts) == 2 and len(parts[0]) + len(parts[1]) == 100


def test_tiny_parts_are_dropped():
    chain = np.array([0] * 50 + [1] * 5)
    assert len(parts_of(chain, n_min=30)) == 1


def test_placement_error_is_detected_when_parts_are_displaced():
    """Two rigid bodies reconstructed PERFECTLY but placed wrongly: per-part
    RMSD is ~0 while global RMSD is large. This is the signature the
    decomposition exists to find."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 5, (60, 3))
    b = rng.normal(0, 5, (60, 3)) + np.array([40.0, 0, 0])
    true = np.vstack([a, b])
    pred = np.vstack([a, b + np.array([8.0, 0, 0])])     # part B shifted
    idx = [np.arange(60), np.arange(60, 120)]
    glob = kabsch_rmsd_numpy(pred, true)
    per = float(np.mean([kabsch_rmsd_numpy(pred[i], true[i]) for i in idx]))
    assert per < 1e-6, "parts should be internally perfect"
    assert glob > 2.0, "global should be dominated by the placement error"


def test_local_error_shows_up_in_both():
    """The other branch: genuinely wrong local geometry raises per-part RMSD
    too, so the decomposition does not mistake it for a placement failure."""
    rng = np.random.default_rng(1)
    true = np.vstack([rng.normal(0, 5, (60, 3)),
                      rng.normal(0, 5, (60, 3)) + np.array([40.0, 0, 0])])
    pred = true + rng.normal(0, 1.5, true.shape)
    idx = [np.arange(60), np.arange(60, 120)]
    glob = kabsch_rmsd_numpy(pred, true)
    per = float(np.mean([kabsch_rmsd_numpy(pred[i], true[i]) for i in idx]))
    assert per > 0.5 * glob, "local noise must not look like a placement error"
