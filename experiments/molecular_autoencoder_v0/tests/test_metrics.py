"""Metric-correctness regression tests (clash normalization, contacts)."""

import numpy as np

from molae.metrics import clash_metrics, contact_map_recovery, build_topology_info
from molae.synthetic import make_synthetic_ala


def test_clash_rate_size_invariant_under_subsampling():
    """Regression: clashes_per_1000_atoms must not depend on the subsample cap.

    A uniform random cloud with a fixed clash density should report ~the same
    per-1000 rate whether or not subsampling fires.
    """
    rng = np.random.default_rng(0)
    coords = rng.normal(scale=5.0, size=(600, 3))
    no_bonds = np.zeros((0, 2), dtype=np.int64)
    radii = np.full(600, 0.77)
    full = clash_metrics(coords, no_bonds, radii, clash_dist=1.0, max_atoms=10000)
    sub = clash_metrics(coords, no_bonds, radii, clash_dist=1.0, max_atoms=200)
    assert sub["clash_subsampled"] is True
    assert full["clash_subsampled"] is False
    # The corrected estimate is within ~2x of the exact rate (sampling
    # variance). The OLD full-n denominator bug produced ~9x undercount
    # (ratio ~0.11), which this window rejects.
    a, b = full["clashes_per_1000_atoms"], sub["clashes_per_1000_atoms"]
    assert 0.5 < b / max(a, 1e-9) < 2.0


def test_contact_map_recovery_perfect_on_identity():
    # Compact spacing (2.5 A) so |i-j|>=3 CA pairs fall within the 8 A cutoff
    # and real contacts exist; identical structures must recover them perfectly.
    m = make_synthetic_ala(8, spacing=2.5)
    topo = build_topology_info(m["atom_name_idx"], m["res_pos"], m["bonds"], m["element_symbol"])
    c = m["coords"].astype(np.float64)
    r = contact_map_recovery(c, c, topo)
    assert r["contact_f1"] == 1.0


def test_contact_map_defined_at_seq_sep_plus_one():
    """Regression: a structure with exactly seq_sep+1 CA residues is valid."""
    m = make_synthetic_ala(4)  # 4 CA residues, seq_sep=3 -> pair (0,3) exists
    topo = build_topology_info(m["atom_name_idx"], m["res_pos"], m["bonds"], m["element_symbol"])
    r = contact_map_recovery(m["coords"].astype(np.float64), m["coords"].astype(np.float64),
                             topo, seq_sep=3)
    assert not np.isnan(r["contact_f1"])
