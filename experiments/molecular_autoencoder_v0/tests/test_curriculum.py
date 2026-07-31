"""Quality-tiered mixing with a curriculum schedule.

The load-bearing test is `test_validation_must_be_experimental_only`. A val set
containing predicted structures measures how well the model learned the
prediction distribution rather than whether it generalises -- and that error is
invisible in a loss curve, because it looks exactly like success.
"""

import numpy as np
import pytest

from molae.curriculum import (
    TierPlan, Curriculum, mixture_indices, assert_clean_validation,
    tier_geometry_stats,
)


def _ramp():
    """90/10 predicted:experimental at the start, inverted by the end."""
    return Curriculum([
        TierPlan("predicted", start=0.9, end=0.0),
        TierPlan("experimental", start=0.1, end=1.0),
    ])


def test_weights_are_normalised_at_every_point():
    c = _ramp()
    for p in np.linspace(0, 1, 11):
        assert abs(sum(c.weights_at(p).values()) - 1.0) < 1e-9


def test_high_quality_share_increases_monotonically():
    c = _ramp()
    shares = [c.weights_at(p)["experimental"] for p in np.linspace(0, 1, 11)]
    assert shares == sorted(shares)
    assert shares[0] < 0.2 and shares[-1] > 0.99


def test_discrete_phase_change_is_expressible():
    """Mid-training as a phase, not a ramp: 90% of the run pretrains, then the
    high-quality tier takes over."""
    c = Curriculum([
        TierPlan("predicted", 1.0, 0.0, ramp_start=0.9, ramp_end=0.9),
        TierPlan("experimental", 0.0, 1.0, ramp_start=0.9, ramp_end=0.9),
    ])
    assert c.weights_at(0.5)["predicted"] == 1.0
    assert c.weights_at(0.95)["experimental"] == 1.0


def test_sample_counts_sum_exactly_despite_rounding():
    c = _ramp()
    sizes = {"predicted": 1000, "experimental": 50}
    for p in np.linspace(0, 1, 7):
        counts = c.sample_counts(p, 997, sizes)
        assert sum(counts.values()) == 997


def test_missing_tier_does_not_shrink_the_epoch():
    """If the predicted tier is absent, its share must be redistributed, not
    silently dropped -- otherwise the epoch quietly gets smaller."""
    c = _ramp()
    counts = c.sample_counts(0.0, 500, {"predicted": 0, "experimental": 40})
    assert counts["predicted"] == 0
    assert counts["experimental"] == 500


def test_degenerate_schedule_does_not_divide_by_zero():
    c = Curriculum([TierPlan("a", 0.0, 0.0), TierPlan("b", 0.0, 0.0)])
    w = c.weights_at(0.5)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_sampling_proportions_track_the_schedule():
    c = _ramp()
    tiers = {"predicted": np.arange(0, 1000), "experimental": np.arange(1000, 1050)}
    rng = np.random.default_rng(0)
    early = mixture_indices(c, tiers, 0.0, 4000, rng)
    late = mixture_indices(c, tiers, 1.0, 4000, rng)
    frac_exp_early = (early >= 1000).mean()
    frac_exp_late = (late >= 1000).mean()
    assert 0.05 < frac_exp_early < 0.15
    assert frac_exp_late > 0.99


def test_small_tier_is_oversampled_with_replacement():
    """The whole point of the late phase: 50 high-quality structures must be
    able to fill a 4000-sample epoch."""
    c = _ramp()
    tiers = {"predicted": np.arange(0, 1000), "experimental": np.arange(1000, 1050)}
    idx = mixture_indices(c, tiers, 1.0, 4000, np.random.default_rng(0))
    assert len(idx) == 4000
    assert len(set(idx.tolist())) <= 50


def test_sampling_is_deterministic_for_a_seed():
    c = _ramp()
    tiers = {"predicted": np.arange(100), "experimental": np.arange(100, 120)}
    a = mixture_indices(c, tiers, 0.5, 200, np.random.default_rng(7))
    b = mixture_indices(c, tiers, 0.5, 200, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_validation_must_be_experimental_only():
    tier_of = {"a": "experimental", "b": "predicted", "c": "experimental"}
    assert_clean_validation(["a", "c"], tier_of)
    with pytest.raises(ValueError, match="experimental only"):
        assert_clean_validation(["a", "b"], tier_of)


def test_geometry_stats_separate_ideal_from_strained():
    """Ideal-geometry builders give a very tight bond-length spread;
    experimental structures carry real strain. This is the measurement to run
    on both tiers BEFORE choosing a mixture."""
    rng = np.random.default_rng(0)
    ideal, strained = [], []
    for _ in range(4):
        n = 30
        base = np.cumsum(rng.normal(0, 1, (n, 3)), axis=0)
        bonds = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
        # ideal: every bond exactly 1.52 A
        c = [base[0]]
        for k in range(1, n):
            step = rng.normal(0, 1, 3)
            c.append(c[-1] + step / np.linalg.norm(step) * 1.52)
        ideal.append({"coords": np.array(c), "bonds": bonds})
        # strained: same but with real spread
        c = [base[0]]
        for k in range(1, n):
            step = rng.normal(0, 1, 3)
            c.append(c[-1] + step / np.linalg.norm(step) * rng.normal(1.52, 0.05))
        strained.append({"coords": np.array(c), "bonds": bonds})
    si = tier_geometry_stats(ideal)
    ss = tier_geometry_stats(strained)
    assert si["bond_length_sd"] < 1e-6
    assert ss["bond_length_sd"] > si["bond_length_sd"] * 100


IDEAL = {("C", "C"): 1.52, ("C", "N"): 1.33, ("N", "N"): 1.40}


def _mixed_chain(n, rng, noise=0.0):
    """Chain with genuinely MIXED bond types, each at its own ideal length.

    Elements come in pairs (C C N N C C N N ...) so C-C, C-N and N-N all
    occur. An earlier fixture alternated C,N which makes every bond C-N --
    a single type, where within-type sd equals the aggregate and the test
    proves nothing.
    """
    elems = [("C" if (k // 2) % 2 == 0 else "N") for k in range(n)]
    c = [np.zeros(3)]
    for k in range(1, n):
        key = tuple(sorted((elems[k - 1], elems[k])))
        L = IDEAL[key] + (rng.normal(0, noise) if noise else 0.0)
        step = rng.normal(0, 1, 3)
        c.append(c[-1] + step / np.linalg.norm(step) * L)
    bonds = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return {"coords": np.array(c), "bonds": bonds, "element_symbol": elems}


def test_aggregate_sd_is_blind_to_refinement_noise():
    """The flaw the cluster found: a mixed-bond-type structure with ZERO
    within-type noise still reports sd ~0.1, because the spread BETWEEN ideal
    types dominates. Both tiers measured 0.115-0.118 for this reason."""
    rng = np.random.default_rng(0)
    s = tier_geometry_stats([_mixed_chain(120, rng, noise=0.0)])
    assert s["bond_length_sd"] > 0.05               # nuisance term, not noise
    assert s["bond_length_sd_within_type"] < 1e-9   # the truth: no noise at all


def test_within_type_sd_separates_ideal_from_refined():
    rng = np.random.default_rng(1)
    a = tier_geometry_stats([_mixed_chain(200, rng, noise=0.0)])
    b = tier_geometry_stats([_mixed_chain(200, rng, noise=0.03)])
    # The aggregate barely moves between an ideal builder and a refined
    # structure; the within-type measure separates them cleanly.
    assert abs(a["bond_length_sd"] - b["bond_length_sd"]) < 0.02
    assert a["bond_length_sd_within_type"] < 1e-9
    assert b["bond_length_sd_within_type"] > 0.015


def test_rg_ratio_flags_an_extended_chain():
    """What bonds cannot see: a low-confidence predicted region is an extended
    ribbon with PERFECT bond lengths and unphysical global shape."""
    rng = np.random.default_rng(2)
    n = 120
    extended = np.stack([np.arange(n) * 3.8, np.zeros(n), np.zeros(n)], axis=1)
    bonds = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    ext = tier_geometry_stats([{"coords": extended, "bonds": bonds,
                                "element_symbol": ["C"] * n}])
    folded = rng.normal(0, 2.2 * n ** 0.38 / np.sqrt(3), (n, 3))
    fold = tier_geometry_stats([{"coords": folded, "bonds": bonds,
                                 "element_symbol": ["C"] * n}])
    assert ext["rg_ratio"] > 5 * fold["rg_ratio"]


# --- the schedule as a sweepable hyper-parameter ---------------------------

def test_from_spec_matches_a_hand_built_ramp():
    a = Curriculum.from_spec({"hq_start": 0.1, "hq_end": 1.0})
    b = _ramp()
    for p in np.linspace(0, 1, 6):
        assert abs(a.weights_at(p)["experimental"]
                   - b.weights_at(p)["experimental"]) < 1e-9


def test_from_spec_defaults_are_sane():
    c = Curriculum.from_spec({})
    assert c.weights_at(0.0)["experimental"] < 0.2
    assert c.weights_at(1.0)["experimental"] > 0.99


def test_ramp_start_delays_the_shift():
    """Mid-training late in the run: high-quality share stays flat until the
    ramp opens."""
    c = Curriculum.from_spec({"hq_start": 0.1, "hq_end": 1.0,
                              "ramp_start": 0.9, "ramp_end": 1.0})
    assert abs(c.weights_at(0.5)["experimental"] - 0.1) < 1e-9
    assert abs(c.weights_at(0.8)["experimental"] - 0.1) < 1e-9
    assert c.weights_at(0.95)["experimental"] > 0.5


def test_a_sweep_over_the_schedule_is_a_normal_config_grid():
    """The point of from_spec: the curriculum is four scalars, so sweeping it
    is a config grid rather than a code change."""
    seen = set()
    for hq_start in (0.0, 0.1, 0.3):
        for ramp_start in (0.0, 0.5, 0.9):
            c = Curriculum.from_spec({"hq_start": hq_start, "hq_end": 1.0,
                                      "ramp_start": ramp_start, "ramp_end": 1.0})
            mid = round(c.weights_at(0.6)["experimental"], 4)
            seen.add((hq_start, ramp_start, mid))
            assert 0.0 <= mid <= 1.0
    assert len(seen) == 9        # every cell is a distinct schedule
