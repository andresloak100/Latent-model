"""Quality-tiered data mixing with a curriculum schedule.

The NLP staging: pretrain on abundant lower-quality data, then raise the
proportion of expensive high-quality data as training proceeds ("mid-training"),
then RL. Applied here, the abundant tier is predicted structures (RCSB hosts
~1M computed structure models) and the expensive tier is the ~230k experimental
entries -- of which our codec has used 2,272.

Where the analogy holds and where it does not
---------------------------------------------
It holds on scale: high-quality data alone runs out long before the model
stops improving, which the data ladder measured directly (val still descending
at n=2272, not plateaued).

It does NOT hold on kind. In NLP both tiers are text and differ in quality.
Predicted structures differ from experimental ones SYSTEMATICALLY, not just in
accuracy: they are built from near-ideal internal coordinates, so bond lengths
are near-perfect, clashes are rare, and there is no crystallographic strain,
no alternate conformation, no ligand. That is closer to domain shift than to a
quality gradient, and our headline metrics are exactly the quantities that
differ (bond-length error, clash rate, chirality). ``tier_geometry_stats``
below exists to measure that gap before committing to a mixture, rather than
discovering it in a training curve.

Consequently the one invariant enforced here: VALIDATION IS NEVER MIXED. A
held-out set containing predicted structures would report how well the model
learned the prediction distribution, which is not the question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TierPlan:
    """Weight of one data tier as a function of training progress."""
    name: str
    start: float          # weight at progress 0.0
    end: float            # weight at progress 1.0
    # Progress at which the ramp begins/ends. Defaults span the whole run;
    # set these for a discrete phase change (90% pretrain, then mid-train)
    # rather than a continuous ramp.
    ramp_start: float = 0.0
    ramp_end: float = 1.0

    def weight_at(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        if p <= self.ramp_start:
            return self.start
        if p >= self.ramp_end:
            return self.end
        span = max(self.ramp_end - self.ramp_start, 1e-9)
        frac = (p - self.ramp_start) / span
        return self.start + (self.end - self.start) * frac


@dataclass
class Curriculum:
    """A set of tier plans, normalised to a sampling distribution."""
    plans: list = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec):
        """Build a two-tier schedule from four sweepable scalars.

        The schedule is a hyper-parameter, so it needs to be a handful of
        numbers in a config rather than a list of plan objects. A grid over
        ``hq_start`` x ``ramp_start`` is then a normal config sweep.

            hq_start   high-quality fraction at the beginning   (e.g. 0.1)
            hq_end     high-quality fraction at the end         (e.g. 1.0)
            ramp_start progress at which the shift begins       (e.g. 0.7)
            ramp_end   progress at which it completes           (e.g. 1.0)

        ramp_start == ramp_end gives a discrete phase change; a wide gap gives
        a gradual ramp. Both are worth sweeping: NLP mid-training is usually
        described as a phase, but nothing says a phase beats a ramp here.
        """
        hq0 = float(spec.get("hq_start", 0.1))
        hq1 = float(spec.get("hq_end", 1.0))
        rs = float(spec.get("ramp_start", 0.0))
        re = float(spec.get("ramp_end", 1.0))
        lo = spec.get("low_tier", "predicted")
        hi = spec.get("high_tier", "experimental")
        return cls([
            TierPlan(lo, 1.0 - hq0, 1.0 - hq1, ramp_start=rs, ramp_end=re),
            TierPlan(hi, hq0, hq1, ramp_start=rs, ramp_end=re),
        ])

    def weights_at(self, progress: float) -> dict:
        raw = {p.name: max(p.weight_at(progress), 0.0) for p in self.plans}
        total = sum(raw.values())
        if total <= 0:
            # Degenerate schedule: fall back to uniform rather than dividing
            # by zero and silently training on nothing.
            n = max(len(raw), 1)
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}

    def sample_counts(self, progress: float, n_samples: int, sizes: dict) -> dict:
        """How many samples to draw from each tier this epoch.

        Tiers with no members get zero regardless of weight, and their share is
        redistributed -- otherwise a missing tier silently shrinks the epoch.
        """
        w = self.weights_at(progress)
        live = {k: v for k, v in w.items() if sizes.get(k, 0) > 0}
        total = sum(live.values())
        if total <= 0:
            return {k: 0 for k in w}
        counts = {k: int(round(n_samples * v / total)) for k, v in live.items()}
        # Fix rounding drift so the epoch size is exact.
        drift = n_samples - sum(counts.values())
        if counts and drift:
            k = max(counts, key=lambda k: live[k])
            counts[k] = max(counts[k] + drift, 0)
        return {k: counts.get(k, 0) for k in w}


def mixture_indices(curriculum, tiers, progress, n_samples, rng):
    """Indices for one epoch, drawn per tier at the scheduled proportions.

    ``tiers`` maps tier name -> array of dataset indices. Sampling is WITH
    replacement within a tier: the whole point is to oversample the small
    high-quality tier late in training, which is impossible without it.
    """
    sizes = {k: len(v) for k, v in tiers.items()}
    counts = curriculum.sample_counts(progress, n_samples, sizes)
    out = []
    for name, n in counts.items():
        pool = np.asarray(tiers.get(name, []))
        if n <= 0 or pool.size == 0:
            continue
        out.append(rng.choice(pool, size=n, replace=True))
    if not out:
        return np.zeros(0, dtype=np.int64)
    idx = np.concatenate(out)
    rng.shuffle(idx)
    return idx.astype(np.int64)


def assert_clean_validation(val_keys, tier_of):
    """Validation must contain only the highest-quality tier.

    A val set with predicted structures in it measures how well the model
    learned the prediction distribution. That is not the question, and the
    error is invisible in the loss curve -- it looks like success.
    """
    bad = sorted({tier_of[k] for k in val_keys if tier_of.get(k, "experimental")
                  != "experimental"})
    if bad:
        raise ValueError(
            f"validation set contains non-experimental tiers {bad}; "
            "held-out evaluation must be experimental only")


def tier_geometry_stats(samples):
    """Geometry statistics that distinguish predicted from experimental.

    Predicted structures are built from near-ideal internal coordinates, so
    these separate the tiers sharply. Run this on both BEFORE choosing a
    mixture: if the distributions are close the curriculum is low risk, and if
    they are far apart the high-quality tier needs more weight, earlier.
    """
    bond_sd, within_sd, clash, n_atoms, rg_ratio = [], [], [], [], []
    for d in samples:
        coords = np.asarray(d["coords"], dtype=np.float64)
        bonds = np.asarray(d["bonds"]).reshape(-1, 2)
        elems = [str(x).upper() for x in d.get("element_symbol", [])]
        n_atoms.append(len(coords))
        if len(bonds):
            lens = np.linalg.norm(coords[bonds[:, 0]] - coords[bonds[:, 1]], axis=1)
            bond_sd.append(float(lens.std()))
            if elems:
                by_type = {}
                for (i, j), L in zip(bonds, lens):
                    by_type.setdefault(tuple(sorted((elems[i], elems[j]))), []).append(L)
                num = sum(len(v) * float(np.std(v)) for v in by_type.values() if len(v) >= 3)
                den = sum(len(v) for v in by_type.values() if len(v) >= 3)
                if den:
                    within_sd.append(num / den)
        if len(coords) > 1:
            cen = coords - coords.mean(axis=0)
            rg = float(np.sqrt((cen ** 2).sum(axis=1).mean()))
            # Rg ~ 2.2 * N^0.38 is calibrated for N = RESIDUES, not atoms.
            # Passing atoms overestimates Rg by a flat 2.2x at ~8 heavy atoms
            # per residue, making every structure look far more compact than
            # it is and putting the "well above 1 = extended" reading out of
            # reach entirely.
            n_res = (len(set(np.asarray(d["res_pos"]).tolist()))
                     if "res_pos" in d else max(len(coords) // 8, 1))
            rg_ratio.append(rg / (2.2 * max(n_res, 1) ** 0.38))
            dm = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            iu = np.triu_indices(len(coords), k=1)
            bonded = set(map(tuple, bonds.tolist()))
            close = sum(1 for a, b in zip(*iu)
                        if dm[a, b] < 2.0 and (a, b) not in bonded)
            clash.append(1000.0 * close / len(coords))

    def m(v):
        return float(np.mean(v)) if len(v) else float("nan")

    return {
        "n_structures": len(samples),
        "mean_atoms": m(n_atoms),
        # NUISANCE-DOMINATED, kept only for continuity. A per-structure spread
        # over all bond types is dominated by the difference BETWEEN ideal
        # types (C-C 1.52, C-N 1.33, C-O 1.23), ~0.11 A by construction. Both
        # tiers measuring 0.115-0.118 mostly reflects bond-type composition,
        # not refinement noise -- so it cannot answer the question it was
        # written for.
        "bond_length_sd": m(bond_sd),
        # THE real measurement: spread WITHIN each element pair, count-weighted.
        # An ideal-geometry builder emits a near-constant length per type; a
        # refined structure carries real strain. Nuisance term removed, so a
        # difference here is signal.
        "bond_length_sd_within_type": m(within_sd),
        "clashes_per_1000_atoms": m(clash),
        # Catches what bonds cannot see at all: low-confidence regions of
        # predicted structures are extended ribbons whose bond lengths are
        # perfect and whose global shape is unphysical. Against the empirical
        # folded-globular scaling Rg ~ 2.2 * N^0.38; well above 1 = not compact.
        "rg_ratio": m(rg_ratio),
    }
