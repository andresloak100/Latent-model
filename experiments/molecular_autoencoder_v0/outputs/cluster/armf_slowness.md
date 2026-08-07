# Slowness-sensitive dimensionality: the ms concern is REAL but it lives in STATE COUNT, not in

> **LOWER BOUND (INBOX 002/011).** Every rank90 on this page is measured IN-SAMPLE, where PCA
> explains variance optimally by construction. Measured out of sample on ATLAS (fit replicas 0+1,
> evaluate replica 2, n=123): the in-sample rank90 modes cover a median **77.1%** of held-out
> variance, not 90% — below 90% in **123/123** systems — and `rank90_out > rank90_in` in **118/118**
> that reach 90% at all, ratio median **3.49×**. Five never reach 90% at any k and they are the
> **largest** in the corpus, so there rank90 is *undefined* out of sample. The N-exponent roughly
> **doubles** out of sample (`+0.4657 ± 0.2158` → `+0.9285 ± 0.2220`, itself biased low by that
> exclusion, so **b ≥ 0.93**). Read every dimensionality figure below as a **floor**, and every
> "does not grow with atom count" as **not shown to grow, on an in-sample measure that would hide it**.
> `n_eff` is 1–7% of frames corpus-wide, so this is not fixable with more data from any available
> corpus — latent width is now sized from the codec's own held-out saturation curve
> (`scripts/armf_atlas_dm.py`), not from rank90. See ROADMAP "WIDTH CHAIN IS A FLOOR".

# latent WIDTH (objective 3, closed)

rank90 counts directions carrying >=10% of VARIANCE, so it is blind to rare brief excursions -- the
exact ms-scale events of concern. Two slowness-sensitive metrics on the same window and temperature
sweeps: TICA dimensionality (ICs for 90% of KINETIC variance -- ranks directions by SLOWNESS, so a
slow rare transition is a leading component even when it is a negligible PCA component) and STATE
COUNT (leader clustering at an RMSD cutoff -- "new states" directly, not a variance proxy).
mdCATH, 28 domains, 5 temps x 5 replicas, ALL-ATOM. Cutoffs set from each domain's OWN 320K
distribution and then held FIXED across temperatures so the temperature comparison stays valid.

## 1. Window sweep at 320K -- dimensionality saturates, state count does NOT
| window | TICA dim | (TICA basis) | rank90 | states@0.75R | states@0.5R |
|---|---|---|---|---|---|
| 200 | 10 | 20 | 68 | 42 | 196 |
| 400 | 18 | 40 | 90 | 83 | 388 |
| 800 | 35 | 80 | 83 | 179 | 768 |
| 1600 | 38 | 100 | 90 | 352 | 1528 |
| 2400 | 36 | 100 | 96 | 609 | 2246 |

TICA dimensionality **saturates at ~36-38** by window 1600 (and stays at 36-38% of its basis, so
not censored). State count grows **~linearly with window** (~0.25 x frames) and does not saturate.

## 2. Temperature sweep, PAIRED within-domain, FOLDED only
| T | n folded | TICA ratio | rank90 ratio | states ratio |
|---|---|---|---|---|
| 348K | 26 | 1.000 | 0.985 | 1.700 |
| 379K | 21 | 0.976 | 0.925 | 2.913 |
| 413K | 8 | 1.014 | 1.082 | 4.147 |

Paired slope vs log10(effective time), Ea=10 kcal/mol:
- **TICA dim: +0.001 +/- 0.024 -> FLAT**, and far TIGHTER than rank90: at 1 ms effective time the
  CI bounds the change to **0.87x - 1.16x**.
- rank90: -0.010 +/- 0.128 -> FLAT [0.44x - 2.02x]
- **state count: +0.303 +/- 0.170 -> RISES**, at 1 ms effective time **2.22x - 16.95x**.

## VERDICT -- the rare-state effect is REAL, and it is NOT a dimensionality effect
The pre-registered fork was "both flat -> ms concern reduced" vs "both rise -> rare-state effect".
The measurement splits them, which is more informative than either:

**The number of SLOW COLLECTIVE DIRECTIONS does not grow with effective time (TICA flat, tightly).**
That is the result that stands.

**CORRECTION (censoring check, run after the first draft of this file): the state-count rise is NOT
usable.** State count as a FRACTION of frames rises 0.254 (320K) -> 0.482 (348K) -> 0.678 (379K) ->
**0.946 (413K)**: the metric becomes progressively censored by the frame ceiling exactly where the
largest effective times are, so the "2-17x more states" slope is measuring the ceiling, not
exploration. At 320K the fraction is a flat ~0.21-0.25 across windows 200-2400, i.e. the count is
essentially 0.22 x frames -- the trajectory never begins revisiting at this cutoff, so the metric
cannot distinguish unbounded exploration from a cutoff too tight to detect recurrence. The tighter
0.5R cutoff is censored outright (fraction 0.94-0.98). **The coverage question is therefore
UNRESOLVED, not answered.** Any claim that ms coverage grows 2-17x should not be quoted.

Consequence, and it separates two components that were being conflated:
- ~~**CODEC capacity (latent width DM) is sized by DIMENSIONALITY**~~ **-- THE SIZING HALF IS
  RETIRED (INBOX 002).** DM is no longer sized from rank90 or TICA-dim at all: rank90 is measured
  in-sample and is a FLOOR whose N-exponent roughly doubles out of sample (`b >= 0.93`), and no
  corpus supplies the `n_eff` to fix it. Latent width now comes from the codec's own held-out
  saturation curve (`scripts/armf_atlas_dm.py`). **The TIME half still stands** -- the ms regime does
  not demand a wider latent, on the TICA-vs-effective-time measurement below. But *"flat in time"*
  was never *"flat in N"*: TICA-dim vs ATOM COUNT was not measured until job 10306540 (INBOX 007).
  Original text: this is what licenses sizing DM from rank90/TICA, and it survives the
  slowness test that rank90 alone could not pass.
- **PROPAGATOR coverage is sized by STATE COUNT -> UNRESOLVED (censored metric).** The concern is
  neither confirmed nor dismissed. Settling it needs a recurrence-sensitive statistic that does not
  saturate at the frame count (e.g. a cutoff rescaled per temperature, which then breaks
  cross-temperature comparability, or an MSM/committor-based state definition).

Objective 3 goes back on the shelf with ONE half closed: latent WIDTH does not need to grow with
simulated time (TICA flat, uncensored, tight CI). Coverage remains open and is the honest residual.

## Limits
- **State count is censored** -- see the CORRECTION above. Fraction-of-frames is the diagnostic that
  exposes it, and it should be reported alongside any clustering count in future.
- Two metric-design errors were caught and fixed before these numbers: an absolute 2A cutoff
  censored the state count at 2399/2400 (every frame its own state), and a temperature-varying TICA
  basis made the TICA dimension track the basis (near-constant 0.42-0.45 ratio) rather than
  slowness. Both are the same failure as rank90's frame cap; the constant basis and the reported
  basis size keep it visible.
- Effective time uses Arrhenius with an ASSUMED Ea=10 kcal/mol; max usable folded effective time is
  ~86 us, so 1 ms is a ~12x extrapolation.
