# Ensemble validation of the rollout: did the propagator learn dynamics?

"0% out of range" is necessary but not sufficient -- a model emitting the training
mean every step also scores 0%, so does one with far too little variance. Test the
sufficient conditions: generated rollout vs REFERENCE MD in ANM-mode coefficient
space, sqrt(kT/lambda) whitening, held-out (frozen-7) systems, 2000-step rollout.

## Result

| system | nCA | std ratio (gen/ref) | IAT gen | IAT ref | IAT ratio | JS | basin cover | spurious |
|---|---|---|---|---|---|---|---|---|
| 3a5zD02 | 64  | 0.48 | 1.6 | 8.3  | 0.19 | 0.29 | 47% | 28% |
| 3jvvA01 | 100 | 0.31 | 1.6 | 4.5  | 0.35 | 0.34 | 35% | 19% |
| 3a9lA00 | 207 | 0.24 | 3.0 | 25.8 | 0.11 | 0.50 | 18% | 25% |
| MEAN    |     | **0.34** | 2.0 | 12.9 | **0.16** | | | |

## Verdict: VARIANCE COLLAPSE -- the in-range result was vacuous

Pre-registered first failure branch (std ratio well under 1 -> narrow blob, in-range
vacuous) is what happened.

- **Marginals collapsed:** generated per-mode std ~1/3 of reference (0.34). The
  rollout samples a tight blob, not the conformational distribution.
- **Kinetics wrong:** generated integrated autocorrelation time ~6x too short (ratio
  0.16). Real MD has slow collective modes (ref IAT up to 25.8 frames); generated
  decorrelates in ~2-3 frames -- fast, structureless noise.
- **Free-energy landscape wrong:** rollout covers only 18-47% of reference basins
  (misses the majority), 19-28% spurious density (filled barriers).

**This retroactively guts the earlier "strongest positive."** The 1000-step "0% out
of range, ensemble sampling" result was VACUOUS: a collapsed narrow blob trivially
stays inside the training range. The propagator did NOT learn dynamics; it learned
bounded noise around a too-narrow region.

## Consequence

Physics solved the codec; learning belongs in the propagator, and the current
skeleton propagator does not do the job. A small conditional MLP DDPM p(z_{t+1}|z_t)
trained on one trajectory is mean-reverting, low-variance, fast-decorrelating. The
obj-3 problem is now sharp: build a propagator whose generated ensemble matches the
reference in (1) per-mode variance, (2) autocorrelation time, (3) free-energy basins.
Ensemble validation (this script) is the acceptance test for any future propagator.

## Step 1 diagnostic: is the conditioning path dead? -- NO (live)

| system | a_ddpm | a_ref | ratio | shift/spread(gen) | ideal | verdict |
|---|---|---|---|---|---|---|
| 3a5zD02 | 0.240 | 0.311 | 0.77 | 2.10 | 2.52 | conditioning live |
| 3jvvA01 | 0.200 | 0.248 | 0.81 | 3.41 | 2.03 | conditioning live |

The DDPM tracks its condition at ~0.8x the reference AR slope; the two-condition shift
is comparable to the ideal AR shift. Conditioning is NOT dead -- the single-input-concat
path works well enough. So the FiLM/cross-attention redesign is NOT warranted (that
branch does not fire).

**Refined mechanism:** the model under-estimates AR persistence by ~20%, and on slow
modes (a_ref -> 1) that is catastrophic. For AR(1), IAT=(1+a)/(1-a), var proportional
to 1/(1-a^2): a_ref 0.96 -> a_ddpm 0.77 gives IAT 49->7.7 (0.16x, matches observed)
and variance 0.19x (matches observed 0.34x). A modest per-step persistence shortfall,
amplified by IAT hypersensitivity near a=1, produces BOTH the variance collapse and
the short autocorrelation -- one cause, both observations, but NOT dead conditioning.
This is exactly what an OU baseline with physics-exact per-mode a_i = exp(-lambda_i
dt/gamma) fixes by construction -> step 2.

## Step 2: OU/Langevin baseline -- the propagator's ANM (physics beats the learned model)

Overdamped Ornstein-Uhlenbeck in the ANM basis: v_i = kT/lambda_i (one global scale),
a_i = exp(-lambda_i/gamma) (one friction gamma fit to reference lag-1 AR). Same
acceptance test as the DDPM.

| system | gamma | g-spread | std ratio | IAT ratio | basin cover |
|---|---|---|---|---|---|
| 3a5zD02 | 2.4 | 8.9 | 1.17 | 0.34 | 82% |
| 3jvvA01 | 1.9 | 5.1 | 1.40 | 0.53 | 71% |
| 3a9lA00 | 2.4 | 5.1 | 1.25 | 0.18 | 72% |
| MEAN | | | 1.27 | 0.35 | ~75% |
| (DDPM) | | | 0.34 | 0.16 | ~33% |

**The one-parameter OU baseline beats the learned DDPM on ALL THREE metrics**
(marginals 1.27 vs 0.34, IAT 0.35 vs 0.16, basins ~75% vs ~33%). The learned
propagator does not earn its inference cost -- the codec lesson repeats: a zero/one-
parameter physics baseline is not beaten. Any learned propagator must beat OU.

Two caveats from the pre-registered branches:
- **Non-Markovian:** per-mode gamma_i spreads 5-9x, so one global gamma cannot match
  all relaxation times -- that is why OU's IAT ratio is 0.35, not ~1. Per-mode gamma
  (= reference AR) would match IAT by construction; the single-gamma shortfall
  measures the non-Markovianity of the real dynamics.
- **Benchmark too easy:** OU covers 72-82% of top-2-mode basins -> that landscape is
  largely single-basin/Gaussian at this lag. The basin test must be STRENGTHENED
  (more modes, a non-Gaussianity/higher-moment metric) before it can discriminate a
  good learned propagator from OU. This is required before step 3 can be judged.

**Well-posed target for any learned propagator (step 3):** capture the non-Gaussian,
multi-basin, non-Markovian structure OU misses -- with a strengthened acceptance test
-- and beat OU on it. That is the floor, exactly as ANM was for the codec.

## Step 3 acceptance test -- STRENGTHENED (pre-registered before the sweep)

Step 2 showed the basin-coverage test is too easy (OU covers 72-82%). Three
discriminators added, in order of value:
1. **Cross-mode correlation.** OU in the ANM basis is independent per mode BY
   CONSTRUCTION -> it cannot produce cross-mode coupling. Measure mean |off-diagonal|
   of the generated coefficient correlation (linear) AND corr(|c_i|,|c_j|) (amplitude
   coupling), gen vs ref. Any nonzero reference coupling is unreachable by OU and
   reachable by a learned propagator -- the cleanest discriminator, costs one covariance.
2. **Basin transition RATE** (not coverage): transitions per 1000 steps between top-2-
   mode basins (median-split quadrants), gen vs ref. Coverage 72-82% may only mean the
   Gaussian is wide enough to touch the region; the RATE of basin-crossing separates a
   dynamics model from a wide blob.
3. **Non-Gaussianity per mode:** excess kurtosis (Gaussian=0) and per-mode marginal
   JS, not just std ratio (std 1.0 is compatible with a wrong shape).

**Pre-registered branch (now live):** if OU also passes all three -- near-zero
reference cross-mode coupling, matched basin-transition rate, near-Gaussian marginals
-- then the reference dynamics at this lag ARE effectively Gaussian and single-basin,
and the honest conclusion is **the task does not need a learned propagator at this
lag**. That is itself a major finding and argues for pushing to longer tau where
basin-hopping appears. Report it as such rather than forcing a learned win.

Then step 3: tau = 1/10/50/100 ns, per-layer (FiLM) conditioning, delta-vs-absolute
ablation, DDPM vs OU at MATCHED lag, with the step-1 conditioning guard as a standing
report, and steps-to-1ms (= 1e6/tau_ns) at each tau.

## Step 3 result: strengthened benchmark discriminates; learned model captures coupling, collapses marginals

Reference has real structure OU cannot produce: xcorr 0.15-0.20, amplitude coupling
0.09-0.10, kurtosis 0.5-0.6. The "OU is enough" branch does NOT fire -- learning target
is real. Two systems, tau=1/10/50/100, FiLM conditioning, delta/absolute.

| model | marginals (std) | cross-mode xcorr (ref 0.15-0.20) | amp (ref ~0.1) | kurt (ref 0.5-0.6) | IAT ratio | steps->1ms |
|---|---|---|---|---|---|---|
| OU | 1.2-1.4 (good) | 0.02 (FAILS) | 0.02 (FAILS) | ~0 (FAILS) | ->1 at long tau | 1e6..1e4 |
| DDPM-absolute | 0.17-0.45 (COLLAPSE) | 0.24-0.33 (captures) | 0.12-0.19 (captures) | 0.2-5.6 (variable) | ->1 at long tau | 1e6..1e4 |
| DDPM-delta | 5.4 / unstable | 0.05-0.25 | 0.79-0.97 (spurious) | -1.8..15 | 0.1-12 (wild) | -- |

**Findings:**
- The strengthened benchmark WORKS: OU provably fails cross-mode coupling + amplitude
  coupling + non-Gaussianity (all ~0 by construction), while the reference has them.
  So OU is a floor on marginals/kinetics but NOT a complete model -- a learned
  propagator is needed for the coupling. (Pre-registered "no learned needed" branch
  refuted.)
- **DDPM-absolute captures the coupling OU cannot** (xcorr 0.24-0.33 vs OU 0.02) and
  matches IAT at long tau -- but COLLAPSES the marginals (std 0.2-0.45). Neither model
  wins both: OU marginals, DDPM coupling+kinetics.
- **delta parameterisation is unstable** (random-walk: std 5.4 at tau=1, IAT 12x at
  tau=10) -- confirms the delta+drift prediction; absolute is the better parameterisation.
- **Longer tau helps** (IAT ratio 0.14->~1 from tau=1->100 for both OU and DDPM-absolute)
  AND cuts steps-to-1ms from 1e6 to 1e4 -- accuracy and objective 3 improve together.
- Conditioning guard: DDPM-absolute cg 0.6-1.6 (live); delta cg 1.6-5.8 (over-amplifying).

**Next for the propagator:** absolute-parameterisation, long-tau DDPM already gets
coupling + kinetics right; the remaining failure is the marginal variance collapse
(the step-1 persistence shortfall). The concrete target: add a variance/marginal-
matching term (or noise-scale prediction) so a single model matches marginals AND
coupling AND IAT -- then it beats OU on the discriminators OU cannot reach.
