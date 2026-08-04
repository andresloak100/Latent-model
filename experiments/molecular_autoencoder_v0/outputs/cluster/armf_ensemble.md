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
