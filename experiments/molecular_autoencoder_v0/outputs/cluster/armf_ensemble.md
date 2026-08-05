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

## Variance-collapse diagnosis: PURE SCALE -> a per-mode calibration is the fix

Guidance off (scale 1.0). At FULL training every sampler collapses (varRatio 0.23-0.40):
posterior variance (beta_tilde vs beta) identical, reverse-step sweep 25/50/100/250 FLAT,
eta=0 same. So steps 2a/2b (the standard sampler causes) are RULED OUT -- it is a model-
level per-mode variance deficit, not a sampler config.

Step 1 (per-mode rescale to training std) is decisive: varRatio -> 1.00, marginal JS ->
0.02, while cross-mode coupling is preserved (xcorr 0.33/0.26, scale-invariant) and basin
transitions are RECOVERED (3jvv 129 -> 571, 3a5z 736/738 matching ref 738/740). Everything
else passes -> **PURE SCALE**. Per the pre-registration, a fixed per-mode CALIBRATION
(target = training-set per-mode std, applied post-hoc to the generated coefficients) is a
legitimate fix -- same class as the whitening already in the pipeline. Reported AS a
calibration, explicitly, not folded in silently.

## FIRST COMPONENT TO PASS A REAL ACCEPTANCE TEST (not a null comparison)

With the calibration, the learned propagator vs the OU floor at tau=50:

| metric (ref) | OU | calibrated DDPM | who wins |
|---|---|---|---|
| marginals varRatio (1.0) | 1.17 / 1.40 | 1.00 / 1.00 | tie (both pass) |
| cross-mode xcorr (0.20/0.15) | 0.02 (FAILS) | 0.33 / 0.26 | **DDPM** (OU cannot) |
| kurtosis (0.5/0.6) | ~0 (FAILS) | 0.4 / 1.9 | DDPM (3a5z clean, 3jvv over) |
| IAT ratio (1.0) | 0.95 / 0.97 | 1.10 / 1.06 | tie (both pass) |
| basin trans (738/740) | 747 / 741 | 736 / 571 | 3a5z tie; 3jvv DDPM under (77%) |

**On 3a5zD02 the calibrated learned propagator passes marginals + coupling + kinetics +
transitions AND beats OU on the two discriminators OU provably cannot reach (cross-mode
coupling, non-Gaussianity). That is the first component in this project to clear a REAL
acceptance test rather than a null comparison.** On 3jvvA01 it passes marginals + coupling
+ kinetics and beats OU on coupling, but under-shoots the basin-transition rate (571 vs
740) and over-shoots kurtosis (1.9 vs 0.6). So: a clean pass on one system, partial on the
other. The remaining work is per-system CONSISTENCY (why 3jvv under-transitions), not a
redesign -- the mechanism works.

## The shape of what's left (structural, not tuning)

- **Propagator:** calibrated + long-tau, it passes the acceptance test (cleanly on 3a5z);
  next is per-system consistency and more systems, then a proper GPU training run. The
  variance deficit is a known DDPM behaviour, fixed by the calibration; a training-time
  fix (loss weighting / v-prediction) is optional polish, not required.
- **Objective 1 (general molecules) STILL BLOCKED:** ANM is protein-biased (38-49% of
  ceiling on ligands vs ~2/3 on proteins). Internal-coordinate / torsional representations
  (arXiv:2101.01618) are the indicated non-protein codec -- the next STRUCTURAL gap.
- Parked: matrix-free LOBPCG eigensolver (O(N^2) shift-invert measured); reactive corpus.

## Prediction on record + pre-registered follow-up (before the deficit-vs-mode-index result)

**Prediction:** the variance deficit will be STRONGLY MODE-DEPENDENT, concentrated in the
SLOW modes -> the global-constant calibration FAILS. Derivation from step 1's own numbers:
step 1 measured a ~20% persistence shortfall (a_ddpm ~= 0.8 a_ref). For AR(1), stationary
variance ~ 1/(1-a^2), so the SAME shortfall gives very different variance deficits by mode:
  slow  a_ref 0.96 -> a_ddpm 0.77 : variance ratio 0.19
  fast  a_ref 0.30 -> a_ddpm 0.24 : variance ratio 0.97
The observed aggregate 0.23-0.40 is what slow-mode-dominated averaging produces. If the
deficit-vs-mode-index shows this shape, a single constant is the WRONG correction.

**Pre-registered follow-up (conditional on a slow-mode-weighted deficit) -- preferred over
training-time reweighting:** PREDICT THE RESIDUAL FROM A PER-MODE OU BASELINE, not the state.

    z_{t+1} = OU_step(z_t) + f_theta(z_t, noise),   f trained on the residual after OU.

OU reproduces per-mode persistence + marginals EXACTLY (by construction), and the learned
model already captures the cross-mode coupling OU provably CANNOT. Compose them: the
denoiser never has to learn near-unity autocorrelation (precisely what it fails at), so the
persistence/variance-collapse problem DISAPPEARS rather than being reweighted around.
Physics floor + learned correction -- the same shape as ANM+learned on the codec side,
except here the learned part adds something OU cannot reach, which is why it is worth
building. Cheaper than loss reweighting and better motivated. Build it IFF the deficit is
slow-mode-weighted; report the deficit-vs-mode-index against the prediction first.

## Pre-registered before the scale-up (locked; post-hoc thresholds would be worthless)

### If the OU-residual hybrid is built: zero-residual control from the start
z_{t+1} = OU_step(z_t) + f_theta(z_t, noise). Force f_theta's output to zero -> recovers
plain OU EXACTLY. Same guard structure as epoch-0==ANM (spring map) and epoch-0==S+SFC
(P-init): the hybrid cannot do worse than the physics floor at init, and any degradation
is then diagnostic, not ambiguous.
- ASSERT zero-residual output == plain OU on every discriminator, within noise.
- LOG the full discriminator set at EVERY checkpoint, not just at the end.
- If ANY discriminator goes BELOW its zero-residual value during training -> STOP and
  report: the learned part is actively hurting (the exact pattern P-init showed --
  monotonic degradation off its warm start).

### Scale-up pass thresholds (PRE-COMMITTED, 10-20 held-out systems)
Per-system pass/fail per discriminator against these fixed thresholds; then the FRACTION
passing each (never a mean over systems). Thresholds printed at the top of the table.

| discriminator | PASS if |
|---|---|
| marginals (varRatio) | 0.80 <= varRatio <= 1.25 |
| cross-mode coupling (xcorr) | 0.5*ref <= gen <= 2.0*ref |
| kinetics (IAT ratio) | 0.50 <= IAT_gen/IAT_ref <= 2.00 |
| basin transitions | 0.50 <= gen/ref <= 2.00 |
| non-Gaussianity (kurtosis) | \|kurt_gen - kurt_ref\| <= 0.5 ABSOLUTE (ratio unstable, ref kurt ~ 0) |

**Second, SEPARATE criterion -- "earns its cost":** does the learned model BEAT OU on the
two discriminators OU provably CANNOT reach (cross-mode coupling, non-Gaussianity)? A
system can pass the thresholds while adding nothing over OU; different claims, both in the
table. Report the FRACTION of systems where the learned model beats OU on each of those two.

Report format: per-system rows (pass/fail per discriminator) -> fraction passing each ->
fraction beating OU on the two OU-impossible discriminators. Thresholds stated at the top.

## CORRECTED: variance collapse is a GLOBAL constant -> a general one-number fix (slow-mode prediction refuted)

Corrected deficit measurement (gen.std/ref.std, raw space; the earlier whitened-space
number was confounded by sqrt(kT/lambda) != empirical):

| system | true deficit (mean) | slow | mid | fast | whiten ratio (emp/sd) |
|---|---|---|---|---|---|
| 3a5zD02 | 0.36 | 0.36 | 0.36 | 0.35 | 2.26 |
| 3jvvA01 | 0.32 | 0.31 | 0.31 | 0.33 | 2.30 |
| 3a9lA00 | 0.29 | 0.27 | 0.28 | 0.31 | 3.37 |
| 2z1kA02 | 0.37 | 0.32 | 0.38 | 0.42 | 4.28 |
| across-system | 0.33 +-13% | | flat vs mode index | | 2.3-4.3 |

- **Slow-mode prediction REFUTED:** deficit is FLAT across modes (~0.33), not the predicted
  ~5x slow-collapse (0.19 slow / 0.97 fast). Weak ~10-30% slow tilt at most. The AR(1)-
  persistence mechanism is not the dominant cause.
- **The ANM per-mode variance error CANCELS in whiten->un-whiten:** whitening by sd
  injects the CV-0.42 error into Zn, the model learns Zn's shape, un-whitening by sd
  removes it -> gen_none has the CORRECT per-mode variance shape (hence deficit flat).
- **What remains is a single global under-scale ~0.33** (the DDPM's intrinsic under-
  production when trained on non-unit-variance data), ~constant across systems (+-13%).
- **A single global constant (x3 = 1/0.33) restores marginals:** varRatio 0.87/0.96/1.08/
  1.11 -> all inside the pre-committed [0.80,1.25]. ZERO per-system parameters. GENERAL.

**Correction to the prior "no general calibration works" conclusion: withdrawn.** It was
based on the script's global-const/anm-permode variants, which targeted ANM's (wrong) sd,
not the global under-scale. The units-rescue intuition was right in spirit -- a global
constant -- sourced from the DDPM's under-production, not ANM's sd. The OU-residual hybrid's
motivation (slow-mode persistence) is also refuted; not needed. Next: confirm the global
constant on 10-15 systems (scale-up), per-system pass/fail against the pre-committed
thresholds, + earns-its-cost vs OU.
