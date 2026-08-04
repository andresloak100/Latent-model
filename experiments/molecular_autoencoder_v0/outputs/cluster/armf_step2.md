# Step 2: first end-to-end compress -> diffuse -> decode skeleton (RUNS)

The largest unexamined risk in the project: nothing has ever run compress ->
diffuse -> decode on MD. This is a SKELETON that RUNS, not a good model.

**Design** (per system, `scripts/armf_step2_diffusion.py`):
- compress: Kabsch-align backbone trajectory, PCA-64 of the displacement (the only
  arm-F signal that cleared the gate: backbone L64 ~0.97A / 75% on mdCATH). -> z_t.
- diffuse: small CONDITIONAL DDPM p(z_{t+1}|z_t) (Tdiff=100) on the latent's
  temporal transitions.
- decode: z -> displacement -> backbone coords.
- validation WITHOUT side chains: CA-CA distances, backbone bond geometry,
  backbone clashes, and DRIFT over a 50-step rollout (not RMSD).

**Result (2 mdCATH domains, 320K):**

| metric | 2cndA01 | 2e2dC02 | native |
|---|---|---|---|
| finite / runs end-to-end | yes | yes | -- |
| CA-CA consecutive (A) | 3.67 | 3.66 | ~3.80 |
| adjacent backbone-atom (A) | 1.49 | 1.48 | ~1.3-1.5 |
| CA drift 1->50 steps (A) | 1.17->1.54 | 1.02->0.99 | ~1-3, no blow-up |
| min CA-CA / clash (A) | 2.63 | 1.37 | >~3.7 |

**Load-bearing risk CLEARED:** the pipeline runs end-to-end, produces finite
structures, and the rollout is STABLE (drift ~1-1.5A over 50 steps, no
divergence). Geometry is roughly plausible (CA-CA ~3.66, bonds ~1.48) but has
CLASHES (min CA-CA 1.4-2.6A << 3.7) -- a quality issue expected of a skeleton, not
a viability blocker. The mechanism works; making it good is the next arc.

**Next (quality, not viability):** reduce clashes (a geometry/energy term or a
better codec), more systems + longer horizon, and the PCA-initialised-AE result
(whether a nonlinear codec beats PCA-64) feeds back into the compress stage.

## Strategic read: physics solved the codec, so the learning belongs in the propagator

ANM beats every learned bottleneck at matched scalars (armf_perceiver_segment.md),
generalises by construction, costs zero parameters, and its eigenvalues even supply
the rollout whitening (sqrt(kT/lambda)). The diffusion model -- the ONE place
learning is irreplaceable, since physics does not hand you a transition operator --
has had a single skeleton run. **We optimised the component that did not need ML and
barely touched the one that does.** This governs what gets built next: effort moves
to the propagator, and the first real test of it is ensemble validation of the
rollout (below), not more codec work.

## Long rollout (1000 steps) -- latents are the primary metric

Cartesian saturation is decoder-forced (output = ref + sum c_i v_i is confined to
the mode affine span), so the real obj-3 test is the LATENT coefficients vs the
training distribution. Over 1000 steps, 2 systems:
- **Steps with ANY coefficient outside the training [min,max]: 0%. Modes ever
  out-of-range: 0/64.** Per-step worst-mode excursion 0.51-0.57 sd (max ~2.5 sd).
  The conditional DDPM stays entirely inside the training manifold -- **it samples
  an ensemble, it does not accumulate error** (obj-3's binding failure mode does
  not appear here). Prediction that coefficients drift out of range: WRONG.
- Drift concentration ~uniform, marginally low-index (0.16-0.18 vs 0.15-0.16 sd) --
  no runaway band.
- Geometry START->END stable: CA-CA 3.78->3.68 / 3.71->3.65 (native ~3.80),
  bb-bond 1.54->1.50 / 1.52->1.48, minCA 2.83->2.83 / 2.41->2.29. Does not degrade
  over 1000 steps (clashes persist but do not worsen). Prediction that geometry
  degrades: WRONG.
- Cartesian (decoder-forced, demoted): saturates ~1.4 A, as expected.

Caveat: same-system rollout (trained on this trajectory's transitions), so this
shows in-manifold stability, not cross-system generalisation. But for "ensemble
vs error-accumulation," the latent evidence is unambiguous: ensemble.

**WITHDRAWN as a positive (see armf_ensemble.md).** The 1000-step "0% out of range"
result was read as ensemble sampling / the project's strongest positive. Ensemble
validation showed it was VACUOUS: generated per-mode variance is ~1/3 of reference
(collapse), autocorrelation time ~6x too short (wrong kinetics), only 18-47% of
reference free-energy basins covered. A collapsed narrow blob trivially stays in
range, so "0% out of range" is necessary but not sufficient and this model fails the
sufficient test. Kept below for the record, but read it as "did not diverge", no more.

## General config: ANM (general) codec rollout -- WITHDRAWN negative (whitening artifact)

Swapping the per-system PCA codec for the general ANM codec (structure-only, no
fit) is the actual objective-1 configuration. With the DEFAULT per-mode empirical
diagonal whitening it LOOKED unstable:
- 2cndA01: 13% of steps outside the training range, 5/64 modes ever out, excursion 1.71sd.
- 2e2dC02: **84% out, ALL 64/64 modes out**, excursion 6.85sd; minCA 0.64 at start.

**This was a NORMALISATION artifact, not a codec property -- WITHDRAWN.** The layered
whitening control below collapses the out-of-range rate to ~0% with proper whitening,
so the "general-codec rollouts are unstable" conclusion does not stand.

### Whitening control (diag vs full ZCA vs structure-derived ANM)

Which whitening is in the default pipeline: **empirical, per-system, per-mode std**
(`zsd = Z[:h].std(0)`). So the default step-2 consumes the held-out system's own
trajectory for whitening (and the DDPM is per-system) -- less general than "general
codec" reads. Layered comparison (2 systems, ANM codec, horizon 1000):

| whitening | 2cndA01 out-of-range | 2e2dC02 out-of-range | excursion |
|---|---|---|---|
| diag (empirical per-mode) | 13% | **84%** (64/64) | 1.7 / 6.9 sd |
| zca (empirical FULL covariance) | **0%** | **0%** (1/64) | 0.5 sd |
| anm (structure-derived sqrt(kT/lambda), ZERO trajectory) | **0%** | **0%** (1/64) | 1.2 sd |

**Read:**
- Full empirical ZCA collapses the rate to ~0% -> the instability was normalisation
  (the diagonal whitening pathologically amplifies low-variance / stiff ANM modes,
  whose empirical std ~ 0).
- **The zero-parameter, structure-derived ANM whitening (1/sqrt(kT/lambda)) ALSO
  collapses it to ~0%** -- better than the pre-registered fork (which expected only
  empirical ZCA to work). So the fix needs NO trajectory: ANM's own eigenvalues give
  the correct per-mode scale. This also removes the whitening's trajectory-dependence
  -- the whitening component is now general (the DDPM training and the mean structure
  are still per-system; a fully general pipeline needs a cross-system DDPM).
- The earlier "correlation (PCA off-diag 0.000 vs ANM 0.23-0.27) is the mechanism"
  reading is superseded: the correlation is real, but a DIAGONAL structure-derived
  rescale (ANM) already stabilises the rollout, so the dominant fault was per-mode
  SCALE of low-variance modes, not the cross-mode correlation. Full ZCA is marginally
  tighter (0.5 vs 1.2 sd excursion) but both keep 0% out of range.

**Net:** the general-codec (ANM basis) latent rollout is STABLE over 1000 steps once
whitened by structure-derived ANM eigenvalues -- 0% of steps leave the training
range, ~1.2sd excursion, same-system DDPM. The obj-3 ensemble-sampling result is not
confined to the per-system PCA codec after all.
