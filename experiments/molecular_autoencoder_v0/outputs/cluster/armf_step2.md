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

**This 1000-step in-manifold result is the strongest positive the project has
produced.** Error accumulation over a long rollout was THE central risk for
objective 3 (a latent that walks off its manifold decodes to garbage); the
diffusion model instead samples an ensemble -- 0% of steps leave the training
coefficient range, geometry stable to 1000 steps. Two on-record predictions
(coefficients drift out of range; geometry degrades start->end) were REFUTED. The
same-system caveat stays attached: this is stability of the mechanism, not yet
cross-system generalisation.

## General config: same rollout with the ANM (general) codec -- stability degrades

Swapping the per-system PCA codec for the general ANM codec (structure-only, no
fit) is the actual objective-1 configuration. The strong same-system stability does
NOT transfer:
- 2cndA01: 13% of steps have a coefficient outside the training range (vs 0% PCA),
  5/64 modes ever out, worst excursion 1.71sd; minCA 1.67 (severe clash) at start.
- 2e2dC02: **84% of steps out of range, ALL 64/64 modes leave range**, worst
  excursion 6.85sd (max 19.3); minCA **0.64** (atoms overlapping) at start.

The raw ANM-decoded structures clash badly at t=0 already -- consistent with the
usability test (ANM needs relaxation to be valid) -- and the DDPM in the ANM-latent
space is far worse conditioned than in the trajectory-fit PCA space. So the
ensemble-sampling stability was a same-system-PCA property; the general codec's
rollout is materially less stable and would need the mandatory relaxation step
(8.1) to be usable. Honest tempering of the positive.

### Whitening control -- the negative is not a scale mismatch

The obvious explanation for "all 64 modes out" is a per-system normalisation
mismatch. It is ruled out:
- **Whitening is already in the pipeline.** Coefficients are standardised per
  system, per mode (`zmu, zsd = Z[:h].mean/std`; the DDPM sees unit variance;
  un-whitened at decode). The 13%/84%-out-of-range figures are the WHITENED result,
  and the DDPM is trained per system, so there is no cross-system scale mismatch to
  begin with. The negative survives the obvious explanation.
- **Concentration:** excursions are spread across ALL 64 modes (64/64 ever-out on
  2e2dC02) with only a mild low-index tilt (low 2.53 vs high 2.07 sd) -- not
  concentrated in a few soft modes.
- **Mechanism (correlation, not scale):** mean |off-diagonal coefficient
  correlation| on the training half is **PCA 0.000 vs ANM 0.23-0.27**. PCA
  diagonalises the covariance (decorrelated latent); ANM does not (correlated
  latent). Per-mode whitening removes scale but NOT correlation, and the per-mode-
  conditional DDPM handles the correlated ANM latent worse. So the general-codec
  rollout instability is a codec-quality property (ANM does not decorrelate), not a
  normalisation artifact -- the negative is real and stronger for surviving the
  control.
