# Arm F — the thinnest end-to-end viability slice (results pending)

## Why this exists (reprioritisation)

Honest state before this: one working artifact (arm A, 0.365 A Ca, a reproduction
of an existing reference), every arbitrary-L arm refuted at 1.81-3.38x, zero
diffusion models trained, zero trajectories predicted, stage 1 of 4 unfinished --
and most of that failure was measured on the **wrong task**. Arms A-E compress
**static structure**; the deployed design keeps reference geometry as static
conditioning so the latent only ever carries **deviation** (~360x smaller). This
slice tests the load-bearing, entirely-unexamined uncertainty: does
compress -> decode of *displacement* transfer to MD at all.

**This is a VIABILITY TEST, not progress on any objective.** It runs at ~4,000
atoms not 1e6, nanoseconds not milliseconds, and zero reactions. It tests the
mechanism that *could* reach those, nothing more.

Parked (kept in docs, built none): all further §7 metric refinement, the FF noise
sweep, the graph-vs-group OOD test, skip-masking A/B, group-distance denoising,
coordinate-parameterisation sweep, spherical ablation. None block this slice.

## Design (arm F)

- **Static conditioning to encoder AND decoder**, never through the bottleneck:
  element, reference geometry (frame 0), graph identity (canonical rank from the
  bond graph).
- **Through the L fixed slots: per-atom displacement only.** L = 16 and 64,
  independent of N.
- **Local-frame output**: displacement predicted in a per-atom frame built from
  the reference bond geometry (equivariant), with a global-aligned fallback flag.
- **Conditioning dropout** (validity requirement): contiguous regions of the
  static conditioning are masked in training (reusing `masking._region_mask` over
  canonical order) so the decoder cannot echo reference geometry -- dropped
  regions must route through the latent. Swept 0 / 0.25 / 0.5; rate 0 is the
  control that shows the effect is real.
- **Latent noise** (validity requirement, for step 2): Gaussian perturbation of
  the latent in training; eval reports RMSD vs perturbation magnitude -- the
  decoder's robustness budget, which tells the step-2 diffusion model how accurate
  it must be. (Latent LDMs regularise/noise-augment for exactly this reason.)

Data: 8 MISATO systems spanning N=1,273-7,823 heavy atoms, 100 frames each,
CA-aligned so displacement = internal deviation (mean 1.24-2.30 A). Train on
frames 1-79, val on 80-99. `scripts/build_armf_data.py` + `scripts/armf_slice.py`
+ `slurm/armf_job.sh`.

## Baselines, controls, measurements

- **Headline baseline: the zero-displacement null** (predict the reference
  unchanged). Absolute RMSD is meaningless without it; improvement over null is
  the number.
- **Zeroed-latent control**: proves the latent, not the conditioning, does the
  work (armF should be far below zero-latent).
- **Per-region conditioning reliance**: mask a conditioning region at eval,
  localise the degradation -- which regions the decoder recovers from conditioning
  vs from the latent.
- **Slot specialisation (MI vs molecule identity)**: DEFERRED -- needs >1 molecule
  in the box; single-system batches here have nothing to specialise across. Noted
  for the multi-molecule step.

## Pre-registered decision rule

**>= 50% reduction vs the zero-displacement null (~< 0.5 A) at L=64, holding as N
varies -> proceed to step 2** (train a small latent diffusion model on short
MISATO segments, decode, check rollout plausibility). Below that -> the codec is
the blocker, stated plainly rather than patched with mechanisms.

## Non-protein (objective 1's only unexamined part)

MISATO is protein-ligand only; there is **no non-protein MD trajectory data**, so
general-molecule *dynamics* is **corpus-blocked, not architecture-blocked**. The
static path (arms A-E, `sdf.py` reads V2000) already handles general-molecule
*structure*. A labelled non-protein static-reconstruction control is a small
follow-up; the dynamics gap is a data problem, recorded here so it is not misread
as an architecture failure.

## Update: first sweep VOID (wiring bug); oracle says the codec IS viable

The first sweep (job 10279117) read ~1-2% below null in every cell and was
**voided** by the decisive test (encode different frames of one system, compare
latents): **cosine 1.0000 across frames** -- the latent was frame-invariant, i.e.
it carried the reference, not the displacement. Cause: the trajectory is Kabsch-
aligned, so the per-atom displacement field has ~zero mean, and a global attention
pool averages it to that mean. Do not interpret any of those 6 cells.

**Oracle ceilings settle viability (no training, pure PCA of the displacement):**

| reduction vs null | PCA-16 | PCA-64 | region-mean-16 | region-mean-64 |
|---|---|---|---|---|
| across 8 systems | 55-71% | **79-87%** | 7-16% | 18-37% |

Displacement is **highly L-compressible** -- PCA-64 ~80% (matches the premise
check's 54 modes/90%), and even **PCA-16 clears the 50% rule**. So the **codec is
NOT the blocker**; viability is achievable. The two failed bottlenecks were both
wrong: the attention pool collapses to the ~zero mean (void); region-mean pooling
carries displacement but is a bad basis (caps ~30%).

**Fix: a modal bottleneck** -- L learned global mode-shapes (per-atom weights from
static conditioning), latent = mode coefficients (PCA-like), order-invariant so it
transfers to general molecules. Verified the wiring is now live (latent varies
with frame; armF < zero-latent). But a quick 1-system/250-epoch test reaches only
~10% -- an **achievable-vs-achieved gap**: the oracle says ~80% is there, the
trained minimal AE has not reached it. Full sweep rerunning on the modal
bottleneck to measure the real trained numbers against the rule; the viability
QUESTION is answered YES by the oracle, the remaining work is optimisation to
reach it.

_Rerun in progress (modal bottleneck)._
