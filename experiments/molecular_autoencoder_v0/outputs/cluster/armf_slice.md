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

**The first oracle LEAKED** (PCA fit on the frames it was scored on; at T=100 the
displacement has rank <=99, so PCA-64 keeping 64 of <=99 directions is mostly rank
arithmetic). Redone LEAK-FREE (fit PCA on frames 1-50, evaluate on 51-100), and
against a structure-only baseline (ANM modes from the reference contact graph, no
MD). Reduction vs the zero-displacement null:

| leak-free reduction | L=8 | L=16 | L=32 |
|---|---|---|---|
| **T1 all-atom** (split PCA -- the codec's actual target) | ~37% | ~40% | ~42% |
| T1 CA-only | ~44% | ~48% | ~51% |
| **T3 ANM** (structure-only contact-graph modes, CA) | ~20% | ~26% | ~32% |

**Viability is NOT established at >=50%.** Leak-free, the *optimal per-system
linear* ceiling for all-atom displacement (what arm F predicts) is **~40% at
L16-32, below the rule**. The 79-87% was leak. The T=50-frame fit caps modes at
<=49, so the rule's **L=64 target is untestable leak-free here -- it needs
mdCATH's 464 frames** (§6.6). Displacement *looks* moderately compressible
(CA-level ~50% at L32), not the ~80% the leaky oracle implied.

**T3 is the useful signal:** structure-only ANM modes capture **~43-91% of the CA
leak-free ceiling** (~2/3 mean) -- the contact graph predicts a substantial chunk
of the collective modes. So the modal bottleneck has a non-learned floor, and ANM
modes are a concrete init/feature. The trained modal AE (~10%) sits below even the
free ANM baseline -> a real optimisation gap, but the **binding constraint is the
~40% all-atom ceiling, not the optimiser**.

**Corrected mechanism (general, not a one-off bug):** the encoder does see the
target frame; the failure was that the CA-aligned displacement field is
**zero-mean**, and a global mean/attention pool averages ANY zero-mean field to
~zero -> frame-invariant latent. This is a **design constraint on every future
pooling choice**, not a wiring typo. The modal bottleneck (per-atom mode weights,
no mean-pool) avoids it and is order-invariant.

**Bottlenecks tried:** attention pool (collapses to zero-mean, void); region-mean
(bad basis, ~30% ceiling); modal (correct structure, but the all-atom ceiling is
~40%). **Redo the whole tier table on mdCATH** once ingested -- 464 frames removes
the rank cap and makes the L=64 question answerable leak-free.

_Modal sweep (job 10279481) finishing; its trained numbers are secondary now --
the leak-free ceiling, not the optimiser, is the binding result, and it is below
the 50% bar at MISATO's frame budget._
