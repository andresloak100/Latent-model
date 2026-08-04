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

_Modal sweep (job 10279481) DONE: 0.8-1.5% across all L x drop cells (armF ~=
zero-latent). Cross-system training collapses the single-system ~10% to ~1.5% --
the transferability gap (learning a SHARED static->modes map across topologies is
far harder than a per-system fit), below even the ANM baseline. Secondary to the
leak-free ceiling, but it separates the failure into three measured layers:
linear ceiling ~40% (binding), transferability 10%->1.5%, optimisation below ANM._

## Collective-vs-jitter split (leak-free, matched L, absolute A alongside %)

| atom set | null A | L8 | L16 | L32 |
|---|---|---|---|---|
| CA-only | 1.64 | 44% (0.89A) | 47% (0.83A) | 51% (0.77A) |
| backbone N,CA,C,O | 1.65 | 43% (0.90A) | 47% (0.84A) | 50% (0.79A) |
| **all-atom** | 2.01 | 37% (1.23A) | 40% (1.19A) | **42% (1.15A)** |
| side-chain | 2.32 | 35% (1.48A) | 37% (1.44A) | 39% (1.39A) |

Side-chain = **66% of all-atom displacement variance** (57-71%). C-C bond ~1.54 A.

**These are LINEAR ceilings (PCA-optimal), not the codec's ceiling** -- a nonlinear
/ time-lagged (VAMP) encoder can exceed PCA on MD, usually not by 2x. So 42% is a
serious negative signal, not a hard wall.

**Absolute terms are worse than "just under a bar":** every residual is chemically
invalid -- backbone L32 = 0.79 A (half a C-C bond), all-atom 1.15 A, side-chain
1.39 A. Linear compression into 16-32 modes yields chemically meaningless local
geometry at ANY threshold, backbone included. Report absolute A next to the %.

**Task-misspecification hypothesis (collective + resample-able jitter): only WEAKLY
supported, not confirmed.** Backbone/CA (50-51%) is ~9 points above all-atom (42%)
-- the predicted direction but modest, not "well above". And side-chain is NOT
incompressible: 39% at L32 (pure per-atom jitter would be ~0%), so side-chain
motion is substantially collective too. On this evidence, do NOT pivot to the
collective-in-latent / jitter-resampled architecture -- the numbers separate too
weakly, and this was to be a MEASURED property, not a guess.

**The 50% rule STAYS FAILED** on this evidence (all-atom 42% linear, 1.15 A). Its
threshold inherited the unjustified tau=0.5 A -- recorded as weak provenance, NOT
as grounds to relax it after seeing the number. Whether the rule measured the
right quantity is exactly what the CA/backbone/all-atom split probes; the split
says backbone is only marginally better, so the rule was not merely mis-targeted.

**Narrowed conclusion:** Arm F establishes only that **80 ps-8 ns all-atom thermal
fluctuations are not linearly compressible into 8-32 modes at chemical accuracy.**
It does NOT establish that slower conformational dynamics are incompressible --
MISATO is 100 frames / ~8 ns, dominated by thermal fluctuation and small
within-basin motion. More modes on this short signal (L=64) will not change the
chemical conclusion (the all-atom curve is near saturation at 37/40/42%), so
mdCATH is NOT primarily an L=64 test.

**mdCATH is the next test -- for TIMESCALE REGIME.** Question: did Arm F measure
the wrong dynamical regime? The MISATO 1 ns stride control is **RETRACTED** (8
frames is too few for PCA). The proper control is WITHIN ONE mdCATH trajectory:
**A = 50 frames over ~50 ns (~1 ns stride), B = 50 frames over ~464 ns (~9 ns
stride)** -- frame count, rank ceiling, system identity, PCA conditions all
matched, only the time window changes. Run the leak-free tier table
(CA/backbone/all-atom/side-chain, L=8/16/32, % + A) for A and B.
Pre-registered (see `armf_mdcath_prereg.md`): **B >> A** -> slow collective
dynamics are more compressible, MISATO tested the wrong regime, the codec question
reopens; **B ~= A** -> the negative generalises across timescale, a stronger
verdict. High-temperature (450 K) trajectories are analysed SEPARATELY for the
transition/unfolding question, not mixed into the equilibrium tier table.

## mdCATH matched-window RESULT (28 domains, leak-free)

**A vs B: the negative generalises across timescale -- now on the CORRECT
(fractional) metric.** Metric correction: fractional % answers "is the signal
compressible" (the hypothesis); absolute A answers "is the output usable". B's
larger absolute residual just follows mechanically from its larger null, so it
does NOT by itself refute the hypothesis. The fractional numbers do: **A%~=B%**
(all-atom L32 A@0 47% vs B 46%; CA 58% vs 56%) -> slow dynamics are NOT more
compressible -> timescale hypothesis DEAD, locked negative stands. (Absolute for
reference: all-atom L32 A@0 2.13A vs B 2.75A, nulls 4.66 vs 5.83.) CA same (A@0 1.41 vs B 1.98A). A-offsets
tight (all-atom L16 2.20/2.00/2.08 at 0/200/400 ns) -> A@0 not inflated by
start-structure relaxation. Slow (500 ns) dynamics are NOT more compressible in
absolute A than fast (50 ns): the wrong-regime hypothesis is refuted, a stronger
negative.

**Window C (separate, not rank-matched): the L=64 saturation prediction was
WRONG.** Full trajectory (500 fr, 1 ns stride, 250/250 split): all-atom L64 =
**63% (1.79A)** vs predicted 43-44%; CA L64 = 77% (0.96A), backbone 75% (0.99A),
side-chain 61% (2.16A). The binding variable is **modes/rank + stride** (C's fine
stride + 250-frame fit vs B's 10 ns + 25-frame fit), NOT timescale -- MISATO's
100-frame rank cap hid it. The "don't bother with L=64" call is overturned.

**But all-atom is still chemically invalid** (L64 1.79A > 1.54A C-C bond).
CA/backbone at L64 reach **~1A (near-chemical)** while side-chain stays 2.16A ->
this **re-supports the collective-vs-jitter split more strongly than MISATO did**:
the backbone collective is nearly recoverable at L64, the side-chain residual is
what keeps all-atom invalid. Net: timescale does not rescue the codec, but modes
(L=64) + fine stride do far more than believed, and the codec question reopens on
the MODES axis (and the backbone/side-chain split), not the timescale axis.

**Nonlinear rerun on mdCATH (250-frame Window C, per-system): inconclusive AGAIN,
new reason.** AE underperforms PCA at every L (all-atom L64 AE 46% vs PCA 63%; CA
L64 55% vs 77%). With 250 frames it is no longer overfitting -- the minimal AE
simply fails to reach the LINEAR baseline (a proper AE can represent PCA, so it
should match it), i.e. an optimisation/architecture limitation, not a statement
about the nonlinear ceiling. The linear-ceiling caveat stays OPEN; the clean test
is a PCA-INITIALISED AE (starts >= PCA, so any improvement is real nonlinear
structure). Deferred, not declared.

**Nonlinear probe (per-system, MISATO): INCONCLUSIVE.** A small nonlinear AE
(per-system, no cross-system sharing, no static shortcut, operating in the
train-PCA span, matched split/L) UNDERPERFORMS PCA at every L (all-atom L32: PCA
42% vs AE 34%; CA L32: 51% vs 39%) -- but that is 50-frame OVERFITTING, not a
nonlinear ceiling. Per pre-registration this does NOT count as a second
independent negative; the linear-ceiling caveat stays open and the nonlinear
question needs mdCATH's frame count.
