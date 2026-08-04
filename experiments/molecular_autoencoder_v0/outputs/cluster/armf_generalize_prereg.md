# Pre-registration: a codec that GENERALISES across systems (before the numbers)

The demo->model gap. The working codec is per-system PCA, fit on each system's own
trajectory -- given an unseen molecule there is no codec, so objective 1 (general
molecules) is blocked on generalisation. The one prior attempt (learned modal
bottleneck, end-to-end, cross-system) collapsed to 1.5%.

Two measured facts make this better-posed now:
1. PCA is the right answer (linear ceiling settled) -> supervise mode prediction
   DIRECTLY against per-system PCA modes (regression onto a known target), not
   end-to-end credit assignment.
2. ANM (structure-only) already gives ~2/3 of the CA ceiling with zero parameters
   -> START FROM ANM, learn the correction (residual), a far smaller ask.

**Method.** Per system: displacement (Kabsch-aligned), per-system PCA modes
M_pca (L=64, orthonormal) = supervision target; ANM modes M_anm (top-K=128 from
the contact graph) = structure-only starting basis. Shared learned map A (KxL),
same across systems: M_pred = orthonormalise(M_anm @ A). Generalises because A is
mode-index->mode-index (spatial-dimension-agnostic), applied to any system's own
ANM.

**Loss = SUBSPACE, not per-vector** (modes are defined only up to sign and up to
rotation within near-degenerate subspaces). Frobenius distance between projection
matrices, computed cheaply via the LxL overlap: loss = L - ||M_pred^T M_pca||_F^2
(both orthonormal) = L - sum of cos^2(principal angles). Stated: **projection-
matrix Frobenius / principal-angle overlap.**

**Level: CA first** (ANM is natural at CA and cheap; PCA showed CA ~= backbone,
77% vs 75% at L64). Backbone is the mechanical extension if CA generalises.

**Evaluation: HELD-OUT SYSTEMS** (never in training; split BY SYSTEM, not frame),
L=64, reconstruction residual in ABSOLUTE A, against three references:
- per-system PCA (ceiling, ~0.96A CA)
- ANM top-L (free non-learned baseline)
- zero-displacement null

**Locked read:**
- **learned ~= per-system PCA** -> generalisation solved; the codec is a model.
- **ANM < learned < PCA** -> learning adds real value; report the fraction of the
  ANM->PCA gap closed.
- **learned <= ANM** -> learning contributes nothing. **AMENDED (fractional-vs-
  absolute trap, 2nd occurrence):** ANM is a win ONLY if it clears the absolute
  accuracy bar. It does not. Every codec claim is now stated in ABSOLUTE A against
  the passing reference, never as a fraction of ceiling:
  - CA: ANM 1.91 A vs CA-PCA ceiling 0.84 A -- ~2.3x the ceiling error, does NOT clear.
  - Backbone gate (what step 2 cleared): per-system PCA 0.99 A. A general codec at
    ~2x that error is not "2/3 of ceiling is fine"; it is a codec that misses the bar.
  "Use ANM directly" stands only if ANM reaches ~1 A absolute, which it does not.

Cohort: the mdCATH 28 already downloaded. Train/test split by system.

**Training-set-size learning curve (added, before the result).** With ~20 systems
the "learned <= ANM" branch is ambiguous -- "learning adds nothing" vs "20 systems
too few for a structure->correction map." Train on 5 / 10 / 20 systems, report
held-out absolute A at each:
- still improving at 20 -> answer is PULL MORE DOMAINS (3,293 filtered available),
  NOT "learning doesn't help".
- flat 10 -> 20 -> the negative is real, ANM stands as the codec.

## Rollout metrics (added -- Cartesian saturation is decoder-forced)

With a PCA decoder every output = ref + sum(c_i v_i), so the structure is confined
to the modes' affine span and cannot wander while coefficients stay bounded --
Cartesian saturation describes the DECODER's geometry, not the diffusion model.
PRIMARY rollout metric = the LATENTS:
- per-mode coefficient mean/std/range over 1000 steps vs the TRAINING SET's per-
  mode statistics;
- fraction of steps where ANY coefficient leaves the training range;
- whether coefficient drift concentrates in low-index (slow, large-amplitude) or
  high-index modes.
Cartesian drift + CA-CA/geometry kept but DEMOTED; Cartesian saturation labelled
decoder-forced.

## Predictions on record (from planning)

- CODEC: ANM < learned < PCA, closing LESS THAN A THIRD of the gap, possibly
  within noise (the PCA target is fit on 250 frames, so its "ground-truth" modes
  carry sampling error that caps any learned map). If learned ~= PCA -> prediction
  WRONG, generalisation solved, report loudly (largest result of the project).
  **RESULT (CA, 20 train systems): WRONG, in a direction not considered.** Learned
  came in at 1.94 A, BELOW ANM (1.91 A) -- the shared map OVERFIT the training
  systems and generalised worse than plain ANM. Not "between ANM and PCA"; below
  ANM. Same overfitting failure as the PCA-init AE. Learning-curve + cross-replica
  gate (below) queued to decide whether the cause is too-few-systems or a noisy
  target.
- ROLLOUT: Cartesian drift SATURATES (decoder-forced, uninformative); coefficients
  DRIFT OUT of the training range; backbone bond/angle geometry DEGRADES start->end.

## GATE (run BEFORE the learning curve): is the supervision target real?

Two independent methods now overfit at this data scale (PCA-init AE; shared
ANM->PCA map, learned BELOW ANM). "Too few systems" is one reading -- the learning
curve tests it. The other: **the supervision target is noise.** Per-system PCA
modes fit on ~250 frames may not be a stable property of the STRUCTURE, only of
the particular trajectory. If so, more data cannot help -- the target is not a
function of anything the model can see. This is the reference we have optimised
against without ever checking it.

mdCATH makes it decisive and cheap: **5 independent replicas per domain at the
same temperature, already on disk.** Per domain (several, spanning the size range;
BACKBONE target to match the 0.99 A gate, plus CA to match the codec):
1. Fit PCA-64 on replica 0.
2. Reconstruct REPLICA 1's trajectory with replica 0's modes -> absolute A (cross).
3. Compare vs: PCA fit on replica 1 itself (within-trajectory ~1 A ceiling), ANM, null.
4. Principal-angle subspace overlap between the two replicas' L=64 mode sets.

**Locked read:**
- **cross ~= within** -> modes ARE a structural property; the generalisation
  problem is purely cross-molecule and a learned map has real headroom. Proceed to
  the learning curve.
- **cross >> within, near ANM** -> per-system PCA modes are largely TRAJECTORY-
  SPECIFIC. That is a HARD UPPER BOUND on any structure->modes codec: no map can
  beat the reproducibility of the modes themselves. It would also mean the 0.99 A
  gate was cleared by a codec that does not transfer across independent runs of the
  SAME molecule -> the step-2 milestone must be restated as a demo, and the
  learned-codec track needs rethinking, not more data. STOP and report before
  spending anything further on the learned map.

_Cross-replica gate pre-registered 2026-08-04, before it was run._

### RESULT (honest temporal split; within & cross scored on the SAME held-out frames)

8 domains spanning n_CA 56..482, backbone target (CA nearly identical):

| ref | absolute A (backbone) | meaning |
|---|---|---|
| null | 3.19 | do nothing |
| ANM | 2.23 | structure-only baseline |
| **cross** | **1.17** | independent replica's modes -> this run (honest, cross-run) |
| within | 0.93 | this run's own past -> its future (temporal split, ~= the 0.99 gate) |
| overlap | 0.60 | mean cos^2 principal angle between the two replicas' L=64 subspaces |

**Verdict: the STOP branch is REFUTED. The target is real.**
- cross 1.17 is nowhere near ANM 2.23 -- it closes **82%** of the ANM->within gap.
  Per-system PCA modes are NOT trajectory-specific noise; they transfer across an
  independent run of the same molecule.
- The 0.99 A step-2 gate is therefore **not a demo**: it transfers to an
  independent replica at 1.17 A.
- But cross != within: there is a real **+0.24 A (26%) reproducibility penalty**.
  Independent runs sample somewhat different subspaces (finite sampling of a rough
  landscape). That component is trajectory-specific and **unlearnable from
  structure**, so the honest ceiling for a GENERAL structure->modes codec is
  **cross ~1.17 A, not within 0.93** -- correcting the reference again.
- Headroom for a learned map is real: ANM 2.23 -> cross-ceiling 1.17, ~1.06 A to
  close. The current learned map (1.94 A CA, below ANM) is failing by OVERFITTING,
  not because the target is noise -> the learning curve is the right next test.
- Caveat: 4e6sA00 (n=85, high flexibility) is an outlier -- within 1.56, cross
  2.08, even ANM 4.05; it inflates the mean. Excluding it, within/cross/ANM =
  0.83/1.05/2.03, gap +0.22 A.
- Outcome sits BETWEEN the two pre-registered branches (cross neither ~=within nor
  near ANM); action is not auto-specified -- reported for the call.

## Pre-authorised continuation (locked before the learning-curve number)

**FROZEN HELD-OUT TEST SET (locked before any new domain enters the cohort).**
7 systems, n_CA 64..482: **3a5zD02, 2z1kA02, 3jvvA01, 3g7dA03, 2k4qA00, 3a9lA00,
3h7lB02** (`outputs/cluster/armf_frozen_test.txt`). New domains grow the TRAINING
POOL only; every curve point (5/10/20/40/60) is scored on exactly these 7, so
"still improving" cannot be a change of test set. Fresh domain selection is
asserted disjoint from this list.

**ONE branch proceeds WITHOUT waiting:**
- spring closes **>= 30%** of the ANM->cross headroom **AND** held-out A still
  improving from 10 -> 20 systems -> pull **~20 GB more** mdCATH domains (disjoint
  from the frozen test set), extend the curve to 40 then 60. Report the pull size
  (as before), then continue.

**Every other outcome reports and HOLDS:**
- spring gains but **FLAT 10->20** -> learning saturated; take the number.
- **spring ~= ANM** -> the physics prior is not improvable from structure at this
  scale; a real blocker for objective 1 (general molecules).
- **any guard trips** -> objective problem; report it, do not present the sweep.

## "Learning worked" != "codec is usable" (locked before the number)

Different questions; conflating them is the recurring error.
- **Headroom closed measures WHETHER LEARNING WORKED** -- does a structure->modes
  map beat the zero-parameter ANM baseline. That is all it measures.
- It does **NOT** establish that the general codec is **usable**. No absolute-A
  figure by itself declares objective 1 solved.
- Arithmetic: ANM 2.23 A, hard ceiling (cross) 1.17 A, step-2 skeleton ran at
  0.99 A **with clashes already appearing**. Closing 50% of headroom -> **~1.70 A**:
  a real ML result, still materially worse than the codec that barely passed.
  Reaching ~1.2 A needs **~85-95%** of the headroom closed.
- Per 8.1, **~1.2 A backbone is an unremovable floor**, so usability was never
  going to be settled by RMSD.

**NAMED FOLLOW-UP (a step-2 test, not a codec test):** usability is decided by
whether a **short local relaxation of the decoded backbone yields a physically
valid structure** -- backbone bond/angle within tolerance, no clashes, Ramachandran-
plausible -- NOT by its RMSD. This is the (b) relax-before-scoring fork (ROADMAP
8.1); run it on decoded frames once a general codec exists.

_Continuation + usability pre-registered 2026-08-04, before the learning-curve number._

### LEARNING-CURVE RESULT (frozen 7 test systems; CA, L=64)

Fixed refs: null 2.63, ANM 1.37, cross(ceiling) 1.01, within 0.78 (ANM->cross
headroom is only 0.36 A on this cohort).

**Guards (read first):**
- GUARD1 epoch-0 == ANM (1.37) at every train size, no wrong-way trip -> sign
  empirically correct. PASS.
- GUARD3 residual rigid-body content of the PCA target = 0.0000 -> tr(HC) clean. PASS.
- **GUARD2 TRIPPED: Pearson(spring train_loss, held-out A) = -0.30 over 15 ckpts.**
  The energy surrogate tr(HC)/tr(H) does NOT track top-64 held-out reconstruction
  (mildly anti-correlated). Per the lock, **the spring sweep is NOT presented as a
  valid result** -- "spring ~= ANM" cannot be read as "physics prior not
  improvable" while the objective is the wrong surrogate.

**Valid findings (not subject to GUARD2):**
- Per-mode cross-replica capture: modes 1-8 **0.88**, 9-16 0.81, 17-32 0.70,
  33-48 0.56, 49-64 **0.39** (mean 0.63). Confirms the leading high-variance modes
  reproduce far better than the tail -> variance weighting is justified (req. 2).
- **DIRECT variant (variance-weighted subspace loss, metric-aligned, valid):
  1.49/1.49/1.47 A at 5/10/20 -- WORSE than ANM (1.37) at every size (-30..-35%
  headroom).** Direct mode prediction overfits and loses to ANM even with variance
  weighting; corroborates the earlier learned-below-ANM finding.

**Continuation did NOT fire** (spring 6% << 30%) and GUARD2 tripped -> report and
HOLD; no auto-pull.

**Open question + pre-specified fallback.** The learner we can trust (direct,
metric-aligned) overfits and loses to ANM; the small-class learner (spring) has an
invalid objective. So "is the physics prior improvable from structure?" is
UNANSWERED. The pre-specified fallback -- a variance-weighted subspace loss on
leading modes for the SPRING variant -- is NOT a trivial reuse of the direct loss:
the direct loss acts on A@M_anm (no eigh), whereas scoring a subspace loss on the
spring READOUT requires backprop through eigh(H(g)). Recommended decisive test:
train spring params with the subspace loss via eigh-in-the-loop, cap training to
smaller systems for tractability (the spring map is per-residue/shared), eval
forward-eigh on the frozen 7. Held for the call.

### FINAL VERDICT: the codec track FINISHES at ANM (reframe + usability)

**ANM is not the fallback; it is the answer.** The general-codec track exists to
serve objective 1, and ANM generalises BY CONSTRUCTION: it needs only a contact
graph from a reference structure, so it applies to any molecule with coordinates
(protein or not) with no fitting and no training corpus. The learned map was trying
to beat a baseline that already satisfies the generality requirement -- losing to
it is the track finishing, not failing. On the frozen 7, ANM already captures 78%
of the null->ceiling range (2.63->1.01) with zero parameters; 0.36 A is the total
remaining prize.

**Usability (armf_usability.md) settles it.** After a short local relaxation,
ANM-decoded backbones are physically valid and ~= PCA: bonds 97% vs 100%, angles
94% vs 98%, Rama 88% vs 90%, minCA 3.85 vs 3.88 (both > 3.7). The 2x reconstruction
gap does not survive relaxation. **Codec question CLOSES at ANM; the eigh build's
conditional trigger did NOT fire.**

**GUARD2 status:** the spring energy surrogate does not track the metric, so
"physics prior not improvable from structure" is NOT established -- the question is
**open-but-deprioritised on headroom grounds** (0.36 A prize, mooted by usability),
not answered. The eigh-in-the-loop build remains the recorded conditional trigger:
run it only if a future stricter usability test shows ANM-relaxed fails while
PCA-relaxed passes with the transition sitting in 1.01-1.37 A.

_Verdict recorded 2026-08-04 after the usability test._

## STOPPING RULE for the codec question (pre-registered before PDB pretraining)

Learned attempts against ANM so far: (1) direct mode map -- below ANM; (2) spring-
constant map -- guard-tripped / no gain; (3) Perceiver/segment AE -- all arms >= ANM,
learned allocation hurts. PDB-scale B-factor pretraining -> mdCATH fine-tune would be
the FOURTH.

**Locked rule:** if pretraining at PDB scale (B-factor pretrain -> PCA-mode fine-tune)
still does NOT beat ANM at matched scalars on the frozen 7 (ANM 1.37 A @ 64 scalars),
the codec question is CLOSED -- no fifth architecture. ANM stands as the codec; effort
stays on the propagator and (deferred) the internal-coordinate non-protein codec.
Stated in advance so the outcome is not relitigated afterward.

Design (when built): PRETRAIN structure -> per-atom fluctuation (crystallographic
B-factors, ~1e5 PDB structures), NORMALISED PER STRUCTURE (B-factor scale tracks
resolution/refinement; without normalisation the model learns resolution, not
dynamics -- normalisation reported explicitly). FINE-TUNE structure -> PCA modes /
coefficient covariance on the mdCATH training pool. EVALUATE unchanged: frozen 7,
absolute A vs ANM at matched scalars. Caveat: B-factors mix crystal packing,
refinement, resolution, static disorder -- fine for PRETRAINING, not a final target
(hence pretrain-then-fine-tune, not train-on-B-factors).
