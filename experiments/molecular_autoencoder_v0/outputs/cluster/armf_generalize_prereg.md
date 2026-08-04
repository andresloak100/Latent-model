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
