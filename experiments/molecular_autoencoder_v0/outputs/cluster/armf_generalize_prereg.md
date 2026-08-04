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
- **learned <= ANM** -> learning contributes nothing; **use ANM directly as the
  general codec and say so plainly** -- a zero-parameter general codec at ~2/3 of
  ceiling is a legitimate win for objective 4, not a failure.

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
- ROLLOUT: Cartesian drift SATURATES (decoder-forced, uninformative); coefficients
  DRIFT OUT of the training range; backbone bond/angle geometry DEGRADES start->end.

_Pre-registered 2026-08-04, before building the shared map; learning curve and
rollout-latent metrics added before their results were read._
