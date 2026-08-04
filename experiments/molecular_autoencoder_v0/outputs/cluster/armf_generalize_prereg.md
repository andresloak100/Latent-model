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

_Pre-registered 2026-08-04, before building the shared map._
