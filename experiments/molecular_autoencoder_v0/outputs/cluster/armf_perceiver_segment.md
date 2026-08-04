# Perceiver vs spatial-segment bottleneck on the deviation task (matched SCALARS)

The intended architecture (atom-native Perceiver, generic learned slots) tested on
the deviation task for the first time, at EQUAL SCALARS = L*d (not equal tokens),
against ANM on the frozen 7. Three arms: P (learned slot cross-attention, key/value
split), P+SFC (P + Morton SFC rank in the key), S+SFC (deterministic segment
assignment by SFC rank, E1-file generalised). d=8 fixed; budgets 64/128/256/512
scalars. All references recomputed in-script on the frozen 7 (G3).

## Guards (read first)

- **G3 references** (frozen 7, held-out rep0 2nd half): null 2.63, ANM at matched
  scalars 1.37 / 1.05 / 0.57 / 0.13 @ 64 / 128 / 256 / 512, cross(L64) 1.01, within
  (L64) 0.78. Reproduces the anchors (ANM 1.37 @ 64 scalars). ANM at 256/512 is
  **capped at 186 modes (3N-6)** on small systems -> it approaches exact
  reconstruction at high budget, so the high-budget ANM bar is near-trivial.
- **G2** scalars = L*d printed on every row.
- **G1 frame-variance** and **G4 zero-latent** per cell, in the table.

## Results (absolute A, GPU, epochs 400)

| arm | L | d | scalars | held-out A | vs ANM | zero-latent | G1 cos | slotStd | VOID |
|---|---|---|---|---|---|---|---|---|---|
| P     | 8  | 8 | 64  | 2.54 | +1.17 | 2.67 | 0.985 | 0.19 | ~inert |
| P+SFC | 8  | 8 | 64  | 2.68 | +1.32 | 2.67 | 0.933 | 0.20 | VOID |
| S+SFC | 8  | 8 | 64  | 2.46 | +1.10 | 3.75 | 0.951 | 0.20 | |
| P     | 16 | 8 | 128 | 2.72 | +1.67 | 2.69 | 0.963 | 0.16 | ~inert |
| P+SFC | 16 | 8 | 128 | 2.63 | +1.58 | 2.63 | 0.999 | 0.05 | VOID |
| S+SFC | 16 | 8 | 128 | 2.14 | +1.09 | 3.21 | 0.922 | 0.25 | |
| P     | 32 | 8 | 256 | 2.66 | +2.09 | 2.63 | 0.983 | 0.15 | ~inert |
| P+SFC | 32 | 8 | 256 | 2.63 | +2.07 | 2.64 | 0.996 | 0.05 | VOID |
| S+SFC | 32 | 8 | 256 | 1.80 | +1.24 | 2.83 | 0.704 | 0.45 | |
| P     | 64 | 8 | 512 | 2.72 | +2.60 | 2.89 | 0.992 | 0.11 | ~inert |
| P+SFC | 64 | 8 | 512 | 2.63 | +2.50 | 2.63 | 0.998 | 0.04 | VOID |
| S+SFC | 64 | 8 | 512 | 1.25 | +1.13 | 2.68 | 0.397 | 0.66 | |

## Pre-registered reads that fire

1. **All arms >= ANM at every matched-scalar budget.** S+SFC (best learned) is
   +1.1 to +1.2 A worse than ANM everywhere. **Learned codecs do not beat the
   physics prior at this data scale -- a converging negative** (consistent with the
   learning curve), not a new one.
2. **S+SFC < P at every budget** (P+SFC VOID). Deterministic SPATIAL assignment
   beats learned allocation; the ordering hypothesis holds and E1-file was not an
   artifact. S+SFC is the only arm that genuinely engages: zero-latent >> held-out,
   G1 cosine falls to 0.40 at L=64, and it scales with budget (2.46 -> 1.25).
3. **The Perceiver collapses.** P and P+SFC are frame-invariant (G1 cos 0.93-0.998)
   and near-inert (held-out ~= zero-latent ~= null 2.63) at every budget --
   posterior collapse: the decoder reaches null by predicting the mean and the
   learned latent never becomes informative. G1 caught it (same failure the modal
   arm-F sweep first hit).

## Caveat -> P-init

Read 2 is confounded: P loses partly BECAUSE it collapsed, not necessarily because
learned allocation is worse when it engages. The P-init arm (warm-start from trained
S+SFC + a learnable attention that can move off the spatial prior; epoch-0 == S+SFC
by construction) separates "learned allocation < spatial" from "learned attention
cannot escape a cold start." Running as job 10283447. Also, per ROADMAP 9.3, P-init
is retained on ARCHITECTURAL grounds regardless of this sweep: the SFC segment
assignment is time-stale (74-97% of atoms change slot, reference vs last frame, at
L=64), so segments are the better-training-now arm but not the right long-run one.
