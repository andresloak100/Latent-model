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
cannot escape a cold start." Also, per ROADMAP 9.3, P-init
is retained on ARCHITECTURAL grounds regardless of this sweep: the SFC segment
assignment is time-stale (74-97% of atoms change slot, reference vs last frame, at
L=64), so segments are the better-training-now arm but not the right long-run one.

## P-init result: learned attention ACTIVELY HURTS (confound resolved)

Warm-start from trained S+SFC; epoch-0 == S+SFC by construction (guard passed
exactly). Then trained with the same schedule.

| L | scalars | epoch0 (=S+SFC) | held-out A | held-out curve | G1 cos | read |
|---|---|---|---|---|---|---|
| 8  | 64  | 2.44 | 2.57 | 2.49->2.55->2.52->2.54->2.57 | 0.95 | degrades |
| 16 | 128 | 2.13 | 2.24 | 2.16->2.19->2.20->2.19->2.24 | 0.89 | degrades |
| 32 | 256 | 1.78 | 1.88 | 1.80->1.84->1.84->1.86->1.88 | 0.76 | degrades |
| 64 | 512 | 1.26 | 1.31 | 1.26->1.26->1.27->1.28->1.31 | 0.52 | degrades |

**Locked read fires: P-init DEGRADES at every budget -> learned attention actively
hurts; the Perceiver's collapse is ARCHITECTURAL, not a cold-start optimisation
artifact.** Given the spatial prior for free (starting AT the segment solution, no
cold start), letting the learned attention move (G1 cos falls from ~1 toward 0.52,
so it does move) makes reconstruction monotonically worse. This resolves the main-
sweep confound (P losing because it collapsed): even without collapse, learned
allocation does not beat -- and slightly hurts -- the deterministic spatial
assignment on this task at this data scale. Magnitude is modest (a few %), and
P-init stays far better than the collapsed pure-P, but the direction is
unambiguous and consistent across all four budgets.

Open (next control): is the segment win about SPATIAL LOCALITY or just DETERMINISM?
S+random (segment pooling with a fixed RANDOM ordering) is the true null and
settles it -- see below.

## S+random control: locality is real (ordering hypothesis STANDS)

S+random = segment pooling with a fixed RANDOM per-atom ordering (Morton replaced by
a seeded permutation everywhere -- assignment AND the sinusoidal feature), same
schedule/guards. It is the true null (WL canonical order was never neutral; random
is). 4-arm sweep, held-out A:

| scalars | P (collapsed) | S+random | S+SFC | ANM |
|---|---|---|---|---|
| 64  | 2.59 | 2.66 | 2.46 | 1.37 |
| 128 | 2.65 | 2.59 | 2.16 | 1.05 |
| 256 | 2.62 | 2.43 | 1.78 | 0.57 |
| 512 | 3.01 | 1.93 | 1.25 | 0.13 |

**S+SFC >> S+random at every budget, gap widening with L (0.20 -> 0.68 A).** Locked
read fires: **spatial locality is real; the "spatial locality made E1-file work"
hypothesis STANDS** -- the §9 SFC material is validated, not rewritten.

Clean decomposition of the segment win into two separable ingredients:
- **Determinism** prevents collapse. S+random engages (zero-latent 3.2-3.8 >> held-
  out, G1 cos 0.86-0.97, scales 2.66 -> 1.93 with L) whereas the LEARNED Perceiver
  collapses (frame-invariant, inert). S+random beats collapsed P at high budget
  (1.93 < 3.01 at 512). So deterministic pooling, even with a meaningless ordering,
  is enough to keep the latent alive.
- **Locality** adds the rest. On top of determinism, SFC-contiguous segments buy a
  further 0.2-0.68 A over random segments -- the larger lever at high budget.

All arms remain >= ANM at matched scalars (converging negative unchanged): the
ordering/locality result is about which learned bottleneck is best, not about
beating the physics prior.
