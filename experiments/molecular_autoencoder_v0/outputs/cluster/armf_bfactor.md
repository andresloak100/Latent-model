# B-factor pretraining track -- CLOSED by the oracle (before any training)

Two pre-training checks (the second closed the track for free, as the PCA oracle and
cross-replica ceiling did for the codec earlier).

## Check 1 -- GNM-vs-B correlation recalibrated (the 0.38 was n=1 noise)

Plain GNM predicted fluctuation vs crystallographic B, Pearson, n=250 PDB structures:

| cutoff | full (mean +- 95% CI) | termini-excluded |
|---|---|---|
| 7.3 A | 0.556 +- 0.021 | 0.527 |
| 10 A  | 0.552 +- 0.019 | 0.524 |
| 13 A  | 0.565 +- 0.019 | 0.545 |

Real correlation is ~0.55, squarely in the literature 0.5-0.6 range; cutoff barely
matters; termini-exclusion slightly LOWERS it. The earlier 0.38 was a single structure.
Baseline calibrated.

## Check 2 -- the oracle (fit springs to a fluctuation diagonal, reconstruct)

Per frozen-7 system: fit per-residue springs so diag(Gamma^+) matches a target diagonal
(differentiable, per-system, no learned map), diagonalise, reconstruct mdCATH held-out
frames in absolute A. This is the UPPER BOUND on what fluctuation supervision can buy.

| oracle | mean A | vs ANM 1.37 | note |
|---|---|---|---|
| ANM (uniform springs) | 1.37 | -- | baseline (reproduced exactly) |
| PCA (ceiling) | 0.78 | -- | linear ceiling |
| **oracle-RMSF** (trajectory's OWN diagonal) | **1.26** | -0.11 | best case: tiny headroom (~19% of ANM->PCA gap) |
| **oracle-B** (crystallographic B, perfect fit) | **1.45** | **+0.08** | WORSE than ANM |

## Verdict: CLOSE the track

- oracle-B 1.45 > ANM 1.37: fitting springs PERFECTLY to real crystallographic
  B-factors is worse than uniform ANM. Per the stopping rule, the codec question
  CLOSES -- a learned map (<= oracle) cannot beat ANM, so there is nothing to train.
- **Mechanism: crystal B is the WRONG diagonal.** B-vs-MD-RMSF correlation across the
  7 systems is 0.01-0.73 (near zero for several) -- crystallographic B measures crystal
  packing / low-T / refinement, not 320 K conformational dynamics. Fitting it points the
  springs the wrong way and hurts.
- Even the RIGHT diagonal (oracle-RMSF, the trajectory's own RMSF) buys only 0.11 A over
  ANM -- the structure->fluctuation->modes direction has little headroom even in the best
  case. The many-to-one concern is real, but the binding failure here is signal mismatch
  (crystal vs MD), not diagonal insufficiency alone.

## Consequence

This was the 4th and final learned attempt vs ANM (direct mode map, spring map,
Perceiver/segment AE, B-factor pretraining). **The codec question is CLOSED: ANM is the
codec** (1.37 A backbone, usable after relaxation, generalises by construction, protein-
biased). Effort stays on the PROPAGATOR; the deferred non-protein codec should decode
INTERNAL COORDINATES (torsional), not fit Cartesian fluctuation.
