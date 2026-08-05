# Internal-coordinate codec oracle (objective 1: general molecules) -- pre-registration

The whole pipeline works only on proteins. ANM captures 38-49% of the achievable signal on
ligands vs ~2/3 on proteins, because small-molecule dynamics is TORSIONAL and local -- what
elastic-network collective modes miss. Fixed-size conformation autoencoders in the literature
decode INTERNAL COORDINATES (bonds/angles/dihedrals), not Cartesians, for this reason
(arXiv:2101.01618). Before building a learned codec, measure the ceiling with an ORACLE --
the same move that changed the codec and B-factor tracks (and once closed one for free).

**Method.** MISATO ligands (already processed). Perceive bonds from the reference geometry;
build a z-matrix; convert the trajectory to internal coordinates (bonds, angles, sin/cos of
dihedrals for periodicity); PCA the internal-coordinate trajectory (fit first half); reconstruct
held-out frames; NeRF back to Cartesian; report absolute A (Kabsch-aligned) at L=4/8/16, vs
null, Cartesian ANM, and Cartesian PCA on the SAME ligands and frames. A round-trip sanity
check (to_internal -> from_internal ~= identity) gates correctness first.

**Pre-registered read:**
- internal-coord PCA >> Cartesian ANM (and approaching Cartesian PCA) -> the REPRESENTATION is
  the fix; build the internal-coordinate codec.
- internal-coord PCA ~= Cartesian PCA (no better) -> representation is NOT the lever; the ligand
  gap is something else, and we need a different diagnosis before building.

Cheap, pure numpy, data in hand; decides whether the build is worth starting.

## RESULT: internal-coordinate representation is the fix (build the codec)

Round-trip sanity: 0.000 A on all 12 ligands (z-matrix + NeRF geometry validated). Mean over
12 MISATO ligands, absolute A, held-out frames:

| L | Cartesian ANM | Cartesian PCA | INTERNAL-coord PCA |
|---|---|---|---|
| 4 | 0.77 | 1.21 | **0.49** |
| 8 | 0.74 | 0.79 | **0.42** |
| 16 | 0.75 | 0.47 | **0.34** |

**Internal-coord PCA beats Cartesian ANM by ~40% at low L (0.49 vs 0.77 at L4, 0.42 vs 0.74 at
L8) and also beats Cartesian PCA.** The pre-registered read fires: the REPRESENTATION is the
lever. A 4-8-dim internal-coordinate codec captures the ligand torsional manifold that
Cartesian modes (ANM or PCA) need far more dimensions for -- exactly the arXiv:2101.01618
thesis. Note Cartesian PCA is WORSE than ANM at L=4 (1.21 vs 0.77): few Cartesian modes cannot
represent a torsion (a nonlinear collective Cartesian motion), while internal coords do it
directly.

**VERDICT: BUILD THE INTERNAL-COORDINATE CODEC for objective 1 (general molecules).** The
oracle changed the picture as oracles have every time this project ran one. Next: a learned
internal-coordinate codec (or, given PCA already does this well, an internal-coord PCA codec
+ the propagator on the torsional latent) -- and the same acceptance discipline (held-out
ligands, absolute A, vs the oracle ceiling here).
