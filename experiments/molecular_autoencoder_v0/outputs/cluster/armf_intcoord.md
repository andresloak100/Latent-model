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

## Pre-build checks (dihedral periodicity, rings) + design decisions

**Dihedral periodicity -- confirmed handled.** The oracle used sin/cos encoding (to_internal
returns [bonds, angles, sin(dih), cos(dih)]; reconstruction via atan2), and 0.49/0.77 are
CARTESIAN RMSD after NeRF round-trip (not internal RMSE). Necessity verified: 12/12 sampled
ligands have torsions that cross +-pi in their trajectory, so raw-dihedral PCA would have been
wrong. The result stands.

**Rings -- the common case, already handled; rule stated.** 10/12 ligands are cyclic (up to 7
rings). The oracle breaks rings via a BFS spanning tree from atom 0 (non-tree edges = ring-
closure bonds are dropped from the internal-coord set); round-trip 0.000 confirms the tree
fully determines Cartesian, and any imperfect ring closure on PCA-reconstruction shows up IN
the Cartesian RMSD (so 0.49 is honest about rings). **Ring-breaking rule for the codec (stated
up front):** deterministic BFS spanning tree from a canonical root, dropping non-tree bonds;
closure error measured as the reconstruction Cartesian RMSD. Refinement option if closure error
proves large: add the ring-closure bonds as redundant internal features so the codec can
enforce closure. Decide once, apply to every molecule.

**Reactive-corpus pilot -- tooling.** No QM code is installed (no xtb/psi4/pyscf/ase; the venv
lacks pip/ssl). Pilot needs the xtb static binary fetched to $SCRATCH (curl, no pip). Pilot is
PILOT-not-campaign per Andres: one reaction family, a handful of trajectories, modest theory;
deliver (a) validated pipeline, (b) event/format spec discovered by needing it, (c) cost per
trajectory. Resource note: the pilot fits in the margins (semi-empirical GFN2-xTB, CPU); the
FULL campaign is a serious sustained-DFT allocation that needs Andres's own resources or an
explicit arrangement on this borrowed account -- flag when the campaign decision comes.

## Torsional-ANM: physics does NOT solve the ligand codec -> learned codec is necessary

Torsional network model (torsional analogue of ANM): Cartesian ANM Hessian projected onto
z-matrix torsional DOF (K=J^T H J, G=J^T J), softest L modes, reconstruct via reference
bonds/angles + torsional displacement -> NeRF -> Cartesian. Round-trip 0.00 (geometry validated).
Same protocol/metric as the oracle. Mean over 12 MISATO ligands, held-out:

| L | Cartesian ANM | internal-coord PCA (ceiling) | TORSIONAL-ANM |
|---|---|---|---|
| 4 | 0.77 | 0.49 | 0.86 |
| 8 | 0.74 | 0.42 | 0.87 |
| 16 | 0.73 | 0.34 | 0.66 |

**Pre-registered read fires (third branch): torsional-ANM is ~= or WORSE than Cartesian ANM at
low L, far from the internal-PCA ceiling.** Inconsistent per system: matches the ceiling on some
(1PU7 0.16 vs 0.10, 2H4G 0.26 vs 0.18, 2XYS 0.18 vs 0.10) but collapses to ~null on others (10GS
1.43 vs 0.33, 5A0A 1.16 vs 0.32); on a few it gets WORSE with more modes (5VC6, 5A0A -- anti-
informative harmonic modes).

**So structure alone does NOT give torsional modes -- the OPPOSITE of proteins (where ANM won
four times over).** Mechanism: the ANM contact-graph Hessian does not predict WHICH torsions are
soft -- rotatable-bond barriers are chemical, not steric-contact, so the softest ANM-torsional
modes are not the actually-sampled torsions. **Verdict: a LEARNED general internal-coordinate
codec IS necessary for ligands.** The representation is right (oracle 0.49 ceiling), the physics
baseline does not reach it (0.86), and there is real headroom (0.86 -> 0.49) for a learned map.
This is the justified next build. (Local-geometry validity fold-in: torsional-ANM decodes have
reference bonds/angles by construction but can clash, minD ~0.5-1.3 A; the bond/angle-deviation
numbers had a reorder bug and will be redone cleanly with the learned codec.)
