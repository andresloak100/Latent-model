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

## Two ceilings: torsions carry the signal AND softness is the right knob -> build the learned map

Same ligands/frames/L/metric. Mean over 12 held-out:

| L | Cartesian ANM | internal PCA | torsion-only PCA | softness oracle |
|---|---|---|---|---|
| 4 | 0.77 | 0.49 | 0.53 | 0.72 |
| 8 | 0.74 | 0.42 | 0.46 | 0.51 |
| 16 | 0.73 | 0.34 | 0.39 | 0.30 |

Variance split (Cartesian amplitude of each DOF alone): dihedral ~74% / angle ~20% / bond ~7%.

- **Ceiling 1 (torsion-only PCA) ~= internal PCA** (0.53 vs 0.49 at L4; gap ~0.04A). Torsions carry
  the Cartesian signal; bond/angle variance (27% internal) contributes ~0.04A (stiff, small-
  amplitude). A torsion-only method targets ~0.53 -- nearly the full ceiling, not an unreachable one.
- **Ceiling 2 (softness oracle) matches/beats torsion-only PCA at usable L**: L8 0.51 vs 0.46, L16
  0.30 vs 0.39; only short at L4 (0.72 vs 0.53, diagonal param can't pick the best 4). Softness is
  the right knob at L>=8.

**VERDICT: BUILD THE LEARNED PER-TORSION-SOFTNESS MAP.** Torsions dominate, the softness
parameterisation reaches the torsional ceiling at L>=8, and the physics baseline (torsional-ANM
0.86) leaves large headroom to the oracle (~0.46-0.51). Learned map: per-torsion chemistry
features -> softness -> torsional modes (diag(k), G) -> reconstruct; trained across ligands,
eval held-out vs these ceilings.

## Learned softness map, v1 (local features, 8 training ligands): generalises at L16, fails at L4/L8

Structure-only per-torsion features (element x4 dihedral atoms, central-bond length, degrees, ring
membership) -> shared MLP -> stiffness -> softest-L modes of (diag(k), G). Trained end-to-end
(through eigh) on 8 ligands' torsional displacement, eval on 4 HELD-OUT ligands (no per-system fit).

| L | learned train | learned HELD-OUT | softness oracle | torsion PCA | Cartesian ANM |
|---|---|---|---|---|---|
| 4 | 0.64 | 1.32 | 0.72 | 0.53 | 0.77 |
| 8 | 0.44 | 1.17 | 0.51 | 0.46 | 0.74 |
| 16 | 0.30 | 0.41 | 0.30 | 0.39 | 0.73 |

- **Fits**: train tracks the oracle (0.44/0.30 at L8/16) -> softness->modes parameterisation is
  expressive and trainable.
- **Generalises at L16 only**: held-out 0.41 beats Cartesian ANM 0.73, near the ceilings.
- **Fails at L4/L8**: held-out 1.17 > cANM 0.74. The softness RANKING (which few torsions are
  softest) does not transfer from local features. With top-4/8 modes you must nail the ranking;
  with top-16 the subspace is wide enough that exact ranking stops mattering.

Diagnosis: not a dead concept (oracle says softness is right; map fits training) -- the MAP under-
generalises. Two joint causes, both must be fixed for a real codec: (1) n=8 training molecules is
tiny; (2) local features lack graph context. NEXT: scale to many MISATO ligands + a graph encoder
(message passing) so each torsion's softness is predicted from its molecular neighbourhood, target
low-L generalisation. This is the "full graph codec" branch of the pre-registered fork.

## Full graph codec (GNN softness map, 142 train / 48 held-out ligands): beats ANM + GENERALISES

Message-passing GNN over the molecular bond graph -> per-atom embeddings -> per-torsion softness
(read from the 4 dihedral atoms + central-bond ring/length) -> softest-L modes of (diag(k), G) ->
reconstruct dihedrals -> NeRF. Trained end-to-end (through eigh) on all frames of 142 ligands,
eval on 48 HELD-OUT ligands (no per-system fit). Stabilised training (cosine LR, L={4,8}) barely
moved held-out from the first run -> ~0.64 at L8 is real capacity, not an optimisation artifact.

**GUARDS (L=8, held-out, NO relaxation) -- reported before RMSD:**
- bond/angle deviation: **0 by construction** (reference bonds+angles reused; only dihedrals moved).
- min non-bonded heavy-heavy distance: recon **1.96 A** vs reference 2.05 A (mean). Reconstruction
  packs marginally tighter; no contact below the reference's own floor -- **no catastrophic clashes**.
  (Sub-2.0A fraction 54% recon vs 33% reference; 2.0A is a soft threshold that already flags a
  third of real frames, so this is mild tightening, not clashing.)

**RMSD (absolute A, held-out frames):**

| L | graph-codec train | graph-codec HELD-OUT | v1 held-out | Cartesian ANM | torsion PCA ceiling | softness oracle |
|---|---|---|---|---|---|---|
| 4 | 0.76 | 0.80 | 1.32 | 0.77 | 0.53 | 0.72 |
| 8 | 0.60 | 0.64 | 1.17 | 0.74 | 0.46 | 0.51 |
| 16 | 0.43 | 0.44 | 0.41 | 0.73 | 0.39 | 0.30 |

**MILESTONE: first learned codec in this project to beat ANM AND generalise.** Train ~= held-out
across 48 held-out molecules (0.76/0.80, 0.60/0.64, 0.43/0.44) -> the graph context + ~140
molecules closed v1's generalisation failure (v1 L8: train 0.44 / held-out 1.17). Beats Cartesian
ANM at L8 (0.64 < 0.74) and L16 (0.44 < 0.73, ~ceiling). **This is the OPPOSITE of the protein
result** (where ANM beat 4 learned attempts): ligand softness is not predicted by ANM's contact
graph but IS learnable by a GNN from molecular structure.

**BOUNDARY (honest):**
- At L4 (extreme compression) it only TIES ANM (0.80 vs 0.77) -- with 4 modes you must nail the
  softness ranking exactly and structure-only prediction can't.
- It does NOT reach the per-system ceilings at L4/L8 (0.64 vs torPCA 0.46 / oracle 0.51 at L8).
  Those ceilings are fit to each test molecule's OWN trajectory; the codec predicts softness from
  structure with ZERO test-time params. The residual gap = the system-specific part of softness
  (conformational/environmental) that structure alone can't see. At L16 the gap nearly closes
  (0.44 vs 0.39).

**Where the codec stands:** a general, zero-test-time-param ligand codec that beats ANM at
practical L and nearly reaches the torsional ceiling at L16. Open levers if more is wanted:
scale ligands (used 190 of ~17k MISATO) + bigger GNN toward the L8 ceiling; or few-shot test-time
softness adaptation to close the per-system gap (reintroduces a small per-system fit).
