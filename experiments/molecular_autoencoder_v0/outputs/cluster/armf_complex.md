# First MULTI-MOLECULE test: protein-ligand complex, both codecs, one latent (§7 additivity)

12 MISATO complexes. Protein coarse-grained to residue beads (sequential runs of atoms_residue,
82-414 beads) -> Cartesian ANM -> z_prot (Lp=8). Ligand -> graph codec -> z_lig (Ld=8).
Concatenated whitened latent (16-dim). §7's additivity assumption -- the 54 modes x 231 systems
latent-budget arithmetic that drove "tokens scale with molecules not atoms" -- has NEVER been
measured. This measures it. Same whitening, calibration, acceptance test, thresholds.

## GATE: §7 additivity -- weak, REAL, heterogeneous (not clean either way)
Cross-block |corr| (mean over the 8x8 protein-ligand block):

| | mean | spread |
|---|---|---|
| reference | **0.120** | +/-0.034 |
| null (frame-permuted) | 0.073 | +/-0.009 |

Above the noise floor by 0.047 (>2σ of null), just under the pre-registered Δ>0.05 bar.
HETEROGENEOUS: ~half the complexes show real coupling (1NPW 0.175 vs 0.069, 1GHW 0.160 vs 0.059,
1C70 0.158 vs 0.071), the other half none (10GS 0.083 vs 0.087, 1JGL 0.071 vs 0.082). So §7
additivity is a GOOD APPROXIMATION with a measurable, system-dependent coupled correction -- not
exactly valid, not grossly violated.

## Joint propagator vs TWO independent (held-out n=8)

| model | marginals | coupling(all) | non-Gauss | CROSS-BLOCK capture (gen vs ref 0.128) |
|---|---|---|---|---|
| JOINT (c=2.87 transfer) | 1/8 | 8/8 | 5/8 | **0.149 vs 0.128** (deficit -0.021) |
| JOINT (refit c=1.34) | 2/8 | 8/8 | 5/8 | **0.149 vs 0.128** (deficit -0.021) |
| INDEPENDENT (refit c) | **8/8** | 8/8 | 6/8 | 0.047 vs 0.128 (deficit +0.081) |

## Verdict -- neither branch wins cleanly; the decisive numbers are the cross-block line
- **Only the JOINT model captures the cross-block coupling** (gen 0.149 ~= ref 0.128); the
  INDEPENDENT model structurally cannot (gen 0.047 = noise floor, misses the real 0.128). A joint
  latent CAN represent multi-molecule coupling; independent propagators cannot, by construction.
- **The joint model's known under-capture weakness is NOT triggered** here (the flagged "adds
  ~0.03-0.18 to strong coupling"): the coupling is weak enough that the joint captures it fully,
  even slightly over (-0.021). That weakness is a strong-coupling phenomenon.
- **BUT the joint model loses on marginals** (2/8 vs independent 8/8), and a refit c does NOT fix
  it (deficit is non-uniform, one constant insufficient). Cause: the 16-dim joint DDPM is data-
  starved on MISATO (T=100 -> ~79 training pairs); two 8-dim models fit their marginals far
  better on the same data. This is a capacity/data limit, not a fundamental one.
- So the choice is objective-dependent: **independent** for marginal fidelity, **joint** if the
  weak cross-block coupling matters downstream. With longer trajectories the joint would likely
  recover its marginals and dominate; on MISATO's 100 frames it does not.

## c calibration: mechanism transfers, value is regime-specific (third data point)
joint-fit c = **1.34** (protein 2.87, ligand 1.03). A single global constant remains the shape of
the fix across all three regimes, but the VALUE refits every time -- it is a regime property
(correlation structure + latent dim), not a universal constant. Report per class.

## Scope
12 MISATO complexes, protein as residue-bead ANM, 100-frame decorrelated trajectories -> this is
EQUILIBRIUM (ensemble) coupling, not dynamic coupling. Ligand-sized ligands, fixed topology. The
additivity number is an ensemble-covariance measurement; dynamic protein-ligand coupling needs
dense-time complex trajectories (not MISATO).
