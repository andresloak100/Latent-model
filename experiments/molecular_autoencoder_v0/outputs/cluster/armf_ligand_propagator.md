# Propagator STACK transfer: proteins -> ligands (objective-1, full stack end-to-end)

First time the full stack runs end-to-end on a NON-PROTEIN molecule: banked graph codec ->
per-ligand torsional-mode latents -> the EXISTING propagator machinery UNCHANGED (FiLMDenoiser
DDPM, sqrt(kT/lambda) whitening, one global variance constant, the pre-committed acceptance test +
thresholds). 187 MISATO ligands (ntors>=8); fit c on 5, evaluate 15 held-out.

## Data reality first (measured, armf_ligand_prop_precheck.py)
MISATO ligand latents are DECORRELATED frame-to-frame: lag-1 autocorr median 0.08, IAT median 1.2,
only 22% of modes lag-1>0.3. The 100 MISATO snapshots are near-independent equilibrium samples,
not a dense-in-time trajectory. Consequence: the DYNAMICS discriminators (kinetics IAT,
transitions) are near-trivial here (ref IAT~1); TAU=50 (long-protein lag) is inapplicable, use
TAU=1. The MEANINGFUL transfer test is the EQUILIBRIUM discriminators (marginals, coupling,
non-Gaussianity). This is a property of the DATA, not the propagator.

## c=2.87 transfer: MECHANISM transfers, VALUE does not
- ligand-fit c = **1.03** (proteins 2.87); per-mode deficit CV **0.17** (low).
- The CALIBRATION MECHANISM transfers: a single global constant still works on ligands (low CV,
  uniformly off not per-mode) -- same structural finding as proteins.
- The VALUE does not: proteins 2.87, ligands 1.03. c~=1 means the ligand propagator has almost NO
  variance deficit. Explanation: the protein deficit came from the TAU=50 multi-step conditional
  rollout accumulating variance collapse; ligands (decorrelated, TAU=1) learn the marginal in one
  step, so no collapse. **The variance deficit is a regime property (correlated/multi-step), not a
  universal constant.** Refit per molecule class -- cheap (5 systems), and the low CV says one
  constant suffices.

## Acceptance test (held-out, n=15)
Applying protein c=2.87 -> marginals **0/15** (over-inflates the already-correct ligand variance).
Applying refit ligand c=1.03:

| discriminator | pass | beats OU |
|---|---|---|
| marginals (varRatio) | **12/15** | -- |
| cross-mode coupling | **15/15** | **15/15** |
| non-Gaussianity (kurtosis) | 7/15 | 9/15 |
| kinetics IAT [dyn*] | 13/15 | -- |
| transitions [dyn*] | 15/15 | -- |

[dyn*] near-trivial: decorrelated latents, so these test decorrelated-sample matching, not learned
dynamics.

## Verdict
**The equilibrium-generation mechanism TRANSFERS from proteins to ligands.** Cross-mode coupling is
perfect (15/15) and beats OU on every ligand (15/15) -- the DDPM reproduces the codec latents'
coupling structure that OU cannot, on a new molecule class. Marginals pass with a refit constant
(12/15); non-Gaussianity is the weak discriminator (7/15) -- the SAME weakness as proteins (7/10),
so it is a propagator property, not ligand-specific. The split the plan asked for: nothing fails
due to the codec's latent statistics; the only non-transfer is the c VALUE, which is a regime
effect (decorrelation) attributable to the DATA, and the mechanism (one global constant) holds.
**Boundary:** ligand DYNAMICS cannot be tested on MISATO (decorrelated); a true ligand-dynamics
propagator test needs dense-in-time ligand trajectories, which MISATO is not.

## Next increment (planned, not started): protein-ligand COMPLEX together
MISATO trajectories are complexes; we now have both codecs (ANM protein + graph-codec ligand).
Concatenated latent, one propagator, one system = the first MULTI-MOLECULE test the project has
run. First step must be the §7 additivity validation (measure before assuming): does a
concatenated protein+ligand latent's cross-block coupling exist and does the joint propagator
capture it, vs two independent propagators. Known boundary (ROADMAP 9.4): absolute-coordinate
content is not equivariant across a multi-molecule global alignment -- relative geometry needed.
