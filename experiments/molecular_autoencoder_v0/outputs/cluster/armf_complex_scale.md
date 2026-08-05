# Corpus-scale complex propagator -- the first corpus-scale training run in the project

453 MISATO complexes (377 train / 76 held-out), joint 16-dim latent (ANM protein residue-bead
block + graph-codec ligand block). The joint propagator is trained ACROSS complexes (pooled pairs),
not one complex's ~79 pairs. Learning curve 8/50/200/377 train complexes. Independent = two pooled
per-block DDPMs at full scale. Batched one-step generation (latents decorrelated, TAU=1). Each
model gets its OWN global calibration constant c.

**Whitening correction (load-bearing):** structure-predicted sqrt(kT/lambda) whitening has an
arbitrary PER-COMPLEX scale (esp. the ligand codec's learned lambda), which breaks cross-complex
POOLING (generation diverges, marginals 0/76). Corpus-scale pooling requires EMPIRICAL per-complex
whitening (unit-variance inputs). The big historical c values (protein 2.87) were partly artifacts
of structure-lambda whitening + the TAU=50 multi-step regime.

## Learning curve (held-out n=76)

| train | JOINT marg | JOINT coup | JOINT x-blk | INDEP marg | INDEP coup | INDEP x-blk |
|---|---|---|---|---|---|---|
| 8 | 75/76 | 73/76 | **0.142** | 76/76 | 42/76 | 0.089 |
| 50 | 75/76 | 44/76 | 0.097 | 76/76 | 40/76 | 0.089 |
| 200 | 75/76 | 44/76 | 0.093 | 76/76 | 42/76 | 0.092 |
| 377 | 76/76 | 41/76 | 0.094 | 76/76 | 42/76 | 0.092 |

(reference cross-block |corr| = 0.136 throughout)

## PRIMARY QUESTION (pre-registered): does the joint recover marginals at scale? -- YES
JOINT marginals 75-76/76 at EVERY scale. The per-system tie (joint 2/8 in armf_complex.md) was a
**corpus-USE artifact, not a data-length limit** -- demonstrated, as suspected. Corpus-scale
training fixes the joint marginals.

## But the joint does NOT win outright -- its coupling edge is small-corpus OVERFITTING
At 8 complexes the joint captures cross-block coupling (x-blk 0.142 ~= ref 0.136; coupling 73/76)
and beats independent (0.089; 42/76). But that edge **evaporates with scale**: by 200-377 the joint
converges to the independent baseline (joint coup 41/76 ~= indep 42/76; joint x-blk 0.094 ~= indep
0.092), and BOTH under-capture the reference (0.136). The scale-8 advantage was overfitting to 8
complexes' coupling, not a real capability.

## Operational verdict: at corpus scale, JOINT ~= INDEPENDENT -> §7 additivity holds
A joint model does NOT earn its complexity at scale: a per-block independent model matches it on
marginals AND coupling. This **validates §7 additivity operationally** -- modeling a complex as
independent latent blocks is as good as a joint model. The weak-but-real cross-block coupling
(0.136, heterogeneous, ~half the complexes, per armf_complex.md) is captured by NEITHER shared
propagator at scale (~0.09): a z_t-only shared model produces the population-AVERAGE coupling, and
the coupling is per-complex heterogeneous. Capturing it would require conditioning on complex /
structure identity -- a documented next lever, not a refutation of additivity.

## Byproduct 1: is c predictable from structure? -- No, but it becomes UNIVERSAL at scale
Per-complex c: mean **1.07 +/- 0.05**, range [0.98,1.24]. corr(c, {mean-lambda, protein size,
ligand size, ratio}) all |<=0.06| -- NOT a function of these features. But it is ~CONSTANT at
scale (down from 1.59 at 8 complexes), so it does not NEED prediction: with empirical whitening at
corpus scale the calibration collapses to a single near-unity constant (~1.07). The last per-system
fitted quantity effectively disappears -- via convergence to universality at scale, not via a
structural regressor.

## Byproduct 2: first corpus-scale training run
Every prior learned component trained on 5-20 systems and OVERFITTING was the recurring diagnosis.
This is the first time the corpus is used at scale (377 complexes, ~3e4 pooled pairs). No
overfitting: marginals stable 75-76/76 across all scales, c converges. Codec question stays closed
(B-factor oracle ceiling argument, independent of data scale; stopping rule stands).

## Scope
453 MISATO complexes, protein as residue-bead ANM, ligand-sized ligands, fixed topology,
decorrelated 100-frame trajectories (equilibrium ensembles). This is corpus-scale EQUILIBRIUM
generation; dynamic protein-ligand coupling needs dense-time complex trajectories (not MISATO).
