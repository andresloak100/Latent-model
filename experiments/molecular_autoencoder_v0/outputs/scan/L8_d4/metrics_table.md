# Metrics

## Learned autoencoder (all-atom, per-structure mean)

| metric | value |
|---|---|
| all_atom_rmsd | 11.64 |
| backbone_rmsd | 11.17 |
| pairwise_distance_error | 5.845 |
| bond_length_error | 1.766 |
| chirality_violation_rate | 0.4398 |
| clashes_per_1000_atoms | 333.1 |
| contact_f1 | 0.1357 |
| compression_ratio | 38.94 |

## All-atom trivial baselines (full variable-size setting)

| method | all_atom_rmsd (A) |
|---|---|
| identity (predict truth) | 0 |
| centroid (predict Rg) | 10.22 |
| **learned autoencoder** | 11.64 |

## Per-structure (learned model)

| pdb_id | n_residues | n_atoms | all_atom_rmsd | backbone_rmsd | bond_length_error | clashes_per_1000_atoms | chirality_violation_rate | contact_f1 | compression_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1CSP | 67 | 505 | 11.45 | 11.15 | 1.634 | 372.3 | 0.4138 | 0.1314 | 47.34 |
| 1IGD | 61 | 468 | 13.05 | 12.68 | 2.006 | 282.1 | 0.4035 | 0.1129 | 43.88 |
| 1L2Y | 20 | 154 | 9.337 | 8.187 | 1.706 | 240.3 | 0.5882 | 0.1455 | 14.44 |
| 1MJC | 69 | 514 | 12.39 | 12.29 | 1.867 | 319.1 | 0.3898 | 0.1288 | 48.19 |
| 1PGB | 56 | 436 | 11.96 | 11.53 | 1.615 | 451.8 | 0.4038 | 0.1598 | 40.88 |

## Baseline comparison (fixed backbone cohort, n_res=20, 80 atoms)

Backbone RMSD (A) vs latent-float budget; lower RMSD + higher compression is better.

| method | latent_floats | compression_ratio | mean_backbone_rmsd |
|---|---|---|---|
| **learned_autoencoder** | 32 | 7.5 | 9.299 |
| identity | 240 | 1 | 0 |
| mean_shape | 0 | inf | 4.966 |
| pca_k2 | 2 | 120 | 1.727 |
| pca_k4 | 4 | 60 | 4.712e-15 |

## Timing & memory

- mean inference: 0.01097 s/structure
- peak host RSS: 441.4 MB
- GPU memory: N/A (CPU-only run)
