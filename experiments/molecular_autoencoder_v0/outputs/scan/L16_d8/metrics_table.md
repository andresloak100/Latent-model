# Metrics

## Learned autoencoder (all-atom, per-structure mean)

| metric | value |
|---|---|
| all_atom_rmsd | 11.15 |
| backbone_rmsd | 10.71 |
| pairwise_distance_error | 5.558 |
| bond_length_error | 1.5 |
| chirality_violation_rate | 0.4124 |
| clashes_per_1000_atoms | 349 |
| contact_f1 | 0.1765 |
| compression_ratio | 9.736 |

## All-atom trivial baselines (full variable-size setting)

| method | all_atom_rmsd (A) |
|---|---|
| identity (predict truth) | 0 |
| centroid (predict Rg) | 10.22 |
| **learned autoencoder** | 11.15 |

## Per-structure (learned model)

| pdb_id | n_residues | n_atoms | all_atom_rmsd | backbone_rmsd | bond_length_error | clashes_per_1000_atoms | chirality_violation_rate | contact_f1 | compression_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1CSP | 67 | 505 | 11.26 | 10.84 | 1.308 | 402 | 0.4483 | 0.1187 | 11.84 |
| 1IGD | 61 | 468 | 11.67 | 11.17 | 1.601 | 482.9 | 0.3333 | 0.1846 | 10.97 |
| 1L2Y | 20 | 154 | 9.043 | 8.379 | 1.642 | 123.4 | 0.5882 | 0.2692 | 3.609 |
| 1MJC | 69 | 514 | 11.58 | 11.56 | 1.512 | 367.7 | 0.2881 | 0.1294 | 12.05 |
| 1PGB | 56 | 436 | 12.21 | 11.58 | 1.436 | 369.3 | 0.4038 | 0.1804 | 10.22 |

## Baseline comparison (fixed backbone cohort, n_res=20, 80 atoms)

Backbone RMSD (A) vs latent-float budget; lower RMSD + higher compression is better.

| method | latent_floats | compression_ratio | mean_backbone_rmsd |
|---|---|---|---|
| **learned_autoencoder** | 128 | 1.875 | 8.874 |
| identity | 240 | 1 | 0 |
| mean_shape | 0 | inf | 4.966 |
| pca_k2 | 2 | 120 | 1.727 |
| pca_k4 | 4 | 60 | 4.712e-15 |

## Timing & memory

- mean inference: 0.1154 s/structure
- peak host RSS: 440.6 MB
- GPU memory: N/A (CPU-only run)
