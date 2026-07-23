# Metrics

## Learned autoencoder (all-atom, per-structure mean)

| metric | value |
|---|---|
| all_atom_rmsd | 11.46 |
| backbone_rmsd | 10.93 |
| pairwise_distance_error | 5.631 |
| bond_length_error | 1.576 |
| chirality_violation_rate | 0.4369 |
| clashes_per_1000_atoms | 380.8 |
| contact_f1 | 0.1544 |
| compression_ratio | 2.434 |

## All-atom trivial baselines (full variable-size setting)

| method | all_atom_rmsd (A) |
|---|---|
| identity (predict truth) | 0 |
| centroid (predict Rg) | 10.22 |
| **learned autoencoder** | 11.46 |

## Per-structure (learned model)

| pdb_id | n_residues | n_atoms | all_atom_rmsd | backbone_rmsd | bond_length_error | clashes_per_1000_atoms | chirality_violation_rate | contact_f1 | compression_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1CSP | 67 | 505 | 11.91 | 11.72 | 1.595 | 404 | 0.4655 | 0.1235 | 2.959 |
| 1IGD | 61 | 468 | 11.86 | 11.4 | 1.485 | 431.6 | 0.5088 | 0.1517 | 2.742 |
| 1L2Y | 20 | 154 | 9.472 | 8.462 | 1.747 | 298.7 | 0.4118 | 0.2264 | 0.9023 |
| 1MJC | 69 | 514 | 12.02 | 11.63 | 1.482 | 402.7 | 0.3559 | 0.1 | 3.012 |
| 1PGB | 56 | 436 | 12.04 | 11.42 | 1.569 | 367 | 0.4423 | 0.1705 | 2.555 |

## Baseline comparison (fixed backbone cohort, n_res=20, 80 atoms)

Backbone RMSD (A) vs latent-float budget; lower RMSD + higher compression is better.

| method | latent_floats | compression_ratio | mean_backbone_rmsd |
|---|---|---|---|
| **learned_autoencoder** | 512 | 0.4688 | 9.362 |
| identity | 240 | 1 | 0 |
| mean_shape | 0 | inf | 6.407 |
| pca_k2 | 2 | 120 | 4.603 |
| pca_k4 | 4 | 60 | 3.787 |
| pca_k8 | 8 | 30 | 3.244 |

## Timing & memory

- mean inference: 0.008659 s/structure
- peak host RSS: 434.7 MB
- GPU memory: N/A (CPU-only run)
