# Metrics

## Learned autoencoder (all-atom, per-structure mean)

| metric | value |
|---|---|
| all_atom_rmsd | 10.78 |
| backbone_rmsd | 10.19 |
| pairwise_distance_error | 5.941 |
| bond_length_error | 1.335 |
| chirality_violation_rate | 0.427 |
| clashes_per_1000_atoms | 450.6 |
| contact_f1 | 0.1667 |
| compression_ratio | 4.868 |

## All-atom trivial baselines (full variable-size setting)

| method | all_atom_rmsd (A) |
|---|---|
| identity (predict truth) | 0 |
| centroid (predict Rg) | 10.22 |
| **learned autoencoder** | 10.78 |

## Per-structure (learned model)

| pdb_id | n_residues | n_atoms | all_atom_rmsd | backbone_rmsd | bond_length_error | clashes_per_1000_atoms | chirality_violation_rate | contact_f1 | compression_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1CSP | 67 | 505 | 11.26 | 10.97 | 1.388 | 487.1 | 0.3966 | 0.1424 | 5.918 |
| 1IGD | 61 | 468 | 10.91 | 10.67 | 1.32 | 427.4 | 0.3158 | 0.1664 | 5.484 |
| 1L2Y | 20 | 154 | 9.614 | 8.525 | 1.484 | 214.3 | 0.4118 | 0.2308 | 1.805 |
| 1MJC | 69 | 514 | 11.22 | 10.75 | 1.155 | 665.4 | 0.4915 | 0.1254 | 6.023 |
| 1PGB | 56 | 436 | 10.88 | 10.04 | 1.33 | 458.7 | 0.5192 | 0.1686 | 5.109 |

## Baseline comparison (fixed backbone cohort, n_res=20, 80 atoms)

Backbone RMSD (A) vs latent-float budget; lower RMSD + higher compression is better.

| method | latent_floats | compression_ratio | mean_backbone_rmsd |
|---|---|---|---|
| **learned_autoencoder** | 256 | 0.9375 | 8.175 |
| identity | 240 | 1 | 0 |
| mean_shape | 0 | inf | 6.407 |
| pca_k2 | 2 | 120 | 4.603 |
| pca_k4 | 4 | 60 | 3.787 |
| pca_k8 | 8 | 30 | 3.244 |

## Timing & memory

- mean inference: 0.006593 s/structure
- peak host RSS: 434.4 MB
- GPU memory: N/A (CPU-only run)
