# Metrics

## Learned autoencoder (all-atom, per-structure mean)

| metric | value |
|---|---|
| all_atom_rmsd | 0.4282 |
| backbone_rmsd | 0.3943 |
| pairwise_distance_error | 0.2785 |
| bond_length_error | 0.1221 |
| chirality_violation_rate | 0 |
| clashes_per_1000_atoms | 5.135 |
| contact_f1 | 0.9597 |
| compression_ratio | 10.94 |

## All-atom trivial baselines (full variable-size setting)

| method | all_atom_rmsd (A) |
|---|---|
| identity (predict truth) | 0 |
| centroid (predict Rg) | 10.79 |
| **learned autoencoder** | 0.4282 |

## Per-structure (learned model)

| pdb_id | n_residues | n_atoms | all_atom_rmsd | backbone_rmsd | bond_length_error | clashes_per_1000_atoms | chirality_violation_rate | contact_f1 | compression_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1BDD | 60 | 478 | 0.4567 | 0.4319 | 0.1447 | 0 | 0 | 0.9313 | 11.2 |
| 1BPI | 58 | 454 | 0.563 | 0.4656 | 0.1454 | 33.04 | 0 | 0.9346 | 10.64 |
| 1CRN | 46 | 327 | 0.392 | 0.3968 | 0.1188 | 0 | 0 | 0.9607 | 7.664 |
| 1CTF | 68 | 487 | 0.4757 | 0.4244 | 0.13 | 2.053 | 0 | 0.9815 | 11.41 |
| 1E0L | 37 | 309 | 0.3758 | 0.3507 | 0.09799 | 0 | 0 | 0.9323 | 7.242 |
| 1ENH | 54 | 466 | 0.3773 | 0.3435 | 0.123 | 0 | 0 | 0.99 | 10.92 |
| 1FKB | 107 | 832 | 0.4675 | 0.4617 | 0.1538 | 1.202 | 0 | 0.9571 | 19.5 |
| 1FSD | 28 | 247 | 0.3281 | 0.3268 | 0.09235 | 0 | 0 | 0.9818 | 5.789 |
| 1R69 | 63 | 484 | 0.4166 | 0.4047 | 0.1322 | 0 | 0 | 0.9423 | 11.34 |
| 1SHG | 57 | 472 | 0.3753 | 0.3677 | 0.1027 | 0 | 0 | 0.961 | 11.06 |
| 1TEN | 90 | 698 | 0.3627 | 0.3484 | 0.1089 | 0 | 0 | 0.9544 | 16.36 |
| 1UBI | 76 | 602 | 0.5103 | 0.426 | 0.1276 | 21.59 | 0 | 0.9607 | 14.11 |
| 1UBQ | 76 | 602 | 0.5002 | 0.4277 | 0.1261 | 21.59 | 0 | 0.9605 | 14.11 |
| 1VII | 36 | 295 | 0.3868 | 0.3422 | 0.1075 | 0 | 0 | 0.9664 | 6.914 |
| 2CI2 | 65 | 521 | 0.5054 | 0.4872 | 0.1348 | 1.919 | 0 | 0.963 | 12.21 |
| 5PTI | 58 | 454 | 0.5747 | 0.4657 | 0.1401 | 26.43 | 0 | 0.9355 | 10.64 |
| 1CSP | 67 | 505 | 0.4313 | 0.4149 | 0.1184 | 0 | 0 | 0.9611 | 11.84 |
| 1IGD | 61 | 468 | 0.3262 | 0.3173 | 0.1029 | 0 | 0 | 0.9789 | 10.97 |
| 1L2Y | 20 | 154 | 0.3407 | 0.2973 | 0.1037 | 0 | 0 | 0.9706 | 3.609 |
| 1MJC | 69 | 514 | 0.385 | 0.3627 | 0.1128 | 0 | 0 | 0.9587 | 12.05 |
| 1PGB | 56 | 436 | 0.4413 | 0.4163 | 0.14 | 0 | 0 | 0.9706 | 10.22 |

## Baseline comparison (fixed backbone cohort, n_res=20, 80 atoms)

Backbone RMSD (A) vs latent-float budget; lower RMSD + higher compression is better.

| method | latent_floats | compression_ratio | mean_backbone_rmsd |
|---|---|---|---|
| **learned_autoencoder** | 128 | 1.875 | 0.3636 |
| identity | 240 | 1 | 0 |
| mean_shape | 0 | inf | 4.917 |
| pca_k2 | 2 | 120 | 2.832 |
| pca_k4 | 4 | 60 | 2.08 |
| pca_k8 | 8 | 30 | 1.267 |
| pca_k16 | 16 | 15 | 0.154 |

## Timing & memory

- mean inference: 0.007482 s/structure
- peak host RSS: 478.8 MB
- GPU memory: N/A (CPU-only run)
