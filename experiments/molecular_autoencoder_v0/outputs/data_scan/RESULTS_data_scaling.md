## Data-scaling result — does the autoencoder generalize with more structures?

Leak-free held-out backbone RMSD (80-atom / 20-res cohort, 231 val folds, PCA fit on train only).
Model: 128-float latent, wd=0. Data: <=800-atom band. Fixed 300-epoch budget unless noted.

| n_train | train bb (A) | **held-out AE (128f)** | mean-shape (0f) | PCA k2 | k4 | k8 | AE < mean-shape? |
|---|---|---|---|---|---|---|---|
| 100 | 2.09 | 9.00 | 5.02 | 3.45 | 2.74 | 2.05 | NO |
| 300 | 2.91 | 9.17 | 5.00 | 3.36 | 2.72 | 2.00 | NO |
| 800 | 3.87 | 7.96 | 5.05 | 3.35 | 2.70 | 2.00 | NO |
| 1129 | 3.86 | 7.64 | 5.03 | 3.35 | 2.69 | 1.99 | NO |
| **800 (converged, 900 ep)** | 3.07 | **8.64** | 5.05 | 3.35 | 2.70 | 2.00 | NO |

**Verdict:** the AE loses to the 0-float mean-shape baseline at every data scale, converged or not.
Apparent held-out 'improvement' with n (9.17->7.64) tracked rising train error = **mean-regression**, not
generalization: converging n=800 (train 3.87->3.07) pushed held-out UP (7.96->8.64), away from the mean.
Held-out is data-scale-INSENSITIVE and architecture-limited (non-equivariant, Kabsch-alignment crutch).
Also note a memorization ceiling: 20/50/100 structs overfit to 1.3/1.6/1.8 A, but 800 plateaus at 3.07 A train.
**Recommendation:** build the equivariant/frame encoder before any large pretraining run; do NOT scale data on this codec.
