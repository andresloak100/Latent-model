# Stage C (small): latent-budget scaling (held-out)

Epochs per point: 500. Held-out val split.

| latent tokens | latent dim | latent floats | mean compression | held-out all-atom RMSD (A) | held-out backbone RMSD (A) | bond err (A) | contact F1 |
|---|---|---|---|---|---|---|---|
| 8 | 4 | 32 | 38.9 | 11.6 | 11.2 | 1.77 | 0.136 |
| 16 | 8 | 128 | 9.74 | 11.2 | 10.7 | 1.5 | 0.176 |
| 16 | 16 | 256 | 4.87 | 10.8 | 10.2 | 1.34 | 0.167 |
| 32 | 16 | 512 | 2.43 | 11.5 | 10.9 | 1.58 | 0.154 |
