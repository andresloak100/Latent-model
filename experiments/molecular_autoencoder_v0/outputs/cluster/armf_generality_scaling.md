# Secondary A: non-protein generality (MISATO ligands) + Secondary B: scaling

## A. Ligand codec -- ANM is meaningfully a PROTEIN codec (prediction confirmed)

MISATO ligand heavy atoms (24-system spanning sample of 16,972), codec vs null / ANM
/ per-system PCA. L reported relative to 3N-6 (compression ratio) on every row.

**ANM captured fraction of the PCA ceiling** = (1 - r_anm/null)/(1 - r_pca/null):

| L | ANM/PCA ceiling (ligands) | protein CA reference | mean compression ratio |
|---|---|---|---|
| 4  | 38% | ~67% (2/3) | 7% |
| 8  | 43% | ~67% | 13% |
| 16 | 49% | ~67% | 27% (vacuous for the smallest ligands) |

**Prediction confirmed:** ANM captures only ~38-49% of the achievable (PCA) signal on
ligands, vs ~2/3 on proteins. The ligand dynamics IS compressible (PCA reaches low
residual, e.g. 2XYS null 0.27 -> PCA 0.06), but ANM's structure-only collective modes
capture much less of it -- exactly the torsional/local motion ANM handles worst. The
zero-parameter general codec is meaningfully protein-biased: a direct, quantified hit
on objective 1 (general molecules), known now rather than later. Caveat: the smallest
ligands (nHvy < ~16) have L=16 approaching the full space (compression vacuous, e.g.
4UFK 9 heavy -> 76%); the L=4 column (7% ratio) is the clean comparison.

## B. Scaling / timing -- Lanczos returns the modes but shift-invert is O(N^2)

Sparse ANM Hessian (contact graph), leading 64 modes via shift-invert Lanczos
(scipy eigsh, sigma=1e-6). CPU timing.

| N | contacts | eigsh_s | dense_s | match (rel) | enc_ms | dec_ms | 1step_ms | peakMB |
|---|---|---|---|---|---|---|---|---|
| 200 | 6.7k | 0.08 | 0.13 | 3e-9 | ~0 | ~0 | 110 | 540 |
| 1000 | 55k | 1.42 | 2.25 | 6e-10 | 0.01 | 0.01 | 111 | 1017 |
| 2000 | 121k | 5.54 | - | - | 0.16 | 0.57 | 100 | 1281 |
| 5000 | 331k | 34.0 | - | - | 0.19 | 0.04 | 99 | 2422 |
| 10000 | 700k | 166 | - | - | 0.06 | 0.10 | 17 | 4433 |
| 20000 | 1.46M | 907 | - | - | 0.12 | 0.29 | 17 | 8784 |

- **Lanczos returns the correct leading 64 modes** (matches dense to ~1e-9 where
  dense is feasible). Encode/decode/diffusion-step are cheap (sub-ms to ~0.1 s) and
  scale near-linearly.
- **But shift-invert Lanczos scales O(N^2.03)** -- the LU factorization of a 3D-mesh
  Hessian has O(N^2) fill-in. Projected to 1e6 atoms: **~1.9e6 s (~22 days) per
  eigensolve**, peak memory blows up (8.8 GB already at N=20k). So the eigensolve, not
  the neural steps, is the bottleneck, and **the search economics do NOT close with
  shift-invert Lanczos (~0 candidates/GPU-hour at 1e6).**
- **Path:** the ROADMAP 8.1 "Lanczos at a fraction of the cost" claim needs
  qualification -- it must be MATRIX-FREE and PRECONDITIONED (LOBPCG + multigrid /
  Jacobi), avoiding the LU factorization, to approach O(N.k.iters). Not yet tested.
  Assumptions: constant contact density; DDPM transition transfers across sizes.
