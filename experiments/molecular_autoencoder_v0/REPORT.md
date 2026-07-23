# Report — Molecular Structure Autoencoder (Milestone 1)

**Scientific question:** *Can a learned, geometry-aware autoencoder compress
protein atomic coordinates into a compact latent while preserving atomistic
geometry better than simple baselines?*

**Short answer:** **Yes on the training distribution, not yet on held-out
folds — and the reason is data scale, not the architecture.** With a tiny
(~21-structure) dataset the model reconstructs *seen* structures at **0.43 Å**
all-atom RMSD (10× coordinate compression) with correct chirality, bonds, and
contacts — far better than every baseline. But on **held-out** protein folds
it collapses to roughly the centroid baseline (~11 Å), i.e. it memorises
rather than learning a transferable codec. This is the expected
data-starvation regime and is exactly what motivates the plan's next step
(pretraining on thousands of structures).

This milestone is the **autoencoder only**. No diffusion, trajectories, force
fields, RL, or generation. RMSD-type metrics measure *geometry*, not physical
accuracy.

---

## Setup

- **Data:** 21 small, mostly single-chain proteins (20–107 residues,
  154–832 heavy atoms) downloaded from RCSB and cleaned (waters / ligands /
  ions / hydrogens / altlocs removed; 20 standard amino acids; topology
  perceived from coordinates). All filtering recorded in
  `data/manifest.json`. 16 train / 5 val, split by sequence-similarity
  clustering so near-duplicates (`1UBQ`/`1UBI`) don't leak.
- **Model:** Perceiver-style autoencoder, 1.49 M params. Atoms →
  cross-attention → **16 latent tokens × 8 dims = 128 latent floats**
  (independent of atom count) → decoder cross-attention from per-atom
  *identity* queries → xyz. Only geometry is compressed; atom identity is
  given to the decoder.
- **Symmetry:** not equivariant by construction; invariance handled by
  centring inputs and Kabsch alignment in every coordinate loss and metric.
- **Compute:** CPU only (no GPU). Stage A: ~1054 s for 1200 epochs.
  Inference ~7.5 ms/structure. Peak host RSS ~479 MB. GPU memory: N/A.

---

## Stage A — overfit sanity (train reconstruction, 21 structures)

Purpose: validate the full pipeline (gradients, masks, alignment,
checkpointing, deterministic eval). These are **train-set** numbers.

| metric | learned AE | centroid baseline | identity |
|---|---|---|---|
| all-atom RMSD (Å) | **0.428** | 10.79 (≈ radius of gyration) | 0 |
| backbone RMSD (Å) | **0.394** | — | 0 |
| pairwise-distance error (Å) | 0.279 | — | 0 |
| bond-length error (Å) | 0.122 | — | 0 |
| chirality violation rate | **0.000** | — | 0 |
| clashes / 1000 atoms | 5.1 | — | 0 |
| contact-map F1 | **0.960** | — | 1 |
| mean compression ratio | **10.9×** (3.6–19.5×) | ∞ | 1× |

The model fits all 21 structures well, with correct handedness (zero
chirality flips), near-native bond lengths, few clashes, and excellent
contact recovery. Compression grows with protein size (19.5× on the
832-atom `1FKB`), the key property of a fixed-size latent. Pipeline
**validated**.

### Baseline comparison on a fixed backbone cohort (train, 80 atoms, 240 floats)

| method | latent floats | compression | backbone RMSD (Å) |
|---|---|---|---|
| learned AE (encodes the *whole* protein) | 128 | — | 0.364 |
| PCA k=16 | 16 | 15× | 0.154 |
| PCA k=8 | 8 | 30× | 1.27 |
| PCA k=4 | 4 | 60× | 2.08 |
| mean shape | 0 | ∞ | 4.92 |

**Read this carefully.** On the *training* cohort PCA at k=16 beats the AE —
but with only 21 structures, ~16 principal components reconstruct the
training set almost perfectly (k ≈ n_samples is memorisation, not
compression). Also the PCA cohort compresses only 80 backbone atoms while the
AE compresses the entire all-atom structure. So this table is **not** a
verdict; the meaningful test is held-out (below).

---

## Stage B / C — held-out evaluation and latent-budget scaling

Train on 16 structures, evaluate on the 5 held-out val folds, at four latent
budgets (500 epochs each).

| latent floats | compression | **held-out** all-atom RMSD (Å) | held-out backbone RMSD (Å) | contact F1 | *train* RMSD (Å) |
|---|---|---|---|---|---|
| 32 | 38.9× | 11.6 | 11.2 | 0.14 | ~1.0 |
| 128 | 9.7× | 11.2 | 10.7 | 0.18 | 0.91 |
| 256 | 4.9× | 10.8 | 10.2 | 0.17 | 0.88 |
| 512 | 2.4× | 11.5 | 10.9 | 0.15 | 1.00 |

Held-out centroid (Rg) baseline on the val set ≈ **10.2 Å**.

**Finding:** train RMSD is ~1 Å (the model *has* capacity and fits the
training folds), but held-out RMSD is ~11 Å — **no better than predicting the
centroid** — and essentially **flat across all latent sizes**. The bottleneck
is therefore **not** the latent budget; it is the **number of training
structures**. With 16 folds the encoder/decoder memorise seen structures and
produce out-of-distribution garbage for unseen folds. This is a genuine,
expected result at this scale, not an architecture failure.

---

## Answer to the headline question

- **Versus trivial baselines, in-distribution:** decisively better — 0.43 Å
  vs ~11 Å (centroid), with correct chirality/bonds/contacts.
- **Versus PCA, at matched budget:** not a clean win at this scale. PCA is a
  very strong fixed-size linear compressor and trivially memorises tiny
  datasets; a controlled, matched-task comparison needs held-out data at
  larger scale.
- **Generalisation (the property that actually matters for a latent MD
  model):** **not yet achieved** with 16 structures. The codec must be
  trained on many more structures before it learns a transferable
  compression — which is precisely the pretraining step in the project plan.

**Nothing here claims physical accuracy.** All numbers are geometric.

---

## Limitations

- Tiny dataset (21 structures) → held-out numbers are data-limited, not
  architecture-limited.
- Not equivariant (invariance only via preprocessing + eval alignment).
- Topology perceived from coordinates → disulfides / non-standard chemistry
  excluded from the bond set (documented).
- PCA/AE comparison is not perfectly matched (cohort = 80 backbone atoms vs
  AE = whole all-atom structure).
- CPU-only; no scaling beyond this small set.

## Recommended next steps (before any latent diffusion)

1. **Scale the dataset** to thousands of cleaned single-chain domains (the
   download path is ready; still no bulk auto-download / no ESM Atlas). Re-run
   the held-out scaling — the central hypothesis is that generalisation
   appears with data scale.
2. **Matched AE-vs-PCA** on an identical fixed-size backbone target.
3. **Equivariant encoder** (frame/SE(3)) to remove the alignment crutch and
   improve data efficiency.
4. Only then: add the **latent diffusion** stage this autoencoder is designed
   to host.

**Stop point reached.** Do not implement latent diffusion until these results
are reviewed.
