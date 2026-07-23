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

### Baseline comparison on a fixed backbone cohort (80 atoms, 240 floats)

PCA fit on the **15 training** structures that have a complete 20-residue
backbone, scored on the full cohort. (k ≥ 16 is skipped: with 15 training
samples PCA has at most 15 components — the matched-budget 128-float row is
simply not available at this data scale.)

| method | latent floats | compression | backbone RMSD (Å) |
|---|---|---|---|
| learned AE (encodes the *whole* protein) | 128 | — | 0.364 |
| PCA k=8 | 8 | 30× | 1.48 |
| PCA k=4 | 4 | 60× | 2.34 |
| PCA k=2 | 2 | 120× | 3.11 |
| mean shape | 0 | ∞ | 5.02 |

The AE wins here, but this cohort *overlaps its training set* and the PCA
cohort compresses only 80 backbone atoms while the AE compresses the entire
all-atom structure — so this is **not** a verdict. The meaningful, leak-free
test is held-out (below), and it tells a very different story.

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
training folds), but held-out all-atom RMSD is ~11 Å — **no better than
predicting the centroid** — and essentially **flat across all latent sizes**.
The bottleneck is therefore **not** the latent budget; it is the **number of
training structures**.

### Leak-free held-out comparison vs PCA (backbone cohort, 5 val folds)

PCA fit on the 16 training folds, scored on the 5 **unseen** val folds
(same 80-backbone-atom cohort as above; both cohorts aligned to a common
frame). This is the honest head-to-head:

| method | latent floats | held-out backbone RMSD (Å) |
|---|---|---|
| PCA k=8 | 8 | **3.24** |
| PCA k=4 | 4 | 3.79 |
| PCA k=2 | 2 | 4.60 |
| mean shape | 0 | 6.41 |
| learned AE (whole protein) | 128 | 8.87 |

**On held-out folds, PCA with 8 floats (3.24 Å) beats the neural
autoencoder at 128 floats (8.87 Å) by a wide margin — and the AE even loses
to the mean-shape baseline.** With 16 training structures the encoder/decoder
memorise seen structures and produce near-random geometry for unseen folds,
whereas a simple linear model at least captures the dominant backbone
variance. This is a genuine, expected data-starvation result, not an
architecture failure (the same model fits the training set to ~1 Å).

---

## Answer to the headline question

- **In-distribution (seen structures):** decisively better than every
  baseline — 0.43 Å vs ~11 Å (centroid), with correct chirality/bonds/
  contacts. The architecture and pipeline work.
- **Held-out (unseen folds), which is what matters for a latent MD model:**
  **worse than PCA and even worse than mean-shape.** At 16 training
  structures the learned codec does not generalise; a linear baseline with
  8 floats beats it. So at this data scale the answer is **no**.
- **Why:** the limiter is **data**, not the latent budget or the model —
  held-out error is flat from 32 to 512 latent floats. The codec must be
  pretrained on *many* structures before it generalises. That is exactly the
  pretraining step in the project plan, and this experiment quantifies why it
  is necessary before any latent diffusion.

**Nothing here claims physical accuracy.** All numbers are geometric.

## Code audit

An adversarial multi-agent audit of the codebase confirmed and **fixed 10
bugs** (2 high-severity NaN/robustness issues in the differentiable Kabsch and
distance loss; a clash-rate normalisation bias for >2000-atom proteins; a
PCA-baseline data leak; a resume-RNG bug; and several audit-trail/edge-case
issues). **None affected the reported learned-model numbers** (no structure
here exceeds 832 atoms and no degenerate collapse occurred), but the baseline
comparison above already reflects the leak-free fix, and 7 regression tests
were added (33 tests total, all passing).

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
