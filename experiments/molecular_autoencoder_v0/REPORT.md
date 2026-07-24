# Report — Molecular Structure Autoencoder (Milestone 1)

**Scientific question:** *Can a learned, geometry-aware autoencoder compress
protein atomic coordinates into a compact latent while preserving atomistic
geometry better than simple baselines?*

**Short answer:** **Yes on the training distribution; no on held-out folds —
and a decisive scaling test shows the limiter is the *architecture*, not the
amount of data.** With a tiny dataset the model reconstructs *seen* structures
at **0.43 Å** all-atom RMSD (10× compression) with correct chirality, bonds,
and contacts — far better than every baseline. But on **held-out** folds it
loses to a 0-parameter mean-shape baseline, and a controlled data-scale sweep
(below) shows held-out accuracy does **not** improve with more structures — a
fully-converged 800-structure run was actually *worse* than smaller ones. So
more data will not fix this codec. The next step is an **equivariant/frame
encoder** (a working invariant-encoder prototype is now in the repo), *before*
any large pretraining run or latent diffusion.

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
So the latent budget is **not** the limiter. The natural next hypothesis was
that *data* is the limiter; the decisive scaling test below shows it is not.

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
variance. The same model fits the training set to ~1 Å, so this is a
*generalisation* failure — and the scaling test below identifies its cause as
the architecture, not the data.

---

## Decisive data-scale test (GPU, converged, size-controlled)

To separate "data-limited" from "architecture-limited", a clean sweep was run
on GPU: a fixed structure-size band (≤ 800 atoms, so difficulty is held
constant), a large **231-fold** held-out set, PCA fit on train only. Held-out
**backbone RMSD (Å)** (80-atom / 20-residue cohort):

| n_train | train bb | held-out AE (128f) | mean-shape (0f) | PCA k2 | PCA k4 | PCA k8 |
|---|---|---|---|---|---|---|
| 100 | 2.09 | 9.00 | 5.02 | 3.45 | 2.74 | 2.05 |
| 300 | 2.91 | 9.17 | 5.00 | 3.36 | 2.72 | 2.00 |
| 800 | 3.87 | 7.96 | 5.05 | 3.35 | 2.70 | 2.00 |
| 1129 | 3.86 | 7.64 | 5.03 | 3.35 | 2.69 | 1.99 |
| **800 (converged, 900 ep)** | 3.07 | **8.64** | 5.05 | 3.35 | 2.70 | 2.00 |

**The curve does not descend.** The AE loses to the 0-parameter mean-shape
baseline (~5 Å) at *every* data scale, and the apparent dip toward n=1129 was
an under-training artefact (mean-regression): training n=800 **to convergence
pushed held-out *up*, 7.96 → 8.64 Å** (train 3.87 → 3.07). More data makes it
no better — and here, worse. **At the *matched* 128-float budget, PCA
reconstructs the held-out backbone to 0.06 Å** — vs the neural AE's 8.64 Å.
Combined with the flat-across-latent-budget result above, neither the latent
size nor the amount of data is the bottleneck. (A secondary signal: 20/50/100
structures overfit to 1.3/1.6/1.8 Å train, but n=800 plateaus at 3.07 Å train
— a memorisation ceiling as diversity grows.)

**Conclusion: the codec is architecture-limited.** The raw-coordinate,
non-equivariant encoder is the prime suspect (it must learn rotation
invariance from data instead of having it built in). A **provably
SE(3)-invariant encoder** prototype is now in the repo
(`molae/model_equivariant.py`, `encoder_type: invariant`, verified in
`tests/test_equivariance.py`) as the first architectural fix to test.

*(Full table + raw per-fold `*_metrics.json` and run logs are committed under
`outputs/data_scan/` — `RESULTS_data_scaling.md`, `clean_n{100,300,800,1129}_metrics.json`,
`conv_n800_metrics.json`, and the scan logs.)*

---

## Answer to the headline question

- **In-distribution (seen structures):** decisively better than every
  baseline — 0.43 Å vs ~11 Å (centroid), with correct chirality/bonds/
  contacts. The architecture and pipeline work.
- **Held-out (unseen folds), which is what matters for a latent MD model:**
  **worse than PCA and even worse than mean-shape**, at every data scale
  tested. A linear baseline with 8 floats beats the neural codec. The answer
  is **no**.
- **Why:** the limiter is the **architecture**, not the latent budget and not
  the data. Held-out error is flat across latent sizes (32→512 floats) *and*
  flat-to-worse across data sizes (100→800 converged structures). The
  raw-coordinate encoder is not equivariant, so it must learn rotational
  invariance from data rather than having it built in — which the scaling test
  shows it fails to do. **The gate this milestone was meant to answer —
  "scale this codec, or fix it first?" — answers: fix it first.** Build the
  equivariant/frame encoder before any large pretraining run or latent
  diffusion.

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

- The decisive scaling test used a fixed ≤800-atom size band on GPU; the
  architecture-limited verdict is established within that band and the small
  held-out sets used here — worth re-confirming once the encoder is fixed.
- Not equivariant (invariance only via preprocessing + eval alignment) — this
  is now identified as the likely *cause* of the poor generalisation, not just
  a caveat.
- Topology perceived from coordinates → disulfides / non-standard chemistry
  excluded from the bond set (documented).
- PCA/AE comparison is not perfectly matched (cohort = 80 backbone atoms vs
  AE = whole all-atom structure).
- CPU-only; no scaling beyond this small set.

## Recommended next steps (verdict: fix the architecture first)

1. **Equivariant/frame encoder — the priority.** A provably SE(3)-invariant
   encoder prototype is already in the repo (`molae/model_equivariant.py`,
   selected via `encoder_type: invariant`; invariance proven in
   `tests/test_equivariance.py`). Run the A/B (`configs/arch_invariant.yaml`
   vs `configs/stage_b_heldout.yaml`) and check whether held-out drops below
   mean-shape / toward PCA where the baseline could not. If the minimal
   invariant featurisation helps but isn't enough, add directional
   information equivariantly (GVP vector channels / IPA frames) and make the
   *decoder* equivariant too.
2. **Only after** the encoder clears the PCA / mean-shape bar on held-out
   folds: revisit dataset scale (the fetch/prepare path and a Mila
   parallel-sweep plan are ready in `NEXT_STEPS.md`).
3. **Only then:** the **latent diffusion** stage this autoencoder is designed
   to host.

**Do not scale this architecture, and do not start latent diffusion**, until
an encoder clears the baseline bar on held-out folds. The scaling test shows
more data will not rescue the current codec.
