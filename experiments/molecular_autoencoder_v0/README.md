# Molecular Structure Autoencoder (v0)

A **minimal, reproducible** autoencoder that compresses protein **atomic
coordinates** into a small latent representation and reconstructs the
structure. This is a research prototype testing the *first* technical
assumption behind a latent molecular-dynamics model:

> Can protein atomic structure be compressed into a compact latent space and
> decoded back with geometry preserved *better than simple baselines*?

This is the encoder/decoder piece that a later **latent diffusion** stage
would operate inside (compress → run diffusion in latent space → decode),
mirroring how modern (2024+) text-to-video models move diffusion out of pixel
space into a learned latent. **This milestone builds only the autoencoder.**

## What the system is being built to do

Four objectives. `ROADMAP.md` §6 audits the current stack against each and
sets the ordering of work.

1. Scale to **1M+ atoms**, general molecules and not only proteins.
2. Model **enzyme-like bond breaking and forming**.
3. Scale to **multi-millisecond** timescales.
4. Be efficient enough to **inference on a single GPU** — training compute is
   worth spending to buy inference efficiency.

Two of these are reachable by scaling the current design; two are not, and the
roadmap says which is which rather than leaving it implied.

## What this is NOT (scope guardrails)

There is **no** diffusion, trajectory prediction, force field, molecular
generation, or reinforcement learning here. No MD trajectories are used. We do
not touch any production code. We do **not** claim physical or energetic
accuracy — RMSD and the other metrics quantify *geometric* reconstruction
only. "Failed" or unsupported chemistry is labelled, not hidden.

## Scientific question and honest limitations

- **Question:** does a learned, geometry-aware autoencoder compress atomic
  coordinates while preserving atomistic geometry (bonds, chirality,
  clashes, contacts) better than PCA / trivial baselines?
- **Symmetry:** the network is **not** rotation/translation equivariant by
  construction. Global rigid motion is removed by (a) **centring** inputs and
  (b) **Kabsch alignment** in every coordinate loss and metric. So the
  *evaluation* is rotation/translation invariant even though the *architecture*
  is not. A future equivariant encoder is an obvious next step.
- **Decoder conditioning:** the decoder is given each atom's **identity**
  (element, residue type, atom name, residue index) and must predict only its
  **3D position** from the latent. Analogy: a video decoder is told the frame
  shape; here the "shape" is the sequence/topology. Only *geometry* is
  compressed, not identity. This is a deliberate, documented choice.
- **Topology** is *perceived* from the deposited coordinates using covalent
  radii (intra- and consecutive-residue pairs only), not read from a monomer
  library — so the pipeline is self-contained. Disulfides and other long-range
  bonds are therefore **not** in the bond set (documented, not silent).
- **Scale:** the shipped benchmark is tiny (~21 small proteins). Numbers are
  illustrative of the *pipeline*, not production quality. Full Stage B/C need
  more structures (the download mechanism is ready; nothing is auto-bulk-
  downloaded, and ESM Atlas is intentionally not used).

## Method

```
atoms (identity + xyz) --cross-attn--> L latent tokens --self-attn-->
        bottleneck (L x latent_dim)  <-- the compressed code
per-atom identity queries --cross-attn--> latents --> MLP --> xyz
```

- **Encoder / Bottleneck / Decoder** are separate modules (`molae/model.py`)
  so any one can be swapped (e.g. for an equivariant or diffusion variant).
- The latent size **(L × latent_dim)** is independent of the atom count, so
  **compression grows with protein size** — the key property motivating a
  latent approach for long sequences.

### Losses (`molae/losses.py`)
Aligned coordinate MSE (Kabsch) · pairwise-distance preservation · bond-length
preservation (perceived topology) · clash hinge penalty · chirality (signed
volume) preservation.

### Metrics (`molae/metrics.py`), reported separately
All-atom RMSD · backbone RMSD · pairwise-distance error · bond-length error ·
chirality violation rate · clash rate (per 1000 atoms) · contact-map F1 ·
compression ratio · inference time · peak memory (GPU memory is **N/A** on a
CPU run; host RSS reported instead).

### Baselines (`molae/baselines.py`)
1. **PCA** on a fixed-length backbone cohort (the only clean way to run a
   linear method across variable-size proteins), at matched float budgets.
2. **Trivial:** identity (predict truth, RMSD 0) and centroid (predict the
   radius of gyration, 0 latent floats) for the full all-atom setting;
   mean-shape for the cohort.
3. **Learned autoencoder.**

> **Fairness caveat (read this before comparing):** the PCA cohort compresses
> only the first-20-residue backbone (80 atoms). The learned AE compresses the
> *entire all-atom* structure into its latent and is merely *measured* on that
> backbone. They are not matched in what they encode, so that table is
> indicative, not a controlled head-to-head. On tiny training sets PCA also
> trivially memorises (k ≈ n_samples), so the **held-out** comparison is the
> meaningful one. The clean apples-to-apples comparison for the AE's actual
> task is against the trivial all-atom baselines.

## Dataset

Curated small, mostly single-chain proteins (`data/pdb_ids_stage_a.txt`),
downloaded as mmCIF from RCSB on demand. Per structure the pipeline: keeps the
first model (NMR → model 1); selects one peptide chain (others recorded);
removes waters, ligands, ions, hydrogens; collapses alternate conformations;
keeps only the 20 standard amino acids' heavy atoms (non-standard residues
dropped and recorded); records missing-residue gaps and incomplete backbones.
Every decision is written to `data/manifest.json`. Train/val split uses
single-linkage clustering on sequence similarity so near-duplicates (e.g.
`1UBQ`/`1UBI`) never straddle the split.

## Reproduce

```bash
pip install -r requirements.txt   # torch CPU wheel is fine (see file header)

# 1. build the dataset (downloads ~22 small PDBs, no bulk download)
python scripts/prepare_dataset.py --config configs/stage_a_overfit.yaml

# 2. Stage A: overfit sanity (validates the full pipeline)
python scripts/train.py --config configs/stage_a_overfit.yaml
python scripts/eval.py  --config configs/stage_a_overfit.yaml

# 3. Stage B (small): held-out evaluation
python scripts/train.py --config configs/stage_b_heldout.yaml
python scripts/eval.py  --config configs/stage_b_heldout.yaml

# 4. Stage C (small): latent-budget scaling table
python scripts/run_scaling.py --base configs/stage_b_heldout.yaml --epochs 500

# tests
python -m pytest tests/ -q
```

All runs are **resumable** (`outputs/<run>/latest.pt`), deterministic
(seeded), and save config, environment (versions + git commit), checkpoints,
metrics, and example reconstructions (`outputs/<run>/reconstructions/*.pdb`).

## Layout

```
molae/            parsing, dataset, alignment, losses, metrics, model,
                  baselines, pdb_io, config, utils, synthetic
scripts/          prepare_dataset, train, eval, run_scaling
configs/          stage_a_overfit.yaml, stage_b_heldout.yaml
tests/            parsing, masking, alignment, losses, reconstruction, synthetic
data/             pdb_ids_stage_a.txt (+ raw/, processed/, manifest.json)
outputs/          checkpoints, metrics, reconstructions
REPORT.md         results + answer to the headline question
```

See `REPORT.md` for measured results and the answer to the scientific
question. **Stop point:** this is the autoencoder milestone. Do not add latent
diffusion until these results are reviewed.
