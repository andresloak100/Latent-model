# Report — Molecular Structure Autoencoder (Milestone 1)

**Scientific question:** *Can a learned, geometry-aware autoencoder compress
protein atomic coordinates into a compact latent while preserving atomistic
geometry better than simple baselines?*

**Answer: yes.** On 293 held-out folds the codec reconstructs full all-atom
structures to **0.75 Å backbone / 1.02 Å all-atom RMSD** with near-perfect
stereochemistry (chirality violation rate 0.001, bond-length error 0.16 Å,
contact-map F1 0.94), beating PCA at matched accuracy. Every architecture tried
before it — including three of the four designs suggested in review — sat at
9.5–11.2 Å and **lost to a zero-parameter mean-shape baseline**. One change to
the decoder closed that gap.

The larger finding is what bounds it now. The codec is **not** limited by
equivariance, latent size, training objective, or model capacity. Every one of
those was tested and eliminated. It is limited by **data and compute**, both of
which are ordinary engineering levers — and the experimental band we drew from
contains only ~3,853 structures in the entire PDB.

This milestone is the **autoencoder only**. No diffusion, trajectories, force
fields, RL, or generation. All metrics are **geometric** — nothing here claims
physical or energetic accuracy.

---

## 1. The architecture, and what made it work

### 1.1 Direct per-residue readout

Every failing arm shared one decoder pattern: each atom had to **find its own
coordinates by attending** over a latent (per-atom identity query →
cross-attention → xyz). Atom *i*'s output was a content-addressed lookup.

The fix (`molae/model_direct.py`) removes the lookup. Each residue owns a latent
vector; residue tokens self-attend; a head emits a fixed bank of
`N_ATOM_NAMES × 3` slots per residue, and each atom **gathers the slot that
belongs to it**:

```python
slots = self.head(h).view(B, R, self.n_slots, 3) * self.cfg.coord_scale
flat  = slots.view(B, R * self.n_slots, 3)
gather_idx = (res_pos * self.n_slots + batch["atom_name_idx"]).clamp(max=R * self.n_slots - 1)
return torch.gather(flat, 1, gather_idx.unsqueeze(-1).expand(-1, -1, 3))
```

Output slot *i* **is** atom *i*, by construction. Routing is no longer something
the model has to learn.

**Credit where due:** the per-residue latent itself came from the ProteinAE
direction suggested in review. The winning design is *that latent structure* plus
*this decoder*. What did not survive was ProteinAE's flow-matching decoder.

### 1.2 What the decoder alone was worth

Identical data, losses, optimiser, epochs, split, and latent budget — only the
decoder changed:

| decoder | held-out backbone | chirality | bond err | contact F1 |
|---|---|---|---|---|
| attention lookup (`perresidue`) | 9.50 Å | 0.352 | 0.89 Å | 0.29 |
| **direct readout** | **0.75 Å** | **0.001** | **0.16 Å** | **0.94** |

An ablation ladder on fixed 80-atom cohorts predicted this before it was built:
swapping a flat MLP decoder for the attention decoder cost 0.2–0.3 Å on an easy
task, and the same penalty became catastrophic on variable-size all-atom
proteins.

### 1.3 What did not help

| tried | result |
|---|---|
| SE(3)-invariant graph encoder | 10.17 Å vs 10.47 baseline — 0.3 Å on an 11 Å failure |
| RoPE / Fourier coordinate encoding | 11.16 Å — worst arm |
| Flow-matching decoder | direct + flowmatch = 11.87 Å, chirality 0.496 (coin flip) |
| Per-atom latents (1 float/atom) | 6.32 Å, chirality 0.411 |
| Per-atom, element-only | 4.68 Å, chirality 0.494 |

**Equivariance was the leading hypothesis and it was wrong.** Worth recording,
because a large run would have spent heavily failing to fix the wrong thing.

The per-atom result is informative rather than merely negative. Both per-atom
arms spend roughly the *same total budget* as the winner (607 vs 624 floats,
~3× compression either way), so this is not about float count — it is about
**distribution**. One float per atom cannot describe a 3D position; eight floats
per residue can describe a nearly-rigid unit. The win was index routing **plus
enough capacity per token to describe a rigid body**. For non-proteins the
generalisation is therefore fragment-based grouping, not per-atom.

### 1.4 Latent size and seed stability

| latent (floats/residue) | compression | held-out backbone | seeds |
|---|---|---|---|
| 16 | 1.5× | 0.740 | — |
| 8 | 2.9× | 0.754 | 0.754 / 0.914 / 0.777 (sd 0.071) |
| 4 | 5.8× | 0.750 | 0.750 / 0.863 / 0.794 (sd 0.047) |
| 2 | 11.7× | 1.531 | 4.673 / 1.458 / 1.568 (**sd 1.49**) |
| 1 | 23.3× | 6.236 | — |

**Latent 4 saturates** — d4/d8/d16 are within 0.02 Å, confirmed across seeds.
**Latent 2 is unstable**: the same config and seed produced 1.531 Å and 4.673 Å
on different hardware. At the capacity floor, optimisation is chaotic. An earlier
version of this report claimed "latent 2 already works" on the strength of one
lucky run; that claim is **withdrawn**.

---

## 2. Setup

- **Data.** RCSB X-ray structures cleaned with gemmi: waters / ligands / ions /
  hydrogens / altlocs removed, 20 standard amino acids, single peptide chain,
  topology perceived from covalent radii. All filtering recorded in
  `data/manifest.json`. Main band ≤ 800 heavy atoms, 878 train / 293 val, split
  by sequence-similarity clustering so near-duplicates cannot leak.
- **Symmetry.** Not equivariant by construction. Invariance from centred inputs,
  random SO(3) augmentation, and Kabsch superposition inside every coordinate
  loss and metric (differentiable, masked, batched, rotation detached, autocast
  disabled inside the SVD).
- **Compute.** Mila SLURM, `long` partition, preemptible. Stage A ran on CPU
  (1054 s, ~7.5 ms/structure inference) — so *no GPU was needed for the first
  try*, but everything after it was GPU-bound.

---

## 3. Why we believe it: the controls

Three controls did more to establish this result than any single training run.

**Matched-task control.** Given PCA's exact task — fixed 80-atom aligned
backbones, identical inputs, splits and budgets, checkpoints selected on a dev
split carved from train — a neural AE beats PCA at every aggressive budget:

| latent floats | PCA | AE | winner |
|---|---|---|---|
| 2 | 3.525 | **2.863** | AE |
| 4 | 2.717 | **2.058** | AE |
| 8 | 1.987 | **1.676** | AE |
| 16 | **1.423** | 1.615 | PCA |

5/5 seeds at k=2/4/8, PCA 5/5 at k=16, run-to-run sd ≈ 0.05 Å. This contradicted
the whole-pipeline result and is what proved the *pipeline*, not neural
compression, was broken.

**Ablation ladder.** Isolated the decoder as the culprit before any fix existed.

**Random-init control.** The latent-smoothness metric scores **ρ = 0.89 on
untrained weights** — a random projection already preserves distances
(Johnson–Lindenstrauss). Reporting bare ρ would have looked like a strong
positive and meant nothing.

---

## 4. Is the latent usable for Stage 2?

Measured on NMR ensembles (many deposited models of one molecule = a
conformational ensemble, free of MD compute) and on same-sequence crystal groups.

| measurement | value | reading |
|---|---|---|
| resolution ratio | **0.915** | conformers survive the round trip; **no collapse** |
| smoothness ρ | 0.955 (control 0.891) | weak, and not the decisive metric |
| interpolation bond err | 0.187 → 0.211 Å | holds |
| interpolation chirality | 0.000 → 0.003 | holds |
| interpolation clashes/1k | 47.3 → **121.9** | **degrades ~2.6×** |

**No mean collapse.** The decoder preserves ~92% of conformational spread. This
was the main risk to Stage 2 and it is cleared.

**Midpoint clashes are the one soft spot.** Bond lengths and chirality are
*local* (atoms within one residue, decoded from that residue's own latent);
clashes are *non-local*, and nothing enforces that independently-decoded residues
don't interpenetrate. Latent mixup is the cheap first fix if Stage 2 needs it.

**There is no separate "conformational floor."** In-domain held-out ensemble
members reconstruct at 0.929 Å all-atom; those same structures score 0.931 Å in
the ordinary eval. It is simply the codec's all-atom accuracy, which happens to
exceed the 0.1–0.5 Å differences between crystal forms. A conformer-margin
fine-tune improved resolution (0.848 → 0.909) without moving accuracy — exactly
what you would expect if there was little collapse to fix.

---

## 5. What actually bounds the codec

Four candidate limits were tested and three eliminated.

**Not the latent budget** — flat from 4 to 16 floats/residue.

**Not the objective** — the conformer-margin fine-tune left accuracy unmoved.

**Not model capacity.** A capacity ladder (lr-matched at 3e-4) improves
monotonically, 1.1M → 15.2M params:

| params | train | held-out all-atom | held-out bb | val/train |
|---|---|---|---|---|
| 1.1M | 0.662 | 1.085 | — | 1.64 |
| 3.8M | 0.517 | 0.938 | 0.639 | 1.81 |
| 7.0M | 0.316 | 0.910 | 0.578 | 2.88 |
| 15.2M | 0.273 | **0.873** | **0.548** | **3.20** |

Note the val/train ratio: at fixed data, bigger models memorise harder. The
15.2M model reconstructs seen data to 0.27 Å — it can already *represent* the
precision needed; it cannot *generalise* from 878 proteins.

**It is data and compute.** A 2-D grid (2 model sizes × 4 data sizes, matched
60k steps, `processed_small` val = 758) shows held-out descending with data and
the generalisation gap closing toward 1.0:

| n_train | 1.1M val | v/t | 15.2M val | v/t |
|---|---|---|---|---|
| 450 | 1.185 | 2.06 | 0.934 | 3.71 |
| 878 | 0.993 | 1.26 | 0.930 | 1.56 |
| 2000 | 0.858 | 1.07 | 0.837 | 1.31 |
| 2272 | 0.916 | 1.04 | 0.844 | 1.23 |

A dedicated compute control (resuming both models at n=2272 to 240k steps)
found **no capacity floor for either**: train RMSD reached 0.51 (1.1M) and 0.39
(15.2M) and was still descending, with 120k→240k improvements of −0.185 and
−0.073, both well past the plateau bar. Best held-out on that set is **15.2M at
240k steps: 0.752 Å all-atom / 0.476 Å backbone.**

At fixed data, extra compute mostly buys *train* fit — held-out becomes
data-limited once train converges.

### The data ceiling

The matched band — single-chain, X-ray, ≤ 2.5 Å, 20–110 residues — contains
**only ~3,853 entries in the entire PDB**. We trained on 2,272. Data is the
lever, and within this band it is nearly exhausted. Scaling further requires
broadening the size band (which needs the capacity work), predicted structures,
or moving to conformational data.

---

## 6. Errors found and corrected

Recorded because several reversed conclusions, and because the pattern —
plausible number, wrong measurement — recurred:

- **10 audit bugs** (Kabsch NaN gradients, distance-loss NaN, clash-rate
  normalisation bias, PCA data leak, resume-RNG, contact-map off-by-one, …), all
  fixed with regression tests.
- **Compression inflated ~60×** — per-residue models reported `latent_dim`
  instead of `latent_dim × n_residues`. The "227×" arms were really ~2.9×, i.e.
  they had *larger* latents than the arms they lost to.
- **PCA baseline data leak**; **val-selection bias** in the matched-task control
  (fixed with a train-carved dev split; conclusion survived).
- **RoPE frequency aliasing** — `2^k·π` broke translation invariance.
- **AMP bug**: `torch.matmul`'s autocast rule silently undid an explicit
  `.float()` inside the Kabsch SVD.
- **Train-contaminated in-domain gate** — the group census globbed train + val;
  0.574 Å was a training number. Fixed with a split filter → 0.929 Å.
- **Metric mismatch** — the ensemble gate reports all-atom RMSD; it was being
  compared against a backbone number, manufacturing a fake 0.75 → 0.93 Å
  "degradation."
- **Cohort-sampling bug** — the NMR fetcher sorted by atom count ascending,
  producing a 19-residue cohort against a 78-residue training distribution. The
  first NMR result measured domain shift, not ensembles.
- **Learning-rate artifact** — `lr=1e-3` (tuned for 1.1M) diverges at 15.2M.
  Caught only because train RMSD was reported next to held-out: a bigger model
  fitting *train* worse is an optimisation failure, never a capacity ceiling.
- **Undertraining read as a capacity floor** — at matched steps, high-n rungs
  got a fifth the epochs. A resume control refuted it. A 0.5 Å threshold would
  have mislabelled the 1.1M model "floored" precisely when it was descending
  fastest; a **plateau criterion** caught it.

Two claims in earlier versions of this report are formally **withdrawn**:
*"more data will not fix this codec"* (measured on the broken decoder) and
*"latent 2 already works"* (one lucky seed).

---

## 7. Engineering

Config-driven, no hard-coded paths. **82 tests passing** across parsing,
masking, batched alignment, every loss term, reconstruction, SE(3)-invariance
proofs, RoPE translation invariance, per-residue and per-atom pooling, flow
matching, AMP dtype safety, and the ensemble metrics. Every run saves config,
environment, resumable checkpoints, training log, metrics, and sample PDBs.
SLURM tooling packages the venv as a tarball extracted to `$SLURM_TMPDIR`,
keeps everything on `$SCRATCH`, and uses `--requeue` for preemption safety.

**Operational notes.** The cu130 venv cannot run on V100 — keep
`SBATCH_CONSTRAINT="turing|ampere|lovelace"`. `--amp` is clean everywhere.
Negative result worth keeping: the vectorised loss is ~2× *slower* than the loop
version on ragged batches; the real fix is length-bucketed batching.

---

## 8. Limitations

- **Size generality is unanswered.** The ≤3000-atom run was killed at an epoch-100
  guard: the 1.1M model cannot fit that training set at all, so the test
  conflated "bigger structures are harder" with "this model is too small." It
  needs redoing with an adequate model.
- Not equivariant by construction; invariance is empirical.
- Topology perceived from coordinates → disulfides and non-standard chemistry
  excluded from the bond set.
- The data-slope projections (~9.8k / ~370k structures for 0.5 Å) are **lower
  bounds from under-converged rungs**, so they *overestimate* the requirement.
  A converged sweep is needed for the true slope.
- Clash rate is small but non-zero (31/1000 atoms); real structures are ~0.
- **No physical claim.** All metrics are geometric.

---

## 9. Answer, and what follows

**Can the model compress protein structures while preserving atomistic geometry
better than simple baselines? Yes** — 0.75 Å backbone held-out with 0.001
chirality violations and 0.94 contact F1, against 5.07 Å for mean-shape and
11.89 Å for the centroid baseline, and beating PCA on a strictly matched task at
every aggressive budget.

**And the thing that limits it now is ordinary.** Not equivariance, not latent
size, not the objective, not capacity — those were tested and eliminated. It is
data and compute. That is a far better position than this project was in when
every architecture lost to a zero-parameter baseline.

Recommended order from here:

1. **Stage 2 (latent diffusion) can start now.** It does not depend on any open
   question above — the per-residue latent is already the right shape for a
   denoising transformer, conformers survive the round trip, and 0.75 Å is
   near-native for generating static structures.
2. **In parallel, settle the data question**, which is now the binding
   constraint and the highest-leverage open item. The matched band holds ~3,853
   structures; broadening it, or moving to conformational data, is the decision.
3. **Redo size generality** with a model large enough to fit the data, before
   any full-scale pretraining run is sized.

### One plan-level question worth deciding first

Stage 1 was scored on **novel-fold reconstruction**, which is close to a folding
problem. Latent MD actually needs **conformational-ensemble compression** — an
easier and more directly relevant target, and one this repo can now measure
(`scripts/latent_suitability.py`, `scripts/xray_ensemble_gate.py`). Given that
experimental novel-fold data is nearly exhausted in this band while
conformational data is not, it may be right to pull trajectory data earlier than
the four-stage plan assumes. That decision changes what Stage 2 pretrains on, so
it is better made now than after.

---

*Raw results for every run are committed under `outputs/cluster/` (metrics JSON,
per-structure breakdowns, training logs, environment captures, sample PDBs), with
cross-arm summaries in `comparison.md`, `capacity_ladder.md`, and
`data_scaling_2d.md`; the latent gates in `outputs/latent_suitability*.json` and
`outputs/xray_ensemble_gate.json`. Superseded analyses are preserved in git
history.*
