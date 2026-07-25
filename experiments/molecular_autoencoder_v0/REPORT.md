# Report — Molecular Structure Autoencoder (Milestone 1)

**Scientific question:** *Can a learned, geometry-aware autoencoder compress
protein atomic coordinates into a compact latent while preserving atomistic
geometry better than simple baselines?*

**Short answer: yes — once the decoder is fixed.**

On 293 held-out folds the final codec reconstructs full all-atom structures to
**0.75 Å backbone / 1.02 Å all-atom RMSD** with **near-perfect stereochemistry**
(chirality violation rate 0.001, bond-length error 0.16 Å, contact-map F1 0.94)
at **2.9× compression**. Every earlier architecture in this study — including
three of the four designs suggested in review — landed at 9.5–11.2 Å, i.e. no
better than predicting the centroid, and **lost to a 0-parameter mean-shape
baseline**. The difference is a single architectural change in the *decoder*,
with data, losses, optimiser, epochs, split and latent budget held constant.

This reverses the previous version of this report, which concluded "no". That
conclusion was correct about the models it had; it was wrong to generalise it
to neural compression of protein structure. Two controls (below) located the
fault in the decoder rather than in the data, the latent budget, or the
equivariance story.

This milestone is the **autoencoder only**. No diffusion, trajectories, force
fields, RL, or generation. All metrics here are **geometric** — nothing in this
report claims physical or energetic accuracy.

---

## 1. The result

### 1.1 What changed: direct per-residue readout

Every failing arm shares one decoder pattern: the decoder holds a set of latent
tokens, and each atom must **find its own coordinates by attending** to them
(per-atom identity query → cross-attention → xyz). Atom *i*'s output is a
content-addressed lookup.

The fix (`molae/model_direct.py`, `encoder_type: direct`) removes the lookup.
Each residue owns a latent vector; residue tokens self-attend; a head emits a
fixed bank of `N_ATOM_NAMES × 3` coordinate slots per residue, and each atom
simply **gathers the slot that belongs to it**:

```python
slots = self.head(h).view(B, R, self.n_slots, 3) * self.cfg.coord_scale
flat  = slots.view(B, R * self.n_slots, 3)
gather_idx = (res_pos * self.n_slots + batch["atom_name_idx"]).clamp(max=R * self.n_slots - 1)
return torch.gather(flat, 1, gather_idx.unsqueeze(-1).expand(-1, -1, 3))
```

Output slot *i* **is** atom *i*, by construction — the same property a flat MLP
autoencoder has, and the property the ablation ladder (§3.2) identified as the
one that matters. Routing is no longer something the model has to learn.

### 1.2 Full grid, 293 held-out folds

All 14 runs below share the same prepared dataset (878 train / 293 val, ≤ 800
atoms), same losses, 900 epochs, batch 16, lr 1e-3, seed 0, random SO(3) input
augmentation. `bb(full)` is full-length backbone RMSD over the whole held-out
structure; `bb(cohort)` is restricted to the 20-residue / 80-atom cohort that
the PCA and mean-shape baselines are computed on (see §4.1 — only `bb(cohort)`
is comparable to those two columns).

| run | latent (floats/res) | compression | bb(cohort) Å | bb(full) Å | all-atom Å | bond err Å | chirality | clash/1k | contact F1 |
|---|---|---|---|---|---|---|---|---|---|
| **direct d16** | 16 | 1.5× | **0.681** | **0.740** | 1.004 | 0.157 | **0.001** | 31 | 0.937 |
| **direct d8** | 8 | 2.9× | 0.693 | 0.754 | 1.020 | 0.161 | **0.001** | 31 | **0.943** |
| **direct d4** | 4 | 5.8× | 0.707 | 0.750 | **1.000** | 0.170 | **0.001** | 40 | 0.938 |
| **direct d2** | 2 | 11.7× | 1.405 | 1.531 | 1.977 | 0.254 | 0.002 | 130 | 0.826 |
| direct d1 | 1 | 23.3× | 4.069 | 6.236 | 6.737 | 0.657 | 0.117 | 970 | 0.440 |
| recipe_proteinae (perresidue + flow) | 8 | 2.9×† | 4.826 | 5.013 | 6.201 | 3.595 | 0.498 | 238 | 0.438 |
| direct d8 + flowmatch | 8 | 2.9× | 4.957 | 5.798 | 7.046 | 4.412 | 0.496 | 213 | 0.386 |
| arch_ab_perresidue | 8 | 2.9×† | 6.181 | 9.728 | 10.277 | 0.884 | 0.402 | 625 | 0.296 |
| perresidue d8 | 8 | 2.9× | 6.187 | 9.498 | 10.034 | 0.889 | 0.352 | 781 | 0.292 |
| perresidue d4 | 4 | 5.8× | 6.420 | 9.942 | 10.439 | 1.002 | 0.337 | 827 | 0.283 |
| arch_ab_invariant (SE(3)-invariant enc.) | 128 total | 14.2× | 6.931 | 10.170 | 10.663 | 1.095 | 0.366 | 752 | 0.267 |
| arch_ab_baseline (raw-coord Perceiver) | 128 total | 14.2× | 7.137 | 10.473 | 10.967 | 1.185 | 0.259 | 792 | 0.262 |
| arch_ab_rope (LLM-style pos. enc.) | 128 total | 14.2× | 8.132 | 11.157 | 11.600 | 1.541 | 0.122 | 1246 | 0.236 |
| direct d4 + flowmatch | 4 | 5.8× | 11.614 | 11.865 | 12.496 | 9.584 | 0.496 | 150 | 0.179 |
| *mean-shape baseline (0 floats)* | 0 | ∞ | *5.067* | — | — | — | — | — | — |
| *PCA k=8 (cohort only)* | 8 total | 30× | *1.987* | — | — | — | — | — | — |
| *centroid / Rg baseline* | 0 | ∞ | — | — | *11.888* | — | — | — | — |

† `arch_ab_perresidue` and `recipe_proteinae` originally reported **227.7×**
compression. That figure was wrong: the per-residue model returned `latent_dim`
(8) as its total latent size instead of `latent_dim × n_residues`. Their true
compression is ~2.9×, identical to `perresidue d8` — i.e. these arms had a
*larger* latent than the 128-float arms they were being compared against, and
still lost. The bug is fixed (`latent_floats_for(n_residues)`); the corrected
value is what the `grid_*` runs report.

**Reading the table:**

- The four `direct` reconstruct arms are in a different regime from everything
  else: **an order of magnitude better RMSD and two orders of magnitude better
  chirality**, at latent budgets that are *smaller*, not larger.
- **Latent 2 already works** (1.53 Å at 11.7× compression); **latent 4
  saturates** — d4, d8 and d16 are within 0.02 Å of each other on `bb(full)`, so
  the codec is not capacity-limited above ~4 floats/residue. Latent 1 breaks
  (6.2 Å).
- **Flow matching sabotages it.** Identical encoder, identical latent, swapping
  the deterministic decoder for a rectified-flow decoder takes `direct d8` from
  0.75 → 5.80 Å and `direct d4` from 0.75 → **11.87 Å**, with chirality going to
  0.496 — a coin flip. A generative decoder trained this way collapses toward the
  conditional mean; it is not a drop-in replacement for reconstruction, and it
  should not be adopted before the diffusion stage without its own study.
- **RMSD alone is misleading here, and the physics columns prove it.**
  `recipe_proteinae` has the 6th-best RMSD in the grid but **0.498 chirality**
  (indistinguishable from random handedness) and **3.60 Å bond errors** — it
  "wins" on RMSD by emitting an average blob. Conversely `arch_ab_rope` has the
  *worst* RMSD (8.13) but the best chirality of the failing arms (0.122). Only
  the `direct` arms are good on *both* axes simultaneously, which is what
  distinguishes genuine reconstruction from mean-regression.

### 1.3 The winner is uniformly good, not good on average

`grid_direct_d8_reco`, per-structure over all 293 held-out folds
(26–113 residues, up to 800 atoms; mean 78 residues / 607 atoms):

| statistic | backbone RMSD (Å) |
|---|---|
| median | 0.771 |
| p90 | 0.923 |
| **worst structure** | **1.701** |

There is no tail of catastrophic failures — the worst held-out fold in the set
is still sub-2 Å. Train RMSD at the end of training was 0.52 Å against 0.75 Å
held-out: a **generalisation gap of ~0.2 Å**. Compare the earlier converged
baseline arm, which sat at 3.07 Å train / 8.64 Å held-out. The failure mode
that dominated this whole study — memorise the training set, emit a blob for
anything unseen — is gone.

---

## 2. Setup

- **Data.** Experimentally-determined structures from RCSB, cleaned with gemmi:
  waters / ligands / ions / hydrogens / altlocs removed, 20 standard amino acids
  only, single peptide chain, topology perceived from coordinates by covalent
  radii (Cordero 2008, tolerance 0.45 Å). Every filtering decision is recorded
  in `data/manifest.json`. Size band **≤ 800 heavy atoms, 20–200 residues**, so
  difficulty is held constant across arms. **878 train / 293 val**, split by
  sequence-similarity clustering so near-duplicates cannot leak.
- **Symmetry.** None of the winning models is equivariant by construction.
  Invariance is handled by (a) centring inputs, (b) **random SO(3) input
  augmentation** during training, and (c) **Kabsch superposition inside every
  coordinate loss and every reported metric** — differentiable, masked, batched,
  with the rotation detached (envelope theorem) and autocast disabled inside the
  SVD. One arm (`arch_ab_invariant`) *is* provably SE(3)-invariant by
  construction (sorted k-NN distances + centroid distance), verified in
  `tests/test_equivariance.py`. It did not help (§4.3).
- **Compute.** Mila SLURM cluster, `long` partition, preemptible, one GPU per
  arm, PyTorch 2.13.0+cu130. ~5100 s wall-clock for 900 epochs × 878 structures
  for the winning arm; the 14-arm grid ran in parallel. Earlier stages ran on
  CPU (Stage A: 1054 s for 1200 epochs on 21 structures, ~7.5 ms/structure
  inference, ~479 MB peak RSS) — so the answer to "do we need a GPU for the
  first try?" was **no for Stage A, yes for everything after it**.

---

## 3. The two controls that located the fault

The previous version of this report concluded "architecture-limited, more data
will not help", based on a data-scale sweep that flattened out and then got
*worse* (n=800 trained to convergence: held-out 7.96 → 8.64 Å). That ruled out
data and latent budget but did not say *which part* of the architecture. Two
controls answered that.

### 3.1 Matched-task control — neural compression does work

Take PCA's exact task away from PCA: fixed 80-atom aligned backbone cohorts,
identical inputs, identical splits, identical latent budgets, ~1.3 M-parameter
flat MLP autoencoder. Checkpoints selected on a **dev split carved out of
train** (15%) — never on val. 870 train / 286 val.

| latent floats | PCA (val Å) | AE (val Å) | AE (train Å) | winner |
|---|---|---|---|---|
| 2 | 3.525 | **2.863** | 1.945 | **AE** (−0.66) |
| 4 | 2.717 | **2.058** | 0.192 | **AE** (−0.66) |
| 8 | 1.987 | **1.676** | 0.176 | **AE** (−0.31) |
| 16 | **1.423** | 1.615 | 0.204 | PCA (+0.19) |
| *0 (mean shape)* | *5.067* | — | — | — |

A neural autoencoder beats PCA at every aggressive budget, and only loses once
the budget is loose enough that the linear subspace is nearly sufficient. This
is the expected shape of the result and it **contradicted** the whole-pipeline
finding — which meant the whole pipeline, not neural compression, was broken.

**Noise check:** repeated over 5 seeds, the AE won 5/5 at k=2/4/8 and PCA won
5/5 at k=16, with run-to-run **std ≈ 0.05 Å** — every verdict is far outside
noise. *(Summary retained; the per-seed JSON was not committed.)*

### 3.2 Ablation ladder — the decoder is the culprit

Same cohort data, same budgets, same epochs; the *only* difference is the
decoder. Rung 1 is the flat MLP (output slot *i* is atom *i*). Rung 2 replaces
it with the attention decoder from the full pipeline (atom *i* must attend to
find its own coordinates), with identity made deliberately uninformative so
both rungs receive the same information.

| latent floats | PCA | rung 1 (flat MLP) | rung 2 (attention decoder) |
|---|---|---|---|
| 4 | 2.717 | **2.124** | 2.382 |
| 8 | 1.987 | **1.673** | 1.908 |
| 16 | **1.423** | 1.613 | 1.811 |

The attention decoder is **worse at every budget** on identical data. The
penalty is small here (0.2–0.3 Å) because the cohort task is easy and
fixed-size, but the *direction* is unambiguous, and the prediction was that on
the full variable-size all-atom task the same penalty becomes catastrophic.
§1.2 confirms it: 9.5 Å with the attention decoder, 0.75 Å with direct readout.

---

## 4. Honest caveats

### 4.1 The "beats PCA" claim is not budget-matched

`outputs/cluster/comparison.md` labels the direct arms "BEATS PCA", comparing
`bb(cohort)` against PCA k=8. That is a fair comparison of **accuracy on the
same atoms**, but not of **compression**:

- PCA k=8 spends **8 floats total** on the 80-atom cohort (30× compression) and
  requires a fixed-size, pre-aligned input. It cannot represent a variable-size
  all-atom protein at all.
- `direct d8` spends **8 floats per residue** on the *whole* all-atom structure
  (2.9× compression) and reconstructs the cohort atoms to 0.69 Å as a by-product.

PCA is doing an easier task at a much higher compression ratio. At the closest
matched compression (`direct d1`, 23.3× vs PCA's 30×) the neural codec gets
4.07 Å on the cohort and **loses** to PCA's 1.99 Å. The clean apples-to-apples
statement is the matched-task control in §3.1, not this row of the grid.

**The defensible claim is therefore:** the direct codec achieves near-native
all-atom geometry on variable-size held-out proteins at modest (3–12×)
compression, which no previous arm could do at any budget; and on a strictly
matched task a neural autoencoder beats PCA at aggressive budgets. **It is not
yet demonstrated at PCA-like compression ratios (≥ 30×).**

### 4.2 Compression is length-proportional, not constant

The direct decoder's latent is *d* floats per residue, so it is a per-residue
"latent point cloud", not a fixed-size bottleneck. This is the right shape for
the eventual latent-diffusion stage (a sequence of latent tokens is exactly what
a denoising transformer wants), but it means compression does not improve with
protein size the way a fixed-size Perceiver latent does. The fixed-size arms did
have that property — and did not work.

### 4.3 What did *not* help

Three of the four architectural directions raised in review were tested on the
same leak-free 293-fold held-out set and **none of them fixed the problem**:

- **SE(3)-invariant graph encoder** (k-NN distance featurisation): 10.17 Å vs
  the raw-coordinate baseline's 10.47 Å. A 0.3 Å improvement on an 11 Å failure.
- **LLM-style RoPE / Fourier coordinate position encoding:** 11.16 Å — the
  *worst* arm on RMSD, though the best of the failing arms on chirality.
  (An initial `2^k·π` frequency schedule aliased and broke translation
  invariance; it was fixed to log-spaced physical wavelengths 0.7–64 Å before
  this run, so the result is not an artefact of that bug.)
- **ProteinAE-style recipe** (per-residue latents + flow-matching decoder):
  5.01 Å with coin-flip chirality (0.498) and 3.60 Å bond errors — mean-collapse.
- **O(N·L) attention** was used throughout and is retained; it is a cost
  property, not an accuracy fix.

Equivariance was the leading hypothesis and it was wrong. The fault was decoder
routing. This is worth recording, because a larger run would have spent a great
deal of compute failing to fix the wrong thing.

### 4.4 Remaining limitations

- Established **within the ≤ 800-atom / 20–200-residue band only**. Whether the
  direct decoder holds on larger proteins is untested and is the first thing to
  check (§6).
- Not equivariant by construction; invariance comes from augmentation +
  Kabsch-in-loss. It works, but it is an empirical property of these runs.
- Topology is perceived from coordinates, so disulfides and non-standard
  chemistry are excluded from the bond set. Bond-length error is measured over
  perceived covalent bonds only.
- Single seed for the grid (seed 0). The matched-task control was seed-repeated;
  the 14-arm grid was not. Given the ~9 Å gap between the direct arms and
  everything else, seed variance cannot explain the headline result, but the
  *ordering within* the direct arms (d16 vs d8 vs d4, spread 0.02 Å) is not
  resolved.
- Clash rate for `direct d8` is 31 per 1000 atoms — small but not zero. Real
  structures have ~0. This is the weakest of the physics metrics.
- **No physical claim.** RMSD, bond length, chirality and contact F1 are
  geometric. Nothing here establishes that a decoded structure is energetically
  reasonable, let alone dynamically meaningful.

---

## 5. Engineering / reproducibility

- Config-driven (`configs/*.yaml`); no hard-coded paths in scripts.
- **64+ unit tests, all passing**, covering parsing, masking, batched alignment,
  each loss term, reconstruction, the synthetic test molecule, SE(3)-invariance
  proofs, RoPE translation invariance, per-residue pooling, flow matching,
  the vectorized loss path, and AMP dtype safety.
- Every run saves config, environment (package versions + git commit),
  resumable checkpoints, a training log, metrics JSON/Markdown, and sample
  reconstructed PDBs (`outputs/cluster/*/reconstructions/`).
- SLURM tooling in `slurm/`: venv packaged as a tarball extracted to
  `$SLURM_TMPDIR` at runtime (avoids transient BeeGFS issues on compute nodes),
  everything on `$SCRATCH`, `--requeue` + append logging for preemption safety,
  checkpoints written to shared storage so preempted jobs resume.
- **Bugs found and fixed during this study** (each with a regression test): a
  clash-rate normalisation bias for >2000-atom proteins; NaN gradients in the
  differentiable Kabsch on degenerate covariance; a NaN in the distance loss; a
  PCA-baseline data leak; a resume-RNG bug; a contact-map off-by-one; a
  chain-recording gap; the RoPE frequency aliasing; the per-residue compression
  mis-report (§1.2); a metric mismatch in `collect.sh` (full-length backbone
  RMSD compared against cohort-based baselines); a val-selection bias in the
  matched-task control (fixed with a train-carved dev split — the conclusion
  survived); and an AMP bug where `torch.matmul`'s autocast rule silently undid
  an explicit `.float()` cast inside the Kabsch SVD.
- **Negative result worth recording:** the vectorized loss implementation is
  ~2× *slower* than the loop version on ragged real batches (padding waste
  dominates). It is kept opt-in (`vectorized=False` default). The real fix is
  length-bucketed batching, not vectorization.

### Operational notes for future runs

- The `cu130` venv **cannot run on V100 nodes** ("no kernel image"). Keep
  `SBATCH_CONSTRAINT="turing|ampere|lovelace"` in `submit.sh` until the venv is
  rebuilt with a wider arch list.
- `--amp` is now clean everywhere (the autocast/Kabsch bug is fixed and covered
  by `tests/test_amp_safety.py`); it can be enabled by default.
- Shared-account etiquette on the borrowed cluster: `scancel` only our own job
  IDs, never `scancel -u $USER`; git identity set with `git config --local` in
  our clone only. *(A `--global` config was set in error and has been unset —
  the account owner needs to re-set their own `user.name` / `user.email`.)*

---

## 6. Answer to the headline question, and the gate

> *Can the model compress protein structures while preserving atomistic geometry
> better than simple baselines?*

**Yes.** On 293 held-out folds the direct per-residue readout codec reaches
0.75 Å backbone RMSD with 0.001 chirality violations, 0.16 Å bond error and 0.94
contact F1, versus 5.07 Å for the mean-shape baseline and 11.89 Å for the
centroid baseline. On a strictly matched task a neural autoencoder also beats
PCA at every aggressive latent budget, 5/5 seeds. The qualifier from §4.1
stands: this is demonstrated at 3–12× compression, not yet at PCA-like ratios.

**The milestone instruction was to stop here and report before implementing
latent diffusion. Stopping.** Recommended order for what comes next:

1. **Confirm the direct decoder holds beyond the ≤ 800-atom band** before
   anything else. Everything above is established inside one size band, and the
   whole point of the architecture is variable size. This is cheap and it is the
   only thing standing between here and Step 2.
2. **A seed repeat of the direct arms** (3 seeds, d2/d4/d8) to resolve the
   ordering, plus a clash-rate follow-up.
3. **Then Step 2, latent diffusion** on the per-residue latent — which is
   already the right shape for a denoising transformer.

### One plan-level question to settle first

The success criterion used here is **reconstruction of novel folds**, which is
close to a folding problem. What latent MD actually needs is **compression of
conformational ensembles** — many conformers of the *same* system — which is a
substantially easier and more directly relevant target. It is worth deciding
explicitly whether Step 1's gate should be re-stated in those terms before
committing compute to the trajectory stage. This changes what dataset Step 2
should be pretrained on, so it is better answered now than later.

---

*Raw results for every run in this report are committed under
`outputs/cluster/` (metrics JSON, per-structure breakdowns, training logs,
environment captures, sample reconstructed PDBs), with the cross-arm summary in
`outputs/cluster/comparison.md`, the matched-task control in
`outputs/cluster/matched_task_pca.json`, and the ablation ladder in
`outputs/cluster/ablation_ladder.json`. Earlier stages are under
`outputs/stage_a/`, `outputs/scan/` and `outputs/data_scan/`; the superseded
Stage A/B narrative and the data-scale sweep that motivated §3 are preserved in
git history.*
