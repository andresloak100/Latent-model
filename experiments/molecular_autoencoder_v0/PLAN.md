# Implementation plan, environment survey, and assumptions

## 1. Environment / repository survey (done before writing code)

- **Repository:** `andresloak100/latent-model` was cloned **empty** — a fresh
  git repo with **no commits and no files**. There is therefore **no existing
  "production Loak" code, no molecular libraries, no datasets, and no
  utilities** to reuse or avoid modifying. This experiment is a clean-slate,
  isolated directory (`experiments/molecular_autoencoder_v0/`).
- **Python:** 3.11 (system interpreter).
- **Pre-installed relevant packages:** only `PyYAML`. Everything else
  (`torch`, `numpy`, `scipy`, `scikit-learn`, `gemmi`) was installed into the
  environment; versions are captured per-run in `outputs/<run>/environment.json`.
- **GPU:** none (`nvidia-smi` absent, `torch.cuda.is_available() == False`).
  See "Do you need a GPU?" below.
- **Network:** RCSB downloads and pip both work through the agent proxy.
- **Compute:** 4 CPU cores, ~15 GB RAM, ~30 GB free disk.

## 2. Do you need a GPU for the first try?

**No.** Stage A (overfit 10–50 small proteins) and the small Stage B/C runs
train in minutes on CPU (~0.5 s/epoch for ~21 proteins, ≈1.5 M params). A GPU
only becomes worthwhile once the dataset grows to thousands of structures or
the model scales up (later Stage B/C). The code is device-agnostic
(`--device cuda` when one is available); nothing here requires it.

## 3. Build order (what was implemented)

1. Chemical constants + vocabularies (`constants.py`).
2. Parsing/cleaning with an auditable filtering record (`parsing.py`).
3. Differentiable + numpy Kabsch alignment (`alignment.py`).
4. Metrics — all rotation/translation invariant (`metrics.py`).
5. Losses — coord/distance/bond/clash/chirality (`losses.py`).
6. Perceiver-style modular AE (`model.py`).
7. Dataset + masked collate (`dataset.py`).
8. Classical baselines (`baselines.py`).
9. Synthetic known-geometry molecule + unit tests.
10. Config, utils (seed/env/checkpoint), scripts (prepare/train/eval/scaling).
11. Stage A overfit → Stage B held-out → Stage C latent sweep.

## 4. Major assumptions (and how each is handled)

| # | Assumption | Handling |
|---|---|---|
| A1 | Global rotation/translation must not be penalised | centre inputs + Kabsch-align in all coord losses/metrics; architecture is explicitly *non*-equivariant (documented) |
| A2 | The decoder may know atom identity/sequence; only geometry is compressed | decoder queries are identity-only, never coordinates; the bottleneck is the sole path for geometry |
| A3 | Topology can be perceived from coordinates | covalent-radius bond perception restricted to intra/consecutive residues; disulfides excluded (documented) |
| A4 | Only single-chain, standard-AA, heavy-atom protein structures | one peptide chain selected; waters/ligands/ions/H/altlocs removed; non-standard residues dropped; all recorded in `manifest.json` |
| A5 | A fixed-size latent independent of atom count is the right compression target | latent is `L × latent_dim`; compression ratio grows with protein size |
| A6 | PCA is the fair linear baseline | only valid on a fixed-size cohort; comparison is caveated (see README), trivial all-atom baselines added for the AE's native task |
| A7 | Small scale is acceptable for a first prototype | ~21 proteins; explicitly labelled illustrative; no bulk/auto download; ESM Atlas not used |

## 5. Known limitations / next steps (not in this milestone)

- Not equivariant → try an SE(3)/frame-based encoder.
- Perceived topology misses disulfides / non-standard chemistry.
- Tiny dataset → real Stage B/C need thousands of structures and a proper
  structural-similarity split (e.g. TM-score / Foldseek clustering).
- Controlled matched-task AE-vs-PCA comparison (same 80-atom backbone target).
- Only then: the latent **diffusion** stage this autoencoder is meant to host.
