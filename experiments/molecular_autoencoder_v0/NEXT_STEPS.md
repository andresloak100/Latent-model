# Next steps — decision memo (post data-scale sweep)

The held-out experiments established: the baseline (raw-coordinate) Perceiver
autoencoder **fits training structures well (~1–2 Å) but does not generalise**
— on unseen folds it loses to PCA and even to the mean-shape baseline, and the
held-out error is flat across latent budgets (32→512 floats). A converged,
size-controlled data-scale sweep is running to decide *why*. Two branches:

```
                 converged, size-controlled n=800 held-out
                 /                                        \
   still ABOVE mean-shape                          DROPS below mean-shape,
   (data doesn't rescue it)                        toward PCA as n grows
        |                                                   |
  ARCHITECTURE-LIMITED                              DATA-LIMITED
  → invariant/equivariant encoder                  → scale up (Mila),
    (prototype ready, below)                          full sweep in parallel
```

## Branch A — architecture-limited (the expected outcome): invariant encoder

**Already built and CPU-verified in this repo** (so the moment the verdict
lands, we test, not design):

- `molae/model_equivariant.py` — `InvariantEncoder` / `InvariantAutoencoder`.
  The latent is a function of only SE(3) invariants (sorted k-NN distances +
  centroid distance + atom identity), so it is rotation/translation invariant
  **by construction** — no alignment crutch. Verified exactly in
  `tests/test_equivariance.py` (latent unchanged under random rigid transforms;
  still overfits a structure to <2 Å).
- Wired into the pipeline via `model.encoder_type: "invariant"`. Run the A/B
  directly:
  ```bash
  # baseline vs invariant at the same data scale / budget
  python scripts/train.py --config configs/stage_b_heldout.yaml   # baseline
  python scripts/train.py --config configs/arch_invariant.yaml    # invariant
  # (or point run_data_scaling at each and compare the held-out curves)
  ```
  The hypothesis: the invariant encoder generalises with **far fewer**
  structures (its held-out RMSD should fall below mean-shape / toward PCA at a
  data scale where the baseline is still flat).

**Follow-ups if the minimal invariant encoder helps but isn't enough:**
- Keep *directional* information equivariantly (GVP vector channels, or IPA-style
  local frames) instead of only scalar distances.
- Make the **decoder** equivariant too (predict local frames / relative
  positions) rather than absolute coords in a canonical frame.

## Branch B — data-limited: scale up on Mila

If more data *does* rescue generalisation, the full sweep (data × latent ×
model size) belongs on the Mila cluster as parallel SLURM jobs. Bridge work
needed (none of it needs a GPU, can be done anytime):

1. **Static dataset** — commit a curated ~3k single-chain PDB-ID list and a
   `prepare-once` step that stages cleaned `.npz` into `/network/scratch`
   (Mila compute nodes have **no internet**, so the RCSB fetch must happen on a
   login node first).
2. **SLURM array script** — one job per (n_train, latent, encoder_type) cell,
   reading the pre-staged data, writing metrics back to scratch.
3. Collect the parallel results into one scaling table.

## Regardless of branch

- The **decoder-conditioning** choice (decoder is given atom identity; only
  geometry is compressed) is a deliberate design decision — confirm it matches
  the intended latent contents before Step 2 (latent diffusion).
- Do **not** start latent diffusion until the autoencoder clears the PCA bar on
  held-out folds. Diffusion in a latent the decoder can't faithfully invert
  won't help.
- Keep reporting held-out against the **full baseline set** (mean-shape + PCA
  k2/4/8), not just centroid — mean-shape is the line that separates "real
  generalisation" from "mean-regression."
