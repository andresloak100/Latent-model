# Latent-model

Research toward a **latent molecular-dynamics model**: compress molecular
structure into a compact latent space, (eventually) run diffusion there, and
decode back — the approach modern text-to-video models use, applied to
proteins / molecular dynamics.

Work proceeds in isolated, reviewable milestones under `experiments/`.

## Milestones

- [`experiments/molecular_autoencoder_v0/`](experiments/molecular_autoencoder_v0/) —
  **Milestone 1 (current): the molecular structure autoencoder.** Compresses
  protein atomic coordinates into a small latent and reconstructs them, with
  geometry-aware losses/metrics and classical baselines. This is the
  encoder/decoder that a later latent-diffusion stage will sit inside.
  **No diffusion / trajectories / RL / generation yet** — that is deliberately
  out of scope until this is validated. See its `README.md`, `PLAN.md`, and
  `REPORT.md`.

> Note: this repository is the persistent home for the project (the "model
> dynamics" work). In the cloud execution sandbox there is no local
> `Documents` folder, so everything is committed here to the feature branch —
> that is what survives and can be reviewed.
