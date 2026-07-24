# Running experiments on a SLURM cluster (Mila)

Persistent tooling for the ongoing experiment programme — set up once, then
each experiment is a single command and sweeps run in parallel.

## One-time setup (login node — it has internet)

```bash
cd experiments/molecular_autoencoder_v0
bash slurm/setup_env.sh                 # venv tarball + dataset (~1100 structures)
# knobs: N_IDS=20000  FORCE_VENV=1  FORCE_DATA=1  WORKROOT=/path
```

Creates `$SCRATCH/latent-model-workspace/` containing `venv.tar.gz`, `logs/`,
and `results/`. Re-run only when dependencies or the dataset change.

## Run experiments

```bash
bash slurm/submit.sh recipe_proteinae                       # one job
bash slurm/sweep.sh arch_ab_baseline arch_ab_invariant \
                    arch_ab_rope arch_ab_perresidue         # parallel sweep
bash slurm/sweep.sh --all                                   # every config
squeue -u $USER                                             # watch
bash slurm/collect.sh                                       # comparison table
```

Per-job knobs: `TIME=6:00:00 MEM=64G GPUS=1 PARTITION=long NO_AMP=1`.

`collect.sh` writes `results/comparison.md` — held-out RMSD for every config
against the mean-shape and PCA baselines, with a verdict column. **Paste that
table back into `REPORT.md`** so the repo stays the durable record.

## Cluster conventions honoured

* **Nothing in `$HOME`** (strict quota) — caches/venv/results under `$SCRATCH`.
* **Compute nodes have no internet** — the dataset and venv are staged on the
  login node; jobs never install anything.
* **venv shipped as a tarball**, extracted to node-local `$SLURM_TMPDIR` at
  runtime (avoids shared-filesystem thrashing on many-small-files).
* Checkpoints stay on the node; only small metrics/configs/reconstructions are
  copied back.

See <https://docs.mila.quebec> for cluster specifics.

## Etiquette on a shared/borrowed account

Cancel only your own job IDs (`scancel <jobid>`) — never `scancel -u $USER` on
an account you share. Don't modify another user's `~/.claude` or dotfiles. For
sustained use, get your own cluster account rather than borrowing one.
