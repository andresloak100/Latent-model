#!/usr/bin/env bash
# Submit ONE experiment to SLURM.  Usage (from the experiment dir):
#
#   bash slurm/submit.sh recipe_proteinae            # configs/recipe_proteinae.yaml
#   TIME=6:00:00 MEM=64G bash slurm/submit.sh arch_ab_invariant
#   NO_AMP=1 bash slurm/submit.sh recipe_proteinae   # if AMP produces NaNs
#
# Assumes `bash slurm/setup_env.sh` has been run once (venv tarball + dataset).
# Results are copied back to $WORKROOT/results/<config>/ and echoed in the log.
set -euo pipefail

CFG_NAME="${1:?usage: submit.sh <config-name-without-.yaml>}"
WORKROOT="${WORKROOT:-${SCRATCH:?SCRATCH is not set}/latent-model-workspace}"
PARTITION="${PARTITION:-long}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
TIME="${TIME:-3:00:00}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPDIR="$(dirname "$HERE")"
[[ -f "$EXPDIR/configs/$CFG_NAME.yaml" ]] || { echo "no such config: configs/$CFG_NAME.yaml"; exit 1; }
[[ -f "$WORKROOT/venv.tar.gz" ]] || { echo "run slurm/setup_env.sh first (no venv.tar.gz)"; exit 1; }

mkdir -p "$WORKROOT/logs" "$WORKROOT/results"

sbatch --job-name="mae_$CFG_NAME" \
       --partition="$PARTITION" --gres=gpu:"$GPUS" --cpus-per-task="$CPUS" \
       --mem="$MEM" --time="$TIME" \
       --output="$WORKROOT/logs/%x_%j.out" \
       --export=ALL,CFG_NAME="$CFG_NAME",WORKROOT="$WORKROOT",EXPDIR="$EXPDIR",PYTHON_MODULE="$PYTHON_MODULE",NO_AMP="${NO_AMP:-}" \
       "$HERE/job.sh"
