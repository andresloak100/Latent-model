#!/usr/bin/env bash
# One-time (idempotent) staging for a SLURM cluster — run on a LOGIN node.
#
# Builds a reusable venv tarball and a persistent dataset so that experiment
# jobs never need the network (compute nodes are usually offline). Re-run it
# only when dependencies or the dataset band change; day-to-day you just call
# submit.sh / sweep.sh.
#
#   bash slurm/setup_env.sh                 # default: ~1100 structures, ≤800 atoms
#   N_IDS=20000 bash slurm/setup_env.sh     # bigger dataset
#   FORCE_VENV=1 bash slurm/setup_env.sh    # rebuild the venv tarball
#
# Cluster conventions honoured (see docs.mila.quebec):
#   * nothing written to $HOME (strict quota) — all caches under $WORKROOT
#   * venv shipped as a tarball, extracted to node-local disk at runtime
#   * dataset prepared once on the login node (compute nodes have no internet)
set -euo pipefail

# Persistent project root. Override WORKROOT to relocate everything.
WORKROOT="${WORKROOT:-${SCRATCH:?SCRATCH is not set}/latent-model-workspace}"
N_IDS="${N_IDS:-5000}"
CONFIG="${CONFIG:-configs/recipe_proteinae.yaml}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPDIR="$(dirname "$HERE")"                       # .../molecular_autoencoder_v0

mkdir -p "$WORKROOT"
export XDG_CACHE_HOME="$WORKROOT/.cache" PIP_CACHE_DIR="$WORKROOT/.pip" \
       UV_CACHE_DIR="$WORKROOT/.uv" TORCH_HOME="$WORKROOT/.torch" \
       HF_HOME="$WORKROOT/.hf" WANDB_MODE=offline
echo "[setup] WORKROOT=$WORKROOT"

module load "$PYTHON_MODULE" 2>/dev/null || module load python 2>/dev/null || true

# --- 1. venv tarball (reusable across all jobs) ---------------------------
if [[ -f "$WORKROOT/venv.tar.gz" && -z "${FORCE_VENV:-}" ]]; then
  echo "[setup] venv.tar.gz exists — skipping (FORCE_VENV=1 to rebuild)"
else
  echo "[setup] building venv..."
  rm -rf "$WORKROOT/venv" "$WORKROOT/venv.tar.gz"
  # `python -m venv` lacks ensurepip on some clusters; prefer virtualenv/uv.
  if command -v virtualenv >/dev/null; then
    virtualenv "$WORKROOT/venv"
  elif command -v uv >/dev/null; then
    uv venv "$WORKROOT/venv"
  else
    python3 -m venv "$WORKROOT/venv"
  fi
  # shellcheck disable=SC1091
  source "$WORKROOT/venv/bin/activate"
  pip install --quiet torch numpy scipy scikit-learn gemmi pyyaml
  (cd "$WORKROOT" && tar czf venv.tar.gz venv)
  echo "[setup] venv.tar.gz built"
fi
# shellcheck disable=SC1091
source "$WORKROOT/venv/bin/activate"

# --- 2. persistent dataset (prepared once, reused by every job) -----------
DATA_DIR="$EXPDIR/data/processed"
if [[ -d "$DATA_DIR" && -n "$(ls -A "$DATA_DIR" 2>/dev/null)" && -z "${FORCE_DATA:-}" ]]; then
  echo "[setup] dataset present ($(ls "$DATA_DIR" | wc -l) structures) — skipping (FORCE_DATA=1 to rebuild)"
else
  echo "[setup] fetching + preparing dataset (n=$N_IDS ids)..."
  cd "$EXPDIR"
  python scripts/fetch_rcsb_ids.py --n "$N_IDS" --pool 8000 --out data/pdb_ids_scale.txt
  python scripts/prepare_dataset.py --config "$CONFIG" \
      --pdb-list data/pdb_ids_scale.txt --jobs 16 | tail -3
fi

echo
echo "[setup] DONE. Next:"
echo "  bash slurm/submit.sh recipe_proteinae        # one experiment"
echo "  bash slurm/sweep.sh cfgA cfgB cfgC           # parallel sweep"
