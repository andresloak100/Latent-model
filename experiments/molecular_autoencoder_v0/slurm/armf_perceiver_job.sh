#!/usr/bin/env bash
set -euo pipefail
: "${WORKROOT:?}" ; : "${EXPDIR:?}"
module load "${PYTHON_MODULE:-python/3.10}" 2>/dev/null || module load python 2>/dev/null || true
echo "=== perceiver-seg $SLURM_JOB_ID on $(hostname) ==="; nvidia-smi || true
tar xzf "$WORKROOT/venv.tar.gz" -C "$SLURM_TMPDIR"
PY="$SLURM_TMPDIR/venv/bin/python"
cp -r "$EXPDIR" "$SLURM_TMPDIR/exp"; cd "$SLURM_TMPDIR/exp"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_HOME="$SLURM_TMPDIR/.torch"
$PY -c "import torch; print('cuda avail', torch.cuda.is_available())"
$PY scripts/armf_perceiver_segment.py --epochs 400 --batch 64 || echo "[perceiver-seg] FAILED"
echo "=== perceiver-seg DONE ==="
