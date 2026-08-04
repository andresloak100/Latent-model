#!/usr/bin/env bash
set -euo pipefail
: "${WORKROOT:?}" ; : "${EXPDIR:?}"
module load "${PYTHON_MODULE:-python/3.10}" 2>/dev/null || module load python 2>/dev/null || true
echo "=== armF slice $SLURM_JOB_ID on $(hostname) ==="; nvidia-smi || true
tar xzf "$WORKROOT/venv.tar.gz" -C "$SLURM_TMPDIR"
PY="$SLURM_TMPDIR/venv/bin/python"
cp -r "$EXPDIR" "$SLURM_TMPDIR/exp"; cd "$SLURM_TMPDIR/exp"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_HOME="$SLURM_TMPDIR/.torch"
for L in 16 64; do
  for DROP in 0 0.25 0.5; do
    echo ""; echo "################ armF L=$L drop=$DROP ################"
    $PY scripts/armf_slice.py --data-dir "$WORKROOT/armf_data" \
        --L $L --drop-rate $DROP --train-noise 0.3 --epochs 300 --local-frame 1 --device cuda \
      || echo "[armF] L=$L drop=$DROP FAILED"
  done
done
echo "=== armF slice DONE ==="
