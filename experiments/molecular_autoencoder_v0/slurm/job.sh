#!/usr/bin/env bash
# The actual SLURM job body. Not called directly — submit.sh / sweep.sh set
# CFG_NAME, WORKROOT, EXPDIR and submit this script.
#
# Runs entirely on node-local disk ($SLURM_TMPDIR): the venv tarball and a copy
# of the code are extracted there, so the shared filesystem is never hit in a
# hot loop and nothing is written to $HOME.
set -euo pipefail

: "${CFG_NAME:?}" ; : "${WORKROOT:?}" ; : "${EXPDIR:?}"
module load "${PYTHON_MODULE:-python/3.10}" 2>/dev/null || module load python 2>/dev/null || true

echo "=== job $SLURM_JOB_ID : $CFG_NAME on $(hostname) ==="
nvidia-smi || true

# node-local working copy
tar xzf "$WORKROOT/venv.tar.gz" -C "$SLURM_TMPDIR"
PY="$SLURM_TMPDIR/venv/bin/python"
cp -r "$EXPDIR" "$SLURM_TMPDIR/exp"
cd "$SLURM_TMPDIR/exp"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_HOME="$SLURM_TMPDIR/.torch" \
       HF_HOME="$SLURM_TMPDIR/.hf" WANDB_MODE=offline

AMP_FLAG="--amp"
[[ -n "${NO_AMP:-}" ]] && AMP_FLAG=""

$PY scripts/train.py --config "configs/$CFG_NAME.yaml" --device cuda $AMP_FLAG --num-workers "${SLURM_CPUS_PER_TASK:-8}"
$PY scripts/eval.py  --config "configs/$CFG_NAME.yaml" --device cuda --n-examples 3

# copy results back to persistent storage (checkpoints stay on the node)
OUT_SUB="$($PY -c "
import sys, yaml
print(yaml.safe_load(open('configs/$CFG_NAME.yaml'))['train']['out_dir'])
")"
DEST="$WORKROOT/results/$CFG_NAME"
mkdir -p "$DEST"
cp -r "$OUT_SUB"/{metrics.json,metrics_table.md,config.yaml,train_log.json,environment.json} "$DEST/" 2>/dev/null || true
cp -r "$OUT_SUB/reconstructions" "$DEST/" 2>/dev/null || true

echo "================ RESULT: $CFG_NAME ================"
$PY - <<'PY'
import json, os, sys
p = os.environ.get("RESULT_METRICS", "")
import glob
cands = glob.glob("outputs/**/metrics.json", recursive=True)
if not cands:
    print("no metrics.json produced"); sys.exit(0)
m = json.load(open(sorted(cands, key=os.path.getmtime)[-1]))
lm = m["learned_model"]
print("held-out backbone RMSD : %.3f A" % lm["backbone_rmsd"])
print("held-out all-atom RMSD : %.3f A" % lm["all_atom_rmsd"])
print("held-out contact F1    : %.3f"   % lm["contact_f1"])
coh = m.get("cohort", {}).get("baselines", {})
if coh:
    print("baselines (backbone)   :", {k: round(v["mean_backbone_rmsd"], 3) for k, v in coh.items()})
    ms = coh.get("mean_shape", {}).get("mean_backbone_rmsd")
    pca = coh.get("pca_k8", {}).get("mean_backbone_rmsd")
    if ms and pca:
        v = lm["backbone_rmsd"]
        verdict = ("BEATS PCA k8" if v < pca else
                   "beats mean-shape, below PCA" if v < ms else
                   "LOSES to mean-shape (no generalisation)")
        print("verdict                :", verdict)
PY
echo "=================================================="
echo "results -> $WORKROOT/results/$CFG_NAME"
