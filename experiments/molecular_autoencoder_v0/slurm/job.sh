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

# node-local working copy: venv + code + dataset are many small files, so they
# live on fast local disk. Outputs do NOT (see below).
tar xzf "$WORKROOT/venv.tar.gz" -C "$SLURM_TMPDIR"
PY="$SLURM_TMPDIR/venv/bin/python"
cp -r "$EXPDIR" "$SLURM_TMPDIR/exp"
cd "$SLURM_TMPDIR/exp"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_HOME="$SLURM_TMPDIR/.torch" \
       HF_HOME="$SLURM_TMPDIR/.hf" WANDB_MODE=offline

# --- preemption safety --------------------------------------------------
# This partition is preemptible: the job can be killed at any moment and
# requeued, and $SLURM_TMPDIR is WIPED when that happens. So checkpoints must
# live on the shared filesystem, not on the node. We point the run's out_dir at
# an absolute path under $WORKROOT (train.py joins with ROOT, and pathlib keeps
# an absolute right-hand side), so `latest.pt` survives preemption and the
# resume logic picks up exactly where it left off (optimizer + RNG included).
RUN_DIR="$WORKROOT/results/$CFG_NAME/run"
mkdir -p "$RUN_DIR"
JOB_CFG="$SLURM_TMPDIR/exp/configs/_job_$CFG_NAME.yaml"
$PY - "$CFG_NAME" "$RUN_DIR" "$JOB_CFG" <<'PY'
import sys, yaml
cfg_name, run_dir, out_path = sys.argv[1:4]
cfg = yaml.safe_load(open(f"configs/{cfg_name}.yaml"))
cfg["train"]["out_dir"] = run_dir          # absolute -> shared filesystem
yaml.safe_dump(cfg, open(out_path, "w"), sort_keys=False)
print(f"[job] checkpoints -> {run_dir} (preemption-safe)")
PY
if [[ -f "$RUN_DIR/latest.pt" ]]; then
  echo "[job] found existing checkpoint — resuming after preemption/requeue"
fi

AMP_FLAG="--amp"
[[ -n "${NO_AMP:-}" ]] && AMP_FLAG=""

$PY scripts/train.py --config "$JOB_CFG" --device cuda $AMP_FLAG --num-workers "${SLURM_CPUS_PER_TASK:-8}"
$PY scripts/eval.py  --config "$JOB_CFG" --device cuda --n-examples 3

# Results already live on shared storage; drop the big checkpoints, keep the
# small artefacts.
DEST="$WORKROOT/results/$CFG_NAME"
mkdir -p "$DEST"
cp "$RUN_DIR"/{metrics.json,metrics_table.md,config.yaml,train_log.json,environment.json} "$DEST/" 2>/dev/null || true
cp -r "$RUN_DIR/reconstructions" "$DEST/" 2>/dev/null || true
rm -f "$RUN_DIR"/latest.pt "$RUN_DIR"/final.pt    # free space once finished

echo "================ RESULT: $CFG_NAME ================"
$PY - "$DEST/metrics.json" <<'PY'
import json, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    print("no metrics.json produced"); sys.exit(0)
m = json.load(open(path))
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
