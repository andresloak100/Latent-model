#!/usr/bin/env bash
# One-command setup + data-scale experiment for a GPU box.
# Run FROM the experiment directory:
#     cd experiments/molecular_autoencoder_v0 && bash scripts/gpu_bootstrap.sh
#
# Env knobs (override on the command line):
#   N_STRUCTURES  how many structures to fetch/prepare   (default 1500)
#   SIZES         data-scale sweep points                (default 100,300,800,1500)
#   TOTAL_STEPS   fixed optimizer-step budget per point  (default 60000)
#   LATENT_TOKENS / LATENT_DIM  latent size              (default 16 / 16 = 256 floats)
set -euo pipefail

N_STRUCTURES="${N_STRUCTURES:-1500}"
SIZES="${SIZES:-100,300,800,1500}"
TOTAL_STEPS="${TOTAL_STEPS:-60000}"
LATENT_TOKENS="${LATENT_TOKENS:-16}"
LATENT_DIM="${LATENT_DIM:-16}"

echo "=== [1/5] Python + deps ==="
python -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null \
  || pip install torch
# torch is usually preinstalled on GPU images; only the rest is guaranteed-needed:
python - <<'PY'
import importlib, subprocess, sys
need = []
for m, pkg in [("numpy","numpy"),("scipy","scipy"),("sklearn","scikit-learn"),
               ("gemmi","gemmi"),("yaml","PyYAML"),("pytest","pytest")]:
    try: importlib.import_module(m)
    except ImportError: need.append(pkg)
if need:
    print("installing", need)
    subprocess.check_call([sys.executable,"-m","pip","install",*need])
else:
    print("all deps present")
PY

echo "=== [2/5] unit tests (must pass before training) ==="
python -m pytest tests/ -q

echo "=== [3/5] confirm GPU is visible ==="
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== [4/5] fetch ids + prepare dataset (${N_STRUCTURES} structures) ==="
python scripts/fetch_rcsb_ids.py --n "${N_STRUCTURES}" --out data/pdb_ids_scale.txt
python scripts/prepare_dataset.py --config configs/stage_b_heldout.yaml \
    --pdb-list data/pdb_ids_scale.txt --val-fraction 0.15 --jobs 16

echo "=== [5/5] data-scale sweep on GPU (the experiment) ==="
python scripts/run_data_scaling.py --base configs/stage_b_heldout.yaml \
    --sizes "${SIZES}" --total-steps "${TOTAL_STEPS}" \
    --latent-tokens "${LATENT_TOKENS}" --latent-dim "${LATENT_DIM}" \
    --device cuda --amp --num-workers 8

echo
echo "=== DONE. Key result: ==="
cat outputs/data_scan/data_scaling_table.md
echo
echo "Compare held-out all-atom RMSD across n_train against the centroid baseline."
echo "If it drops well below the baseline as n_train grows, generalisation is emerging."
