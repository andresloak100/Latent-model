#!/usr/bin/env bash
# Architecture A/B on a GPU: does the SE(3)-invariant encoder generalise where
# the baseline (raw-coordinate) encoder did not? Both are trained on the SAME
# prepared dataset at the same 128-float budget and ≤800-atom band, matching the
# data-scale sweep. Run FROM the experiment directory:
#     cd experiments/molecular_autoencoder_v0 && bash scripts/gpu_arch_ab.sh
#
# Env knobs: N_STRUCTURES (default 1500), SIZES (default 100,300,800,1129),
#            TOTAL_STEPS (default 80000).
set -euo pipefail

N_STRUCTURES="${N_STRUCTURES:-1500}"
SIZES="${SIZES:-100,300,800,1129}"
TOTAL_STEPS="${TOTAL_STEPS:-80000}"

echo "=== [1/5] deps ==="
python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))" 2>/dev/null || {
  python -c "import torch" 2>/dev/null || pip install torch
  python -c "import torch; assert torch.cuda.is_available(), 'NO GPU VISIBLE'"
}
python - <<'PY'
import importlib, subprocess, sys
need=[p for m,p in [("numpy","numpy"),("scipy","scipy"),("sklearn","scikit-learn"),
                    ("gemmi","gemmi"),("yaml","PyYAML"),("pytest","pytest")]
      if not importlib.util.find_spec(m)]
if need: subprocess.check_call([sys.executable,"-m","pip","install",*need])
PY

echo "=== [2/5] unit tests (incl. invariance proofs) ==="
python -m pytest tests/ -q

echo "=== [3/5] fetch + prepare ≤800-atom dataset ONCE (shared by both arms) ==="
python scripts/fetch_rcsb_ids.py --n "${N_STRUCTURES}" --out data/pdb_ids_scale.txt
python scripts/prepare_dataset.py --config configs/arch_ab_baseline.yaml \
    --pdb-list data/pdb_ids_scale.txt --val-fraction 0.15 --jobs 16

echo "=== [4/5] baseline encoder sweep ==="
python scripts/run_data_scaling.py --base configs/arch_ab_baseline.yaml --tag _baseline \
    --sizes "${SIZES}" --total-steps "${TOTAL_STEPS}" \
    --latent-tokens 16 --latent-dim 8 --device cuda --amp --num-workers 8

echo "=== [4b/5] invariant encoder sweep (same dataset/splits) ==="
python scripts/run_data_scaling.py --base configs/arch_ab_invariant.yaml --tag _invariant \
    --sizes "${SIZES}" --total-steps "${TOTAL_STEPS}" \
    --latent-tokens 16 --latent-dim 8 --device cuda --amp --num-workers 8

echo "=== [5/5] head-to-head ==="
python - <<'PY'
import json, glob, os
def load(tag):
    p=f"outputs/data_scan_{tag}/data_scaling.json"
    return {r["n_train"]: r for r in json.load(open(p))["rows"]} if os.path.exists(p) else {}
b, i = load("baseline"), load("invariant")
# mean-shape / PCA bars from the invariant run's per-n metrics (same cohort)
def bars(tag, n):
    f=f"outputs/data_scan_{tag}/n{n}/metrics.json"
    if not os.path.exists(f): return (float('nan'),float('nan'))
    c=json.load(open(f)).get("cohort",{}).get("baselines",{})
    return (c.get("mean_shape",{}).get("mean_backbone_rmsd",float('nan')),
            c.get("pca_k8",{}).get("mean_backbone_rmsd",float('nan')))
print(f"{'n_train':>8} | {'baseline AE':>12} | {'invariant AE':>13} | {'mean-shape':>10} | {'PCA k8':>7}")
print("-"*62)
for n in sorted(set(b)|set(i)):
    ms,pca=bars("invariant",n)
    bb=b.get(n,{}).get("heldout_backbone_rmsd",float('nan'))
    iv=i.get(n,{}).get("heldout_backbone_rmsd",float('nan'))
    print(f"{n:>8} | {bb:>12.2f} | {iv:>13.2f} | {ms:>10.2f} | {pca:>7.2f}")
print("\nWin condition: invariant AE drops below mean-shape and toward PCA k8,")
print("where the baseline AE stayed flat/above mean-shape.")
PY
echo "Commit outputs/data_scan_baseline/ and outputs/data_scan_invariant/ and push."
