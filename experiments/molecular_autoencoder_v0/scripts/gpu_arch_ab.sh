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

echo "=== [4/5] three encoder sweeps on the SAME data (baseline / invariant / rope) ==="
for arm in baseline invariant rope; do
  echo "--- $arm ---"
  python scripts/run_data_scaling.py --base "configs/arch_ab_${arm}.yaml" --tag "_${arm}" \
      --sizes "${SIZES}" --total-steps "${TOTAL_STEPS}" \
      --latent-tokens 16 --latent-dim 8 --device cuda --amp --num-workers 8
done

echo "=== [5/5] three-way head-to-head ==="
python - <<'PY'
import json, os
arms=["baseline","invariant","rope"]
def load(tag):
    p=f"outputs/data_scan_{tag}/data_scaling.json"
    return {r["n_train"]: r for r in json.load(open(p))["rows"]} if os.path.exists(p) else {}
data={a:load(a) for a in arms}
def bars(n):  # mean-shape / PCA bars from whichever arm has them (same cohort)
    for a in arms:
        f=f"outputs/data_scan_{a}/n{n}/metrics.json"
        if os.path.exists(f):
            c=json.load(open(f)).get("cohort",{}).get("baselines",{})
            return (c.get("mean_shape",{}).get("mean_backbone_rmsd",float('nan')),
                    c.get("pca_k8",{}).get("mean_backbone_rmsd",float('nan')))
    return (float('nan'),float('nan'))
ns=sorted({n for a in arms for n in data[a]})
print(f"{'n_train':>8} | {'baseline':>9} | {'invariant':>9} | {'rope':>9} | {'mean-shape':>10} | {'PCA k8':>7}")
print("-"*66)
for n in ns:
    ms,pca=bars(n)
    def g(a): return data[a].get(n,{}).get("heldout_backbone_rmsd",float('nan'))
    print(f"{n:>8} | {g('baseline'):>9.2f} | {g('invariant'):>9.2f} | {g('rope'):>9.2f} | {ms:>10.2f} | {pca:>7.2f}")
print("\nWin condition: an encoder's held-out drops below mean-shape and toward PCA k8,")
print("where the baseline stayed flat/above mean-shape. That encoder is the fix.")
PY
echo "Commit outputs/data_scan_baseline|invariant|rope/ and push."
