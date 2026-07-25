#!/usr/bin/env bash
# Summarise every finished experiment in one comparison table.
# Run on the login node after a sweep:  bash slurm/collect.sh
set -euo pipefail

WORKROOT="${WORKROOT:-${SCRATCH:?SCRATCH is not set}/latent-model-workspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPDIR="$(dirname "$HERE")"
# shellcheck disable=SC1091
source "$WORKROOT/venv/bin/activate" 2>/dev/null || true

python - "$WORKROOT" <<'PY'
import json, os, sys, glob
root = sys.argv[1]
rows = []
for mpath in sorted(glob.glob(os.path.join(root, "results", "*", "metrics.json"))):
    name = os.path.basename(os.path.dirname(mpath))
    m = json.load(open(mpath))
    lm = m["learned_model"]
    coh = m.get("cohort", {}).get("baselines", {})
    # IMPORTANT: compare like with like. learned_model.backbone_rmsd is measured
    # over the FULL-length backbone, whereas mean_shape/pca_* are measured on the
    # fixed 20-residue / 80-atom cohort. Use the model's score on that SAME cohort
    # (cohort.learned_autoencoder) for the head-to-head, and keep the full-length
    # number in a separate column.
    ae_cohort = m.get("cohort", {}).get("learned_autoencoder", {})
    rows.append({
        "config": name,
        "bb": ae_cohort.get("mean_backbone_rmsd", lm.get("backbone_rmsd", float("nan"))),
        "bb_full": lm.get("backbone_rmsd", float("nan")),
        "aa": lm.get("all_atom_rmsd", float("nan")),
        "f1": lm.get("contact_f1", float("nan")),
        "mean_shape": coh.get("mean_shape", {}).get("mean_backbone_rmsd", float("nan")),
        "pca_k8": coh.get("pca_k8", {}).get("mean_backbone_rmsd", float("nan")),
        "n": m.get("n_structures", "?"),
    })
if not rows:
    print("no finished results under", os.path.join(root, "results")); sys.exit(0)

hdr = f"{'config':<26} {'bb(cohort)':>10} {'bb(full)':>9} {'contF1':>7} {'mean-shape':>11} {'PCA k8':>7}  verdict"
print(hdr); print("-" * len(hdr))
lines = ["Matched cohort: bb(cohort), mean-shape and PCA are all the 20-residue/80-atom cohort.",
         "bb(full) is the full-length backbone (NOT comparable to the baselines).", "",
         "| config | bb RMSD (cohort) | bb RMSD (full) | contact F1 | mean-shape | PCA k8 | verdict |",
         "|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda x: x["bb"]):
    v = ("BEATS PCA" if r["bb"] < r["pca_k8"] else
         "beats mean-shape" if r["bb"] < r["mean_shape"] else
         "loses to mean-shape")
    print(f"{r['config']:<26} {r['bb']:>10.3f} {r['bb_full']:>9.3f} {r['f1']:>7.3f} "
          f"{r['mean_shape']:>11.3f} {r['pca_k8']:>7.3f}  {v}")
    lines.append(f"| {r['config']} | {r['bb']:.3f} | {r['bb_full']:.3f} | {r['f1']:.3f} | "
                 f"{r['mean_shape']:.3f} | {r['pca_k8']:.3f} | {v} |")
out = os.path.join(root, "results", "comparison.md")
open(out, "w").write("\n".join(lines) + "\n")
print("\nwrote", out, "— paste this table back into the repo/report")
PY
