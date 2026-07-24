#!/usr/bin/env bash
# Copy finished experiment results from the cluster workspace into the git repo
# and (optionally) push them.  Run on a login node:
#
#   bash slurm/push_results.sh              # stage + commit + try to push
#   NO_PUSH=1 bash slurm/push_results.sh    # stage + commit only
#
# Only small artefacts are committed — metrics, tables, configs, logs and a few
# example reconstructions. Trained checkpoints (*.pt) stay on $SCRATCH; they are
# gitignored and far too large for the repo.
set -euo pipefail

WORKROOT="${WORKROOT:-${SCRATCH:?SCRATCH is not set}/latent-model-workspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPDIR="$(dirname "$HERE")"
SRC="$WORKROOT/results"
DEST="$EXPDIR/outputs/cluster"

[[ -d "$SRC" ]] || { echo "no results at $SRC — run a sweep first"; exit 1; }
mkdir -p "$DEST"

n=0
for d in "$SRC"/*/; do
  cfg="$(basename "$d")"
  [[ -f "$d/metrics.json" ]] || continue
  mkdir -p "$DEST/$cfg"
  for f in metrics.json metrics_table.md config.yaml train_log.json environment.json; do
    [[ -f "$d/$f" ]] && cp "$d/$f" "$DEST/$cfg/"
  done
  # a couple of example structures only, not the whole set
  if [[ -d "$d/reconstructions" ]]; then
    mkdir -p "$DEST/$cfg/reconstructions"
    find "$d/reconstructions" -name '*.pdb' | head -6 | xargs -r -I{} cp {} "$DEST/$cfg/reconstructions/"
  fi
  n=$((n+1))
done
[[ -f "$SRC/comparison.md" ]] && cp "$SRC/comparison.md" "$DEST/comparison.md"
echo "[push] staged $n result set(s) -> $DEST"

cd "$EXPDIR"
git add -A "$DEST" >/dev/null 2>&1 || true
if git diff --cached --quiet; then
  echo "[push] nothing new to commit"; exit 0
fi
git commit -q -m "Cluster results: $(ls "$SRC" | tr '\n' ' ' | sed 's/ $//')" || true
echo "[push] committed."

if [[ -n "${NO_PUSH:-}" ]]; then
  echo "[push] NO_PUSH set — skipping push."; exit 0
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if git push origin "$BRANCH" 2>/dev/null; then
  echo "[push] pushed to $BRANCH"
else
  cat <<'EOF'
[push] push failed (no credentials configured on this machine).
       Either configure a fine-grained token for this repo only:
         git remote set-url origin https://<TOKEN>@github.com/<owner>/<repo>.git
       (stored inside the clone under $SCRATCH, not in $HOME)
       ...or just report the contents of results/comparison.md instead.
EOF
fi
