#!/usr/bin/env bash
# Submit MANY experiments in parallel — one SLURM job per config.
# This is the main reason to be on a cluster: a sweep that would take days
# sequentially finishes in roughly the time of its slowest single run.
#
#   bash slurm/sweep.sh arch_ab_baseline arch_ab_invariant arch_ab_rope arch_ab_perresidue
#   bash slurm/sweep.sh --all            # every configs/*.yaml
#   TIME=6:00:00 bash slurm/sweep.sh recipe_proteinae arch_ab_rope
#
# Collect results afterwards with:  bash slurm/collect.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPDIR="$(dirname "$HERE")"

if [[ "${1:-}" == "--all" ]]; then
  mapfile -t CONFIGS < <(cd "$EXPDIR/configs" && ls *.yaml | sed 's/\.yaml$//' | grep -v '^_')
else
  CONFIGS=("$@")
fi
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "usage: sweep.sh <config> [config ...] | --all"; exit 1; }

echo "[sweep] submitting ${#CONFIGS[@]} jobs: ${CONFIGS[*]}"
for cfg in "${CONFIGS[@]}"; do
  bash "$HERE/submit.sh" "$cfg"
done
echo
echo "[sweep] watch:    squeue -u \$USER"
echo "[sweep] collect:  bash slurm/collect.sh"
