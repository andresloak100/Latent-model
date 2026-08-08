#!/bin/bash
# INBOX 61d: refuse to submit a job whose name is already in the queue.
#
# WHY THIS EXISTS. Two sessions submitted identical oracle chains thirteen minutes apart
# (10318365/66 and 10318412/13), running the same sbatch and the same script, both targeting
# oracle_channel_n300.json. That is the lost-update race 44a named for atlas_dm.json, and 52b
# established this project already has concurrent sessions acting under one identity. The duplicate
# was cancelled -- but by someone noticing, and noticing does not scale. The guard belongs at
# submission, the way check_ack.py belongs at pre-push rather than in an operator's memory.
#
#   usage:  bash scripts/armf_submit.sh <sbatch-file> [extra sbatch args...]
#           ARMF_FORCE=1 bash scripts/armf_submit.sh ...   # a deliberate second chain
set -uo pipefail
SB="${1:?usage: armf_submit.sh <sbatch-file> [args...]}"; shift || true
[ -f "$SB" ] || { echo "armf_submit: no such sbatch file: $SB" >&2; exit 2; }

NAME=$(grep -oE -- '--job-name=[^ ]+' "$SB" | head -1 | cut -d= -f2)
[ -n "${NAME:-}" ] || { echo "armf_submit: $SB has no --job-name; refusing, since a nameless job cannot be guarded" >&2; exit 2; }

EXISTING=$(squeue -u "$USER" -h -n "$NAME" -o "%i %T" 2>/dev/null)
if [ -n "$EXISTING" ] && [ -z "${ARMF_FORCE:-}" ]; then
  echo "armf_submit: REFUSING -- job name '$NAME' is already in the queue:" >&2
  echo "$EXISTING" | sed 's/^/    /' >&2
  echo "  Two chains writing one stamped results file is a lost-update race (44a)." >&2
  echo "  If this second chain is deliberate, re-run with ARMF_FORCE=1." >&2
  exit 1
fi
JID=$(sbatch --parsable "$@" "$SB") || exit $?
echo "$JID"
