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
# A job chained with --dependency=afterany CANNOT run concurrently with the job it waits on, so it
# is not a lost-update race. Refusing it forced ARMF_FORCE=1, which disables the check entirely --
# a guard whose only escape hatch is "turn the guard off" gets turned off for the wrong reasons too.
# INBOX 077f predicted this and it happened on the next submission: "allow when a dependency
# exists" lets TWO INDEPENDENT CHAINS both pass, which is 61d's hole reopened by its own fix. Two
# atlas_dm2 jobs were queued at once as a result. The tight predicate is that the dependency must NAME
# a job id that is itself queued under this same job name -- i.e. this submission is genuinely behind
# the thing it would otherwise duplicate, not merely behind something.
CHAINED=""
for a in "$@"; do
  case "$a" in
    --dependency=*)
      deps=$(printf '%s' "$a" | sed 's/^--dependency=//' | tr ':,?' '\n')
      for d in $deps; do
        case "$d" in
          [0-9]*) if printf '%s\n' "$EXISTING" | grep -q "^$d "; then CHAINED=1; fi ;;
        esac
      done ;;
  esac
done
if [ -n "$EXISTING" ] && [ -n "$CHAINED" ] && [ -z "${ARMF_STRICT:-}" ]; then
  echo "armf_submit: '$NAME' is queued, and this submission depends on THAT job id, so it cannot" >&2
  echo "  run concurrently. Allowing. (ARMF_STRICT=1 to refuse anyway.)" >&2
elif [ -n "$EXISTING" ] && [ -z "${ARMF_FORCE:-}" ]; then
  echo "armf_submit: REFUSING -- job name '$NAME' is already in the queue:" >&2
  echo "$EXISTING" | sed 's/^/    /' >&2
  echo "  Two chains writing one stamped results file is a lost-update race (44a)." >&2
  echo "  If this second chain is deliberate, re-run with ARMF_FORCE=1." >&2
  exit 1
fi
JID=$(sbatch --parsable "$@" "$SB") || exit $?
echo "$JID"
