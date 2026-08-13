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
# INBOX 107e. `scontrol update Dependency=` REPLACES the list; it does not append. That is what
# killed pretrain1m in 28 s -- re-pointing it at the new latentvideo job dropped its prep1m
# dependency, so it became eligible before its data existed. Second time an action taken TO AVOID a
# defect created one. This refuses a submission whose dependency list is SHORTER than that of a
# same-named job already queued, unless ARMF_FORCE=1 -- the same shape as the 61d guard that fired
# correctly on the 100c resubmission.
NDEP=0
for a in "$@"; do
  case "$a" in
    --dependency=*) NDEP=$(printf '%s' "$a" | sed 's/^--dependency=//' | tr ',' '\n' | grep -c .) ;;
  esac
done
PRIOR=$(printf '%s\n' "$EXISTING" | awk 'NF{print $1; exit}')
if [ -n "$PRIOR" ] && [ -z "${ARMF_FORCE:-}" ]; then
  PN=$(scontrol show job "$PRIOR" 2>/dev/null | grep -oE 'Dependency=[^ ]*' | sed 's/Dependency=//' \
       | tr ',' '\n' | grep -vc '^(null)$' 2>/dev/null || echo 0)
  if [ "$NDEP" -lt "$PN" ] 2>/dev/null; then
    echo "[armf_submit] REFUSING: job $PRIOR ($NAME) has $PN dependencies and this submission has"
    echo "  $NDEP. scontrol/sbatch REPLACE the dependency list rather than appending, and dropping one"
    echo "  is how pretrain1m became eligible before its data existed. Re-run with ARMF_FORCE=1 if"
    echo "  the reduction is deliberate."
    exit 1
  fi
fi

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
# INBOX 118e: THE CHEAP HALF OF 109e, SPLIT OUT SO IT STOPS BEING DEFERRED BEHIND THE EXPENSIVE
# HALF. Every python script this sbatch invokes must COMPILE before the job is queued. A sed
# introducing an f-string syntax error, with the submit in the same block, queued 10352325 against a
# file that would not parse -- the second near-miss of exactly this shape. py_compile catches that
# one completely. It does NOT catch NameError-class defects, which is what 109e's full CPU smoke run
# is for, and that remains owed.
PYBIN="${ARMF_PY:-$WR/venv/bin/python}"
[ -x "$PYBIN" ] || PYBIN=python3
NCHK=0
for f in $(grep -oE '[^ ]+\.py' "$SB" | sort -u); do
  ff="$f"
  case "$ff" in
    *'$'*) ff=$(eval echo "$ff" 2>/dev/null) ;;
  esac
  [ -f "$ff" ] || continue
  if ! "$PYBIN" -m py_compile "$ff" 2>&1; then
    echo "armf_submit: REFUSING to submit -- $ff does not compile" >&2
    exit 3
  fi
  NCHK=$((NCHK+1))
done
echo "armf_submit: py_compile OK on $NCHK script(s) referenced by $(basename "$SB")"

JID=$(sbatch --parsable "$@" "$SB") || exit $?
echo "$JID"
