#!/bin/bash
# Watch the 31d/34c tied ladder (10311656 -> 10311657 -> 10311658) for its verdict block.
#
# Lives in the REPO, not in the session and not loose in $WR: monitors die with session restarts and
# the recovery then depends on someone remembering the grep. In-repo it is versioned and reviewable
# like every other guard here. Re-arm with:
#   Monitor(command="bash experiments/molecular_autoencoder_v0/scripts/ladder_watch.sh")
#
# INBOX 36a: the verdict fires at len(pts)>=2, so two rungs (6 arms) trips it and prints a bound
# measured over 2.6x for a question posed over 6.0x. armf_tied_ladder.py now tags that PARTIAL, and
# this watch treats a PARTIAL verdict as NON-TERMINAL -- the early read is emitted because it is
# useful, but it cannot end the watch or be quoted as the fork.
#
# The tag alone is not sufficient coverage. Job 10311656 was ALREADY RUNNING when the fix landed and
# has the old code in memory, so if it reaches the verdict with two rungs it will print NO tag at all.
# The store is therefore the second, independent check: terminate only when a verdict line carries no
# PARTIAL *and* tied_ladder.json actually holds all three rungs. Either condition alone has a hole.
WR=/network/scratch/j/jacob-junqi.tian/latent-model-workspace
JOBS=${LADDER_JOBS:-10314117,10314118,10314119}
# overridable so the THREE exit branches below can be driven on fixtures -- an exit condition that
# has never been executed is the same defect this watch exists to catch.
STORE=${LADDER_STORE:-$WR/tied_ladder.json}
LOGS=${LADDER_LOGS:-$WR/logs}
rungs() {   # distinct completed, non-VOID rungs actually on disk
  STORE="$STORE" "$WR/venv/bin/python" -c '
import json, os
p = os.environ["STORE"]
r = json.load(open(p)) if os.path.exists(p) else []
print(len({x.get("n_train") for x in r if not x.get("improving")}))' 2>/dev/null || echo 0
}

# SEEN persists ACROSS re-arms. A fresh mktemp each time meant every session restart re-emitted
# every arm already reported -- three times over, which is noise that trains you to skim the very
# notifications this exists to make you read. On resume it prints ONE state line instead. Genuinely
# new arms still notify normally. Delete this file to force a full re-emit.
SEEN=${LADDER_SEEN:-$WR/.ladder_seen}
touch "$SEEN"
if [ -s "$SEEN" ]; then
  echo "resumed watch: $(wc -l < "$SEEN" | tr -d ' ') lines already reported, $(rungs)/3 rungs on disk"
fi


while true; do
  # ONLY the jobs in this chain. A bare tiedladder_*.log glob also picks up 10310946/10311078/10311546
  # -- superseded attempts that were preempted -- and their historical "CANCELLED ... SIGNAL
  # Terminated" lines read exactly like the live chain dying.
  for j in ${JOBS//,/ }; do
    f=$LOGS/tiedladder_$j.log
    [ -f "$f" ] || continue
    # verdict + partial-ladder flags + per-arm completions (<=9 total, doubles as proof-of-life)
    # + every failure signature worth acting on: silence must not be the report.
    grep -hE "PARTIAL LADDER|EARLY READ|also short of seeds|CEILING BINDS HARDER|at the ceiling,|max/min ratio|AUTHORITATIVE|PRIMARY \(fixed before results|slope [+-][0-9.]+ \+/-|THE CEILING MOVES WITH DATA|NO EFFECT LARGER THAN|OPTION \(1\) IS NOT RETIRED|^ +tied n[0-9]+ .*: FVE|VOID|FAIL [A-Za-z]*Error|Traceback|CUDA out of memory|DUE TO TIME LIMIT|CANCELLED" "$f" 2>/dev/null \
    | while IFS= read -r line; do
        grep -qxF "$line" "$SEEN" 2>/dev/null && continue
        printf '%s\n' "$line" >> "$SEEN"
        printf '[%s] %s\n' "$(basename "$f" .log | sed 's/tiedladder_//')" "$line"
      done
  done

  NR=$(rungs)
  # A verdict line that does NOT carry the PARTIAL tag.
  if grep -E "THE CEILING MOVES WITH DATA|NO EFFECT LARGER THAN" "$SEEN" 2>/dev/null \
     | grep -qv "PARTIAL LADDER"; then
    if [ "${NR:-0}" -ge 3 ]; then
      echo "LADDER VERDICT COMPLETE -- untagged verdict AND all 3 rungs on disk. This is the fork."
      # The job that printed it may hold PRE-GUARD code: 10311656 started before 36a/37a/38b/39b
      # landed, so ITS verdict block has no ceiling flag, no asymmetry reading and no pessimistic-SD
      # sensitivity. Reporting that as the result would report the unguarded number. Replay the store
      # through the CURRENT code and emit that instead -- which is what REPORT_ONLY was added for.
      echo "--- replaying the store through the CURRENT (guarded) report code ---"
      LADDER_REPORT_ONLY=1 "$WR/venv/bin/python" \
        "$(dirname "$0")/armf_tied_ladder.py" 2>/dev/null \
        | sed -n '/STEP BUDGET BY RUNG/,/^  SCOPE/p' | grep -v '^$'
      exit 0
    fi
    echo "EARLY READ ONLY: an untagged verdict printed but the store holds ${NR}/3 rungs -- that is the"
    echo "pre-36a code in job 10311656, which cannot tag itself. NOT terminal; still watching for n300."
  fi

  if ! squeue -j "$JOBS" -h -o "%T" 2>/dev/null | grep -qE "RUNNING|PENDING"; then
    echo "LADDER CHAIN ENDED with ${NR}/3 rungs and no complete verdict -- preemption, time limit, or"
    echo "all n300 arms VOID. Check \$WR/logs/tiedladder_*.log; replay the store with"
      echo "--- replaying the store through the CURRENT (guarded) report code ---"
    LADDER_REPORT_ONLY=1 "$WR/venv/bin/python" "$(dirname "$0")/armf_tied_ladder.py" 2>/dev/null \
      | sed -n '/STEP BUDGET BY RUNG/,/^  SCOPE/p' | grep -v '^$' 
    exit 0
  fi
  sleep 120
done
