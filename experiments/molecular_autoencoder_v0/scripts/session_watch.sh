#!/bin/bash
# SELF-WAKE. Replaces the ladder-specific watch. Fires on any of four things and NEVER terminates on
# a match -- more than one will fire, and a watch that exits on the first is a watch that misses the
# rest. Re-arm with:
#   Monitor(command="bash experiments/molecular_autoencoder_v0/scripts/session_watch.sh")
#
#   1. NEW INBOX      -- fetches the branch and compares last_acted in ACK.md against the highest
#                        "## NNN" in INBOX.md, both read from the REMOTE, so it sees a push before
#                        this session has pulled it.
#   2. LADDER VERDICT -- PRIMARY / CEILING MOVES / NO EFFECT LARGER / PARTIAL.
#   3. JOB DEPARTURE  -- any job leaving the queue, with its final sacct State, so a TIMEOUT or
#                        FAILED wakes the same as a COMPLETED. Silence on failure is the failure
#                        mode this whole apparatus exists to prevent (21c/27a/27b went unread).
#   4. LOG FAILURES   -- Traceback / CUDA OOM / FAIL / DUE TO TIME LIMIT in any recently-live log.
#
# De-duplicates against a persistent file so a re-arm reports state in one line instead of replaying.
set -uo pipefail
WR=/network/scratch/j/jacob-junqi.tian/latent-model-workspace
REPO=/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model
EXP=$REPO/experiments/molecular_autoencoder_v0
BRANCH=${WATCH_BRANCH:-claude/latent-diffusion-md-model-8rlhv1}
LOGS=${WATCH_LOGS:-$WR/logs}
SEEN=${WATCH_SEEN:-$WR/.session_seen}
JOBSTATE=${WATCH_JOBSTATE:-$WR/.session_jobs}
INTERVAL=${WATCH_INTERVAL:-120}
touch "$SEEN"

emit() {  # emit once, ever
  grep -qxF "$1" "$SEEN" 2>/dev/null && return 0
  printf '%s\n' "$1" >> "$SEEN"; printf '%s\n' "$1"
}

# Baseline the queue on first run so pre-existing jobs are not reported as departures.
if [ ! -f "$JOBSTATE" ]; then
  squeue -u "$USER" -h -o "%i %j" 2>/dev/null | sort > "$JOBSTATE" || : > "$JOBSTATE"
fi
if [ -s "$SEEN" ]; then
  echo "resumed self-wake: $(wc -l < "$SEEN" | tr -d ' ') lines already reported, $(wc -l < "$JOBSTATE" | tr -d ' ') jobs tracked"
fi

while true; do
  # ---- 1. NEW INBOX, read from the remote so a push is seen before this session pulls ----
  if git -C "$REPO" fetch -q origin "$BRANCH" 2>/dev/null; then
    hi=$(git -C "$REPO" show "FETCH_HEAD:experiments/molecular_autoencoder_v0/COMMS/INBOX.md" 2>/dev/null \
         | grep -oE '^## [0-9]{3}' | grep -oE '[0-9]{3}' | sort -n | tail -1)
    la=$(git -C "$REPO" show "FETCH_HEAD:experiments/molecular_autoencoder_v0/COMMS/ACK.md" 2>/dev/null \
         | grep -oE '^last_acted:[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | tail -1)
    if [ -n "${hi:-}" ] && [ -n "${la:-}" ] && [ "$((10#$hi))" -ne "$((10#$la))" ]; then
      emit "NEW INBOX: item $hi is unACKed (ACK.md last_acted $la) -- pull and ACK per PROTOCOL."
    fi
  fi

  # ---- 2/4. verdict lines and failure signatures, in logs touched in the last 12 h ----
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    while IFS= read -r line; do
      emit "[$(basename "$f" .log)] $line"
    done < <(grep -hE "PRIMARY \(fixed before results|THE CEILING MOVES WITH DATA|NO EFFECT LARGER THAN|PARTIAL LADDER|CEILING BINDS|Traceback \(most recent|CUDA out of memory|FAIL [A-Za-z]*Error|DUE TO TIME LIMIT|CANCELLED AT" "$f" 2>/dev/null | tail -40)
  done < <(find "$LOGS" -maxdepth 1 -name '*.log' -mmin -720 2>/dev/null)

  # ---- 3. JOB DEPARTURES, with the final state ----
  now=$(mktemp); squeue -u "$USER" -h -o "%i %j" 2>/dev/null | sort > "$now"
  while read -r jid jname; do
    [ -n "${jid:-}" ] || continue
    if ! grep -q "^$jid " "$now"; then
      st=$(sacct -j "$jid" --format=State%20,Elapsed,ExitCode -n 2>/dev/null | head -1 | tr -s ' ')
      emit "JOB LEFT QUEUE: $jid ($jname) ->${st:- state unavailable}"
    fi
  done < "$JOBSTATE"
  mv -f "$now" "$JOBSTATE"

  sleep "$INTERVAL"
done
