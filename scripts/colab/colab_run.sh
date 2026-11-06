#!/bin/bash
# Laptop side: run one job chain on a fresh Colab GPU session and bring its results back.
#   create session -> upload code bundle, setup.sh, head.sh, the chain and extra files -> run setup -> start the chain
#   detached on the VM -> every 10 min pull /content/out into artifacts/colab_runs/out/x_<sess>/x -> when the chain logs
#   GA_CHAIN_DONE (or after CAPH hours) pull once more and stop the session.
# Robustness (lessons of 2026-11-01 and 11-05):
#   * a lost network only pauses polling; the session is never stopped because a check failed
#   * killing this script does NOT stop the session (no EXIT trap); re-attach with ATTACH=1 and the same arguments
#   * every path is absolute, so a remount of the external drive does not break later writes
#   * launch it through scripts/colab/launch.sh, which detaches it with screen (survives the Claude Code process)
# usage: [GPU=L4] [HIGHMEM=1] [ATTACH=1] bash scripts/colab/colab_run.sh <sess jwm*> <chain.sh> "<local:remote ...>" <cap hours>
set -u
PROJ=/Volumes/ns_external/Shared/jepa-wms-project; C=$PROJ/scripts/colab
SESS=$1; CHAIN=$2; FILES=${3:-}; CAPH=${4:-6}
case $SESS in jwm*) ;; *) echo "session names must start with jwm"; exit 1 ;; esac
OUT=$PROJ/artifacts/colab_runs/out/x_$SESS; LOGD=$PROJ/artifacts/colab_runs/logs; mkdir -p "$OUT" "$LOGD"
LOG=$LOGD/x_$SESS.log
log() { echo "$(date -u +%F' '%T) $*" >> "$LOG"; }
SSHO=(-o "ProxyCommand=colab ssh --proxy-mode -s $SESS" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
      -o LogLevel=ERROR -o ServerAliveInterval=30 -o ConnectTimeout=60 -i "$HOME/.ssh/id_ed25519")
net() { curl -s -m 15 -o /dev/null https://colab.research.google.com; }
alive() { colab sessions 2>/dev/null | grep -F "[$SESS]" | grep -q "Variant: GPU"; }   # never auto-creates a CPU runtime
r() { alive || return 3; timeout "${T:-300}" ssh "${SSHO[@]}" "root@colab-$SESS" "$1"; }
put() { for i in 1 2 3 4; do alive || return 3; timeout 1800 scp "${SSHO[@]}" "$1" "root@colab-$SESS:$2" 2>> "$LOG" && return 0
        sleep 20; done; log "upload failed: $1"; return 1; }
pull() {
  T=600 r "cd /content && tar czf /content/res.tgz --exclude='*.npz' --exclude='*.mp4' --exclude='*.pt' out 2>/dev/null; echo ok" \
      > /dev/null || return 1
  timeout 900 scp "${SSHO[@]}" "root@colab-$SESS:/content/res.tgz" "$OUT/res.tgz.tmp" 2>> "$LOG" || return 1
  mv "$OUT/res.tgz.tmp" "$OUT/res.tgz" && mkdir -p "$OUT/x" && tar xzf "$OUT/res.tgz" -C "$OUT/x" --strip-components=1 2>/dev/null
  log "pulled: $(tail -1 "$OUT/x/chain.log" 2>/dev/null | cut -c1-200)"
}
finish() { pull || pull; colab stop -s "$SESS" >> "$LOG" 2>&1; log "session $SESS stop requested ($1); $(colab usage 2>/dev/null | head -1)"; exit 0; }

log "orchestrator start ($SESS, $CHAIN, cap ${CAPH}h, attach=${ATTACH:-0})"
T0=$(date +%s); CAP=$((CAPH * 3600))
if [ "${ATTACH:-0}" != 1 ]; then
  B=$OUT/code.tgz
  ( cd "$PROJ" && COPYFILE_DISABLE=1 tar czf "$B" --exclude='._*' --exclude='.git' --exclude='__pycache__' --exclude='outputs' \
      code/third_party/stable-worldmodel code/lewm_explorations/paa scripts/pod scripts/hpc env/conda.sh env/paths.sh ) \
      || { log "bundle failed"; exit 1; }
  colab new -s "$SESS" --gpu "${GPU:-L4}" ${HIGHMEM:+--high-mem} >> "$LOG" 2>&1 || { log "could not create the session"; exit 1; }
  for i in $(seq 1 30); do alive && break; sleep 10; done
  alive || { log "session never became active on a GPU"; exit 1; }
  put "$B" /content/ga_code.tgz || exit 1
  for f in "$C/setup.sh" "$C/head.sh"; do put "$f" "/content/$(basename "$f")" || exit 1; done
  put "$CHAIN" /content/chain.sh || exit 1
  for f in $FILES; do r "mkdir -p $(dirname "${f#*:}")" > /dev/null; put "${f%%:*}" "${f#*:}" || exit 1; done
  out=$(T=2400 r 'bash /content/setup.sh 2>&1 | tail -6'); log "setup: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  echo "$out" | grep -q GA_SETUP_DONE || { log "setup failed"; finish "setup failed"; }
  r 'cd /content && mkdir -p out && (nohup bash /content/chain.sh > /content/chain.nohup 2>&1 < /dev/null &) ; sleep 3; echo started' \
      > /dev/null && log "chain started"
fi
while true; do
  sleep 600
  [ $(( $(date +%s) - T0 )) -ge $CAP ] && finish "cap ${CAPH}h"
  if ! net; then log "network down, waiting"; continue; fi
  if ! alive; then log "session $SESS no longer active on a GPU; stopping the watcher (nothing to stop)"; exit 1; fi
  pull || { log "pull failed, retry in 10 min"; continue; }
  grep -q "GA_CHAIN_DONE" "$OUT/x/chain.log" 2>/dev/null && finish "chain done"
done
