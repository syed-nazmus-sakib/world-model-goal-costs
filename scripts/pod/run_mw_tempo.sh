#!/bin/bash
# TEMPO baseline on the 8 Metaworld tasks. Waits for the head queue's data and heads, trains the temporal map on the
# same pilot data (expert frames + goals), then after HEAD_DONE runs the closed loop with w = 0.5 (TEMPO) and w = 1 (pure map).
# usage: bash scripts/pod/run_mw_tempo.sh [out=/root/out/TM] [head_queue_out=/root/out/H]   (after run_mw_rtc_head.sh, same pod)
source $(dirname $0)/jwm_env.sh
O=${1:-/root/out/TM}; mkdir -p $O; H=${2:-/root/out/H}; C=$JEPAWM_HOME/configs
log() { echo "$(date -u +%F' '%T) $*" >> $O/queue.log; }
until grep -q -E "heads rc=|ABORT" $H/queue.log 2>/dev/null; do sleep 30; done
TASKS="button-press-topdown push drawer-open plate-slide faucet-open window-open drawer-close reach-wall"
TR=""; for t in $TASKS; do TR="$TR $H/data/$t/train"; done
log "tempo map train start"
timeout 1800 python -m evals.failure_attribution.mw_tempo_train --dirs $TR --out $O/tempo_map.pt > $O/train.log 2>&1; log "tempo map rc=$? $(tail -1 $O/train.log | cut -c1-200)"
[ -f $O/tempo_map.pt ] || { log "ABORT no map"; log "TEMPO_DONE"; exit 0; }
until grep -q HEAD_DONE $H/queue.log 2>/dev/null; do sleep 60; done
closed() {  # task w
  local t=$1 w=$2 d=$O/closed_$1_w$2; mkdir -p $d; log "closed $t w=$w start"
  timeout 7200 python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_$t.yaml --out_dir $d --episodes 12 \
    --planners sim_latent --num_workers 5 --cost tempo:$O/tempo_map.pt:$w >> $d.log 2>&1; rc=$?
  python -m evals.failure_attribution.summarize_closed --run_dir $d > $d/summary.txt 2>&1
  log "closed $t w=$w rc=$rc $(grep -E '^  sim_' $d/summary.txt | sed -E 's/ +\| success at any replan.*//' | tr -s ' ')"
}
( for t in button-press-topdown drawer-open faucet-open drawer-close; do closed $t 0.5; done ) &
( for t in push plate-slide window-open reach-wall; do closed $t 0.5; done ) &
wait
( closed button-press-topdown 1.0; closed drawer-open 1.0 ) & ( closed push 1.0; closed plate-slide 1.0 ) & wait
log "TEMPO_DONE"
