#!/bin/bash
# Approach-aware learned cost head (rtc) across 8 Metaworld tasks.
#   1. pilot data with candidate futures: train = episodes 100-123, test = episodes 0-11 (the closed-loop episodes), 8 tasks
#   2. one rtc head and one state head (ablation) on all 8 tasks' training data (waits for /root/HEADS_READY: the trainer upload)
#   3. offline tables per task; 4. closed loop, 12 episodes, --cost head:<rtc head>, 8 tasks, two lanes; then the state head on
#   the two tasks where rtc wins most (ablation). Writes HEAD_DONE in the queue log at the end.
# Rule (stated before the runs): rtc PASSES if >= 6/12 on >= 4 of the 7 object tasks, with reach-wall and drawer-close >= 10/12.
# usage: bash scripts/pod/run_mw_rtc_head.sh [out=/root/out/H]      (JEPA-WM venv + Metaworld data; check_egl.sh first)
source $(dirname $0)/jwm_env.sh
O=${1:-/root/out/H}; mkdir -p $O; C=$JEPAWM_HOME/configs
log() { echo "$(date -u +%F' '%T) $*" >> $O/queue.log; }
TASKS="button-press-topdown push drawer-open plate-slide faucet-open window-open drawer-close reach-wall"
log "head queue start"
data() { local t=$1 split=$2 st=$3 n=$4; mkdir -p $O/data/$t/$split
  timeout 5400 python -m evals.failure_attribution.mw_pilot_data --fname $C/mw_jepa-wm_$t.yaml --out_dir $O/data/$t/$split \
    --ep_start $st --episodes $n --num_workers 6 >> $O/data/$t/$split.log 2>&1; log "data $t $split rc=$? eps=$(ls $O/data/$t/$split/ep_*.npz 2>/dev/null | wc -l)"; }
( for t in button-press-topdown drawer-open faucet-open drawer-close; do data $t train 100 24; data $t test 0 12; done ) &
( for t in push plate-slide window-open reach-wall; do data $t train 100 24; data $t test 0 12; done ) &
wait

TR=""; TE=""; TN=""; for t in $TASKS; do TR="$TR $O/data/$t/train"; TE="$TE $O/data/$t/test"; TN="$TN mw-$t"; done
timeout 5400 python -m evals.failure_attribution.mw_pilot_heads --train_dir $TR --test_dir $TE --task $TN --out_dir $O/heads --modes rtc,state \
  --steps 12000 > $O/heads.log 2>&1; log "heads rc=$? $(grep -E '^(rtc|state) ' $O/heads.log | cut -c1-120 | tr '\n' ' ')"
[ -f $O/heads/head_rtc.pt ] || { log "ABORT no rtc head"; log "HEAD_DONE"; exit 0; }
closed() {  # task head
  local t=$1 h=$2 d=$O/closed_$1_$2; mkdir -p $d; log "closed $t $h start"
  timeout 7200 python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_$t.yaml --out_dir $d --episodes 12 \
    --planners sim_latent --num_workers 5 --cost head:$O/heads/head_$h.pt >> $d.log 2>&1; rc=$?
  python -m evals.failure_attribution.summarize_closed --run_dir $d > $d/summary.txt 2>&1
  log "closed $t $h rc=$rc $(grep -E '^  sim_' $d/summary.txt | sed -E 's/ +\| success at any replan.*//' | tr -s ' ')"
}
( for t in button-press-topdown drawer-open faucet-open drawer-close; do closed $t rtc; done ) &
( for t in push plate-slide window-open reach-wall; do closed $t rtc; done ) &
wait
( closed button-press-topdown state ) & ( closed push state ) & wait
log "HEAD_DONE"
