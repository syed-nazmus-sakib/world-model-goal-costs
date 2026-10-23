#!/bin/bash
# Method D pilot on Metaworld (pod E, 2026-10-22): which supervision makes a goal-cost head plan?
#   1. mw_pilot_data: simulator-labelled encodings; test = the closed-loop eval episodes 0-11, train = episodes 100-123
#   2. mw_pilot_heads: heads state (privileged), time, evid (64 success + 64 near-miss frames), evid_time; offline test
#   3. closed loop, 12 episodes: a head runs on object tasks only if it ranks above chance offline (accuracy >= 0.6
#      or its top-1 candidate ever succeeds); evid_time always runs on the two controls
# Writes ALL_DONE at the very end (also on abort) for a backup-and-stop watcher (backup_and_stop.sh).
# usage: bash scripts/pod/run_mw_methodD_pilot.sh [out=/root/out/D] [methodC_time_head=/root/out/head_time.pt]
source $(dirname $0)/jwm_env.sh
O=${1:-/root/out/D}; CH=${2:-/root/out/head_time.pt}; mkdir -p $O
C=$JEPAWM_HOME/configs
log() { echo "$(date -u +%F' '%T) $*" >> $O/queue.log; }
disk_ok() { local f=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); log "disk free ${f}G"; [ "$f" -ge 10 ]; }
finish() { log "$1"; touch $O/ALL_DONE; exit 0; }
TASKS="drawer-open plate-slide drawer-close reach-wall"
log "queue start"
for t in $TASKS; do
  for split in test:0:12 train:100:24; do
    IFS=: read name st n <<< "$split"
    d=$O/$t/$name; mkdir -p $d
    disk_ok || finish "ABORT low disk before data $t $name"
    timeout 5400 python -m evals.failure_attribution.mw_pilot_data --fname $C/mw_jepa-wm_$t.yaml --out_dir $d \
      --ep_start $st --episodes $n --num_workers 10 >> $d.log 2>&1
    log "data $t $name rc=$? eps=$(ls $d/ep_*.npz 2>/dev/null | wc -l)"
  done
done
for t in $TASKS; do
  [ -f $CH ] && c_head="--c_head $CH" || c_head=""
  timeout 5400 python -m evals.failure_attribution.mw_pilot_heads --train_dir $O/$t/train --test_dir $O/$t/test --task mw-$t \
    --out_dir $O/$t/heads $c_head > $O/$t/heads.log 2>&1
  log "heads $t rc=$?"
done
passes() { python - "$1" "$2" <<'P'
import json, sys
r = json.load(open(sys.argv[1]))["results"][sys.argv[2]]["eval_goal"]
sys.exit(0 if (r["acc"] >= 0.6 or r["top1"] > 0) else 1)
P
}
closed() {  # task mode [always]
  local t=$1 m=$2 h=$O/$1/heads/head_$2.pt d=$O/$1/closed_$2
  [ -f $h ] || { log "closed $t $m SKIPPED (no head)"; return; }
  if [ "$3" != always ] && ! passes $O/$t/heads/offline.json $m; then log "closed $t $m SKIPPED (offline gate)"; return; fi
  mkdir -p $d
  timeout 7200 python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_$t.yaml --out_dir $d --episodes 12 \
    --planners sim_latent --num_workers 5 --cost head:$h >> $d.log 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir $d > $d/summary.txt 2>&1
  log "closed $t $m $(grep -m1 sim_latent $d/summary.txt)"
}
disk_ok || finish "ABORT low disk before closed loop"
( closed drawer-open state; closed drawer-open evid_time; closed plate-slide evid_time; closed reach-wall evid_time always ) &
( closed drawer-open evid; closed plate-slide state; closed plate-slide evid; closed drawer-close evid_time always ) &
wait
finish "QUEUE_DONE"
