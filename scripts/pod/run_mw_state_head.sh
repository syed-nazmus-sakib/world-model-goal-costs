#!/bin/bash
# Criterion-aligned (state) head on Metaworld tasks (pods F/G, 2026-10-23): per task, simulator-labelled encodings
# (mw_pilot_data), a head trained on the task's own success distance (mw_pilot_heads --modes state), then closed loop
# with exact dynamics (12 episodes) for the head and for the released cost on the same episodes.
# Rule stated before the runs: the head "works" if it beats the released cost by >= 3/12 on >= 4 of the 8 object tasks,
# with reach-wall and drawer-close (controls) >= 10/12.
# Existing heads can be reused: put them at <out>/<task>/heads/head_state.pt and the data step is skipped.
# usage: bash scripts/pod/run_mw_state_head.sh [out=/root/out/F] [tasks="push button-press-topdown ..."] [released=1]
source $(dirname $0)/jwm_env.sh
O=${1:-/root/out/F}; mkdir -p $O
TASKS=${2:-"push button-press-topdown door-open window-open faucet-open drawer-close reach-wall drawer-open plate-slide"}
RELEASED=${3:-1}
C=$JEPAWM_HOME/configs
log() { echo "$(date -u +%F' '%T) $*" >> $O/queue.log; }
disk_ok() { local f=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); log "disk free ${f}G"; [ "$f" -ge 10 ]; }
finish() { log "$1"; touch $O/ALL_DONE; exit 0; }
log "queue start"
# fetch the DINOv2 hub code once: two runs starting together race on the torch hub cache
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)" > /dev/null 2>&1
for t in $TASKS; do
  [ -f $O/$t/heads/head_state.pt ] && continue
  for split in test:0:12 train:100:24; do
    IFS=: read name st n <<< "$split"
    d=$O/$t/$name; mkdir -p $d
    disk_ok || finish "ABORT low disk before data $t $name"
    timeout 5400 python -m evals.failure_attribution.mw_pilot_data --fname $C/mw_jepa-wm_$t.yaml --out_dir $d \
      --ep_start $st --episodes $n --num_workers 10 >> $d.log 2>&1
    log "data $t $name rc=$? eps=$(ls $d/ep_*.npz 2>/dev/null | wc -l)"
  done
  timeout 3600 python -m evals.failure_attribution.mw_pilot_heads --train_dir $O/$t/train --test_dir $O/$t/test --task mw-$t \
    --out_dir $O/$t/heads --modes state > $O/$t/heads.log 2>&1
  log "heads $t rc=$?"
  rm -f $O/$t/train/ep_*.npz  # keep the disk small; the test encodings stay for offline checks
done
closed() {  # task name [cost args]
  local t=$1 n=$2 d=$O/$1/closed_$2; shift 2
  mkdir -p $d
  timeout 7200 python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_$t.yaml --out_dir $d --episodes 12 \
    --planners sim_latent --num_workers 5 "$@" >> $d.log 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir $d > $d/summary.txt 2>&1
  log "closed $t $n $(grep -m1 sim_latent $d/summary.txt)"
}
jobs_for() {  # one lane: every other task
  local i=0
  for t in $TASKS; do
    if [ $((i % 2)) = $1 ]; then
      [ -f $O/$t/heads/head_state.pt ] && closed $t state --cost head:$O/$t/heads/head_state.pt
      [ "$RELEASED" = 1 ] && closed $t released
    fi
    i=$((i + 1))
  done
}
disk_ok || finish "ABORT low disk before closed loop"
jobs_for 0 & jobs_for 1 & wait
finish "QUEUE_DONE"
