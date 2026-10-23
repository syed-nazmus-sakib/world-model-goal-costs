#!/bin/bash
# Method C on Metaworld (2026-10-22): one time-to-goal head on frozen JEPA-WM encodings of the released expert dataset
# (self-supervised), then closed loop with exact simulator dynamics (sim_latent, the eval's CEM), head vs released cost.
# usage: bash scripts/pod/run_mw_time_head.sh [out=/root/out/MW] [tasks="drawer-open plate-slide ..."]
# Needs /root/venv_jwm (setup_jwm_venv.sh). About 15 GB of disk for the encodings; checked before every step.
source $(dirname $0)/jwm_env.sh
O=${1:-/root/out/MW}; mkdir -p $O
TASKS=${2:-"drawer-open plate-slide faucet-open drawer-close reach-wall push door-open button-press-topdown window-open"}
C=$JEPAWM_HOME/configs
log() { echo "$(date -u +%F' '%T) $*" | tee -a $O/queue.log; }
disk_ok() { local f=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); log "disk free ${f}G"; [ "$f" -ge 10 ]; }
DATA_TASKS=mw-reach-wall,mw-push,mw-drawer-close,mw-button-press-topdown,mw-door-open,mw-drawer-open,mw-faucet-open,mw-plate-slide
if [ ! -f $O/head_time.pt ]; then  # window-open is not in the dataset: it stays a held-out task
  disk_ok || exit 1
  python -m evals.failure_attribution.mw_head_data --fname $C/mw_jepa-wm_drawer-open.yaml --tasks $DATA_TASKS \
    --episodes 60 --stride 3 --out $O/mw_latents.npz > $O/data.log 2>&1 || { log "data FAILED"; exit 1; }
  python -m evals.failure_attribution.mw_head_train --data $O/mw_latents.npz --out $O/head_time.pt > $O/train.log 2>&1 || { log "train FAILED"; exit 1; }
fi
for t in $TASKS; do
  for lane in head released; do
    d=$O/$t/$lane; mkdir -p $d
    disk_ok || exit 1
    extra=""; [ $lane = head ] && extra="--cost head:$O/head_time.pt"
    python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_$t.yaml --out_dir $d --episodes 12 \
      --planners sim_latent --num_workers 8 $extra >> $d.log 2>&1 || log "$t $lane FAILED"
    python -m evals.failure_attribution.summarize_closed --run_dir $d > $d/summary.txt 2>&1
    log "$t $lane $(grep -m1 sim_latent $d/summary.txt)"
  done
done
log "DONE"
