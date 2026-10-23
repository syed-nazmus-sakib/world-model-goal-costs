#!/bin/bash
# Stage 1 of the reach-then-change method (pods H-J, 2026-10-23): the oracle check. Simulator states give the reach term
# and the gate; the change term is the released latent cost (sim_latent_reach) or the true object cost (sim_true_reach).
# Controls on the same episodes: plain true object cost (sim_true); released cost alone is known 0/12 on all four tasks.
# Rule stated before the runs: PASS if sim_latent_reach >= 6/12 on >= 2 of the 4 object tasks and reach-wall (fallback) >= 10/12.
source $(dirname $0)/jwm_env.sh
# usage: bash scripts/pod/run_mw_reach_oracle.sh [out=/root/out/R1]      (needs the JEPA-WM venv and the Metaworld data)
O=${1:-/root/out/R1}; mkdir -p $O
C=$JEPAWM_HOME/configs
log() { echo "$(date -u +%F' '%T) $*" >> $O/queue.log; }
finish() { log "$1"; touch $O/ALL_DONE; exit 0; }
log "queue start"
# smoke: one episode, both reach planners; abort the queue if the new code path fails
timeout 1200 python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_drawer-open.yaml --out_dir $O/smoke --episodes 1 \
  --planners sim_latent_reach sim_true_reach --num_workers 8 > $O/smoke.log 2>&1 || finish "ABORT smoke failed rc=$? (see R1/smoke.log)"
python -m evals.failure_attribution.summarize_closed --run_dir $O/smoke > $O/smoke/summary.txt 2>&1
log "smoke OK: $(grep -E 'sim_(latent|true)_reach' $O/smoke/summary.txt | tr -s ' ' | tr '\n' ';')"
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)" > /dev/null 2>&1
closed() {  # task planners...
  local t=$1 d=$O/$1; shift
  mkdir -p $d; log "closed $t start [$*]"
  timeout 10800 python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_$t.yaml --out_dir $d --episodes 12 \
    --planners "$@" --num_workers 5 >> $d.log 2>&1; rc=$?
  python -m evals.failure_attribution.summarize_closed --run_dir $d > $d/summary.txt 2>&1
  log "closed $t rc=$rc $(grep -E '^  sim_' $d/summary.txt | sed -E 's/ +\| success at any replan.*//' | tr -s ' ' | tr '\n' ';')"
}
( closed drawer-open sim_latent_reach sim_true sim_true_reach; closed push sim_latent_reach sim_true sim_true_reach; closed reach-wall sim_latent_reach ) &
( closed button-press-topdown sim_latent_reach sim_true sim_true_reach; closed door-open sim_latent_reach sim_true sim_true_reach ) &
wait
finish "QUEUE_DONE"
