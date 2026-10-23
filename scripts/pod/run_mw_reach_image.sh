#!/bin/bash
# Stage 2 of the reach-then-change method (2026-10-23/24): the cost from images only (costs.ReachThenChangeImage).
#   1. robot-mask training frames: expert and goal frames of episodes 100-123 on the four tasks (mw_pilot_data --robot_mask --no_cands)
#   2. robot map: per-patch logistic head on the frozen patch features (mw_robot_map.py --modes mask; the proprio mode is at chance)
#   3. smoke: one episode of button-press-topdown with --cost reach:<robot_mask.pt>; the log line "[reach-img] ..." shows the two phases,
#      "reach-then-change: {...}" the change map and the 3D object estimate (must be within a few cm of the true object)
#   4. closed loop, 12 episodes, button / push / drawer-open; then the door-open oracles again with the fixed true cost
# Rule (docs/11, stated before the runs): PASS if >= 4/12 on >= 2 of the tasks. Writes ALL_DONE at the end for backup_and_stop.sh.
# usage: bash scripts/pod/run_mw_reach_image.sh [out=/root/out/S2]      (needs the JEPA-WM venv and the Metaworld data; check_egl.sh first)
source $(dirname $0)/jwm_env.sh
O=${1:-/root/out/S2}; mkdir -p $O; C=$JEPAWM_HOME/configs
log() { echo "$(date -u +%F' '%T) $*" >> $O/queue.log; }
finish() { log "$1"; touch $O/ALL_DONE; exit 0; }
TASKS="drawer-open button-press-topdown push door-open"
log "stage 2 start"
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)" > /dev/null 2>&1
data() { local t=$1; mkdir -p $O/data/$t
  timeout 3600 python -m evals.failure_attribution.mw_pilot_data --fname $C/mw_jepa-wm_$t.yaml --out_dir $O/data/$t --ep_start 100 --episodes 24 \
    --robot_mask --no_cands --num_workers 4 >> $O/data/$t.log 2>&1; log "data $t rc=$? eps=$(ls $O/data/$t/ep_*.npz 2>/dev/null | wc -l)"; }
( data drawer-open; data push ) & ( data button-press-topdown; data door-open ) & wait
timeout 1800 python -m evals.failure_attribution.mw_robot_map --dirs $O/data/drawer-open $O/data/button-press-topdown $O/data/push $O/data/door-open \
  --modes mask --out_dir $O/robot > $O/robot.log 2>&1; log "robot map rc=$? $(grep -E '^mask ' $O/robot.log)"
[ -f $O/robot/robot_mask.pt ] || finish "ABORT no robot map"
timeout 1200 python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_button-press-topdown.yaml --out_dir $O/smoke_button --episodes 1 \
  --planners sim_latent --num_workers 8 --cost reach:$O/robot/robot_mask.pt > $O/smoke_button.log 2>&1
log "smoke rc=$? $(grep -m1 'reach-then-change' $O/smoke_button.log) $(grep -m1 'reach-img' $O/smoke_button.log) $(grep -m1 'ep 000 done' $O/smoke_button.log | cut -c1-50)"
grep -q "ep 000 done" $O/smoke_button.log || finish "ABORT smoke failed (see smoke_button.log)"
closed() {  # task [extra args]
  local t=$1 d=$O/$1; shift; mkdir -p $d; log "closed $t start"
  timeout 7200 python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_$t.yaml --out_dir $d --episodes 12 \
    --planners sim_latent --num_workers 5 --cost reach:$O/robot/robot_mask.pt "$@" >> $d.log 2>&1; rc=$?
  python -m evals.failure_attribution.summarize_closed --run_dir $d > $d/summary.txt 2>&1
  log "closed $t rc=$rc $(grep -E '^  sim_' $d/summary.txt | sed -E 's/ +\| success at any replan.*//' | tr -s ' ')"
}
( closed button-press-topdown; closed drawer-open ) & ( closed push ) & wait
# door-open oracles with the corrected true cost (the first run used obj_to_target, which the door env reports as 0)
d=$O/door-open_oracle; mkdir -p $d
timeout 7200 python -m evals.failure_attribution.run_closed --fname $C/mw_jepa-wm_door-open.yaml --out_dir $d --episodes 12 \
  --planners sim_true sim_true_reach --num_workers 8 >> $d.log 2>&1
python -m evals.failure_attribution.summarize_closed --run_dir $d > $d/summary.txt 2>&1
log "closed door-open oracles $(grep -E '^  sim_' $d/summary.txt | sed -E 's/ +\| success at any replan.*//' | tr -s ' ')"
finish "STAGE2_DONE"
