#!/bin/bash
# Method C on LeWM (2026-10-22): learned goal cost ||f(z) - f(g)||^2 on frozen LeWM latents, against the released cost.
#   1. latents: 120k 6-frame clips of the env's dataset (real frames + the predictor's rollout), with simulator states
#   2. heads:   --target state (task success distance, labels at training time) and --target time (steps, self-supervised)
#   3. plans:   released cost h5, released + min-over-plan h10 (the fair control), each head h5 and + min h10;
#               goal offsets 50 and 75 (budget 2x offset), 50 goals per run
# usage: bash scripts/pod/run_lewm_methodC.sh <pusht|tworoom|cube> [seed=0] [out=/root/out/C_<env>]
# Needs /root/venv_swm (setup_swm_venv.sh). Runs one at a time: a Cube eval needs about 20 GB of RAM.
set -u
ENV=$1; SEED=${2:-0}; OUT=${3:-/root/out/C_$ENV}; mkdir -p $OUT
R=${JEPAWM_HOME:-/root/jepa-wms-project}
SWM=$R/code/third_party/stable-worldmodel; CS=$R/code/lewm_explorations/scripts; PY=/root/venv_swm/bin/python
export MUJOCO_GL=egl HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
case $ENV in
  pusht)   CFG=pusht;   M=quentinll/lewm-pusht;    DS=galilai-group/lewm-pusht;    KEY=state ;;
  tworoom) CFG=tworoom; M=quentinll/lewm-tworooms; DS=tworoom.h5;                  KEY=pos_agent ;;
  cube)    CFG=cube;    M=quentinll/lewm-cube;     DS=galilai-group/ogb_cube_single; KEY=privileged_block_0_pos ;;
  *) echo "unknown env $ENV"; exit 1 ;;
esac
log() { echo "$(date -u +%F' '%T) $*" | tee -a $OUT/queue.log; }
cd $SWM
$PY -c "import stable_worldmodel as swm; swm.wm.utils.load_pretrained('$M')" > $OUT/ckpt.log 2>&1  # fetch once: parallel downloads race
if [ ! -f $OUT/head_time.pt ]; then
  log "latents start"
  $PY $CS/candC_data.py --out $OUT/latents.npz --clips 120000 --model $M --dataset $DS --state_key $KEY > $OUT/latents.log 2>&1 || { log "latents FAILED"; exit 1; }
  for t in state time; do
    $PY $CS/candC_train.py --data $OUT/latents.npz --target $t --env $ENV --out $OUT/head_$t.pt > $OUT/train_$t.log 2>&1 || log "train $t FAILED"
  done
fi
plan() {  # name offset extra...
  local name=$1 off=$2; shift 2
  [ -f $OUT/$name.log ] && grep -q '\[summary\]' $OUT/$name.log && return
  log "$name start"
  $PY scripts/plan/eval_wm.py --config-name=$CFG policy=$M seed=$SEED eval.dataset_name=$DS \
    eval.goal_offset_steps=$off eval.eval_budget=$((2 * off)) "$@" > $OUT/$name.log 2>&1 || log "$name FAILED"
  echo "$name $(grep -a '\[summary\]' $OUT/$name.log)" >> $OUT/summary.txt
}
for off in 50 75; do
  plan released_h5_off${off}_s$SEED $off
  plan relmin_h10_off${off}_s$SEED $off +metric_head=released +metric_agg=min plan_config.horizon=10
  for t in state time; do
    plan ${t}_h5_off${off}_s$SEED $off +metric_head=$OUT/head_$t.pt
    plan ${t}_h10min_off${off}_s$SEED $off +metric_head=$OUT/head_$t.pt +metric_agg=min plan_config.horizon=10
  done
done
log "DONE env=$ENV seed=$SEED (success rates in $OUT/summary.txt)"
