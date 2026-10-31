#!/bin/bash
# Graph anchor on other LeWM envs (2026-10-31). Same protocol as PushT: goal offset d, budget 2d, 50 pairs, seed = pair sampler.
#   fetch : dataset from the HF collection quentinll/lewm (fetch_lewm_data.py)
#   nodes : encode the dataset's 5-step grid with the frozen encoder
#   eval  : released h5, relmin h10 (released + min-over-plan), graph anchor span 6 and 8 (min, h10) at each offset
# Offsets: OFFS (default "50 100 150"); an offset is skipped when fewer than 60 episodes are longer than it.
# usage: bash scripts/pod/run_ga_env.sh <tworoom|cube|reacher> [seeds="0"] [out=/root/out/ga_<env>] [lane=all|base|anchor]
set -u
ENV=$1; SEEDS=${2:-"0"}; OUT=${3:-/root/out/ga_$ENV}; LANE=${4:-all}; mkdir -p "$OUT"
R=${JEPAWM_HOME:-/root/jepa-wms-project}
SWM=$R/code/third_party/stable-worldmodel; P=$R/code/lewm_explorations/paa; PY=${SWM_PY:-/root/venv_swm/bin/python}
export MUJOCO_GL=egl HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
case $ENV in
  tworoom) CFG=tworoom; M=quentinll/lewm-tworooms; KEY=pos_agent ;;
  cube)    CFG=cube;    M=quentinll/lewm-cube;     KEY=none ;;
  reacher) CFG=reacher; M=quentinll/lewm-reacher;  KEY=none ;;
  *) echo "unknown env $ENV"; exit 1 ;;
esac
DS=${DS:-$(cat "$OUT/dataset_name" 2>/dev/null)}
NODES=$OUT/nodes.npz
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$OUT/queue.log"; }
abort() { log "$1"; touch "$OUT/ABORT"; exit 1; }
cd "$SWM"
if [ "$LANE" != anchor ]; then
  if [ -z "$DS" ]; then
    $PY $P/fetch_lewm_data.py $ENV > "$OUT/fetch.log" 2>&1 || abort "fetch FAILED"
    DS=$(grep -a '^H5 ' "$OUT/fetch.log" | awk '{print $2}' | grep -iv test | head -1)
    [ -n "$DS" ] || abort "no dataset found"
    echo "$DS" > "$OUT/dataset_name"; log "dataset $DS"
  fi
  $PY -c "import stable_worldmodel as swm; swm.wm.utils.load_pretrained('$M')" > "$OUT/ckpt.log" 2>&1 || abort "checkpoint FAILED"
  [ -f $NODES ] || { log "nodes start"; $PY $P/paa_nodes.py --out $NODES --model $M --dataset $DS --state_key $KEY --workers 8 \
      > $OUT/nodes.log 2>&1 || abort "nodes FAILED"; log "nodes: $(tail -1 $OUT/nodes.log)"; }
  $PY -c "
import numpy as np; d=np.load('$NODES'); ep,st=d['ep'],d['step']; u=np.unique(ep)
ln=np.array([st[ep==e].max() for e in u]); print('episodes',len(u),'nodes',len(ep),'len median',int(np.median(ln)),
{o:int((ln>o).sum()) for o in (25,50,75,100,150,200)})" | tee -a "$OUT/queue.log"
  touch "$OUT/READY"
fi
while [ ! -f "$OUT/READY" ]; do [ -f "$OUT/ABORT" ] && { log "anchor lane: setup aborted"; exit 1; }; sleep 30; done   # anchor lane waits for the nodes
DS=$(cat "$OUT/dataset_name")
ok_off() { $PY -c "
import numpy as np; d=np.load('$NODES'); ep,st=d['ep'],d['step']
print(int(sum(st[ep==e].max()>$1 for e in np.unique(ep))>=60))"; }
ev() {  # name seed offset extra...
  local name=$1 seed=$2 off=$3; shift 3
  [ -f "$OUT/$name.log" ] && grep -q '\[summary\]' "$OUT/$name.log" && return
  log "$name start"
  $PY scripts/plan/eval_wm.py --config-name=$CFG policy=$M seed=$seed eval.dataset_name=$DS \
    eval.goal_offset_steps=$off eval.eval_budget=$((2 * off)) "$@" > "$OUT/$name.log" 2>&1 || log "$name FAILED"
  echo "$name $(grep -a '\[summary\]' "$OUT/$name.log")" >> "$OUT/summary.txt"
  log "$name: $(grep -a '\[summary\]' "$OUT/$name.log" | grep -oE 'success_rate=[0-9.]+')"
}
for seed in $SEEDS; do for off in ${OFFS:-50 100 150}; do
  [ "$(ok_off $off)" = 1 ] || { log "skip d$off: fewer than 60 episodes longer than $off"; continue; }
  if [ "$LANE" != anchor ]; then
    ev released_h5_d${off}_s$seed $seed $off
    ev relmin_h10_d${off}_s$seed $seed $off +metric_head=released +metric_agg=min plan_config.horizon=10
  fi
  if [ "$LANE" != base ]; then
    for sp in ${SPANS:-6 8}; do
      ev ga_sp${sp}_d${off}_s$seed $seed $off +paa_nodes=$NODES +paa_mode=fixed +paa_span=$sp +metric_agg=min plan_config.horizon=10
    done
  fi
done; done
log "GA_LANE_DONE $ENV $LANE"
