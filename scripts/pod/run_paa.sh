#!/bin/bash
# PAA pipeline on LeWM PushT (2026-10-31). Phases (comma list): nodes,fixed,harvest,head,paa
#   nodes   : encode the dataset's 5-step grid (graph nodes)
#   fixed   : training-free anchors (span 2/4/6) at d50/75/100/150, same pairs as the baselines (seed = pair sampler)
#   harvest : planner attempts toward same-episode anchors 2..12 blocks ahead, from training episodes that never appear
#             in any evaluation pair (seeds 0/42/1/2, all offsets); horizon 10 + min, 10-block budget, final latents dumped
#   head    : attainability ensemble on the harvest (label = final real latent within eps of the anchor)
#   paa     : PAA runs at d50/75/100/150
# usage: bash scripts/pod/run_paa.sh <phases> [seeds="0"] [out=/root/out/paa]
set -u
PH=$1; SEEDS=${2:-"0"}; OUT=${3:-/root/out/paa}; mkdir -p "$OUT/harvest"
R=${JEPAWM_HOME:-/root/jepa-wms-project}
SWM=$R/code/third_party/stable-worldmodel; P=$R/code/lewm_explorations/paa; PY=${SWM_PY:-/root/venv_swm/bin/python}
M=quentinll/lewm-pusht; DS=galilai-group/lewm-pusht; NODES=$OUT/nodes.npz
export HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$OUT/queue.log"; }
cd "$SWM"
ev() {  # name seed offset budget extra...
  local name=$1 seed=$2 off=$3 bud=$4; shift 4
  [ -f "$OUT/$name.log" ] && grep -q '\[summary\]' "$OUT/$name.log" && return
  $PY scripts/plan/eval_wm.py --config-name=pusht policy=$M seed=$seed eval.dataset_name=$DS \
    eval.goal_offset_steps=$off eval.eval_budget=$bud "$@" > "$OUT/$name.log" 2>&1 || log "$name FAILED"
  echo "$name $(grep -a '\[summary\]' "$OUT/$name.log")" >> "$OUT/summary.txt"
  log "$name: $(grep -a '\[summary\]' "$OUT/$name.log" | grep -oE 'success_rate=[0-9.]+')"
}
if echo $PH | grep -q nodes; then
  [ -f $NODES ] || { log "nodes start"; $PY $P/paa_nodes.py --out $NODES > $OUT/nodes.log 2>&1 || log "nodes FAILED"; log "nodes: $(tail -1 $OUT/nodes.log)"; }
fi
if echo $PH | grep -q fixed; then
  for seed in $SEEDS; do for off in 50 75 100 150; do for sp in ${SPANS:-6 8 10}; do
    ev fixed_sp${sp}_d${off}_s$seed $seed $off $((2 * off)) +paa_nodes=$NODES +paa_mode=fixed +paa_span=$sp +metric_agg=min plan_config.horizon=10
  done; done; done
fi
if echo $PH | grep -q harvest; then
  # episodes used by any evaluation pair (seeds 0, 42, 1, 2 at every offset) are never harvested; pairs come from
  # eval_wm's own sampler (pairs_only), so the exclusion matches the evaluation exactly
  mkdir -p $OUT/pairs
  for seed in 0 42 1 2; do for off in 25 50 75 100 125 150; do
    [ -f $OUT/pairs/s${seed}_d$off.npz ] || $PY scripts/plan/eval_wm.py --config-name=pusht policy=$M seed=$seed eval.dataset_name=$DS \
      eval.goal_offset_steps=$off eval.eval_budget=$((2 * off)) +pairs_npz=$OUT/pairs/s${seed}_d$off.npz +pairs_only=true > $OUT/pairs/s${seed}_d$off.log 2>&1
  done; done
  $PY -c "import numpy as np,glob; e=np.unique(np.concatenate([np.load(f)['ep'] for f in glob.glob('$OUT/pairs/*.npz')])); np.save('$OUT/eval_eps.npy', e); print(len(e), 'evaluation episodes excluded')" | tee -a $OUT/queue.log
  for rep in $(seq 100 109); do for h in 2 4 6 8 10 12; do
    ev harvest/h${h}_r$rep $rep $((5 * h)) 50 +metric_head=released +metric_agg=min plan_config.horizon=10 \
      +exclude_eps=$OUT/eval_eps.npy +dump_npz=$OUT/harvest/h${h}_r$rep.npz
  done; done
fi
if echo $PH | grep -q head; then
  $PY $P/paa_head.py --nodes $NODES --dumps "$OUT/harvest/*.npz" --out $OUT/head_latent.pt > $OUT/head_latent.log 2>&1
  $PY $P/paa_head.py --nodes $NODES --dumps "$OUT/harvest/*.npz" --out $OUT/head_success.pt --label success > $OUT/head_success.log 2>&1
  log "head: $(grep -E 'attempts|val acc' $OUT/head_latent.log | tr '\n' ' ')"
fi
if echo $PH | grep -q paa; then
  for seed in $SEEDS; do for off in 50 75 100 150; do
    ev paa_d${off}_s$seed $seed $off $((2 * off)) +paa_nodes=$NODES +paa_mode=paa +paa_head=$OUT/head_latent.pt +paa_span=2 +metric_agg=min plan_config.horizon=10
  done; done
fi
log "PAA_PHASES_DONE $PH"
