#!/bin/bash
# PAA day 1 (2026-10-31): baseline replication on LeWM PushT under the long-horizon protocol used by TEMPO / NGC / SAGE
# (goal offset d, budget 2d, 50 start-goal pairs from the dataset, seed = pair sampler). Two costs per offset:
#   released  : released latent cost, horizon 5                  (published "LeWM + CEM")
#   relmin    : released cost + min-over-plan, horizon 10         (our fair control from Method C)
# Each log keeps the sampled row indices ("[...]" line) so later PAA runs are scored on the same pairs.
# usage: bash scripts/pod/run_paa_baselines.sh [seeds="0 42"] [offsets="25 50 75 100 150"] [out=/root/out/paa_base]
set -u
SEEDS=${1:-"0 42"}; OFFS=${2:-"25 50 75 100 150"}; OUT=${3:-/root/out/paa_base}; mkdir -p "$OUT"
R=${JEPAWM_HOME:-/root/jepa-wms-project}
SWM=$R/code/third_party/stable-worldmodel; PY=${SWM_PY:-/root/venv_swm/bin/python}
M=quentinll/lewm-pusht; DS=galilai-group/lewm-pusht
export HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$OUT/queue.log"; }
cd "$SWM"
$PY -c "import stable_worldmodel as swm; swm.wm.utils.load_pretrained('$M')" > "$OUT/ckpt.log" 2>&1 || { log "checkpoint fetch FAILED"; exit 1; }
plan() {  # name seed offset extra...
  local name=$1 seed=$2 off=$3; shift 3
  [ -f "$OUT/$name.log" ] && grep -q '\[summary\]' "$OUT/$name.log" && return
  log "$name start"
  $PY scripts/plan/eval_wm.py --config-name=pusht policy=$M seed=$seed eval.dataset_name=$DS \
    eval.goal_offset_steps=$off eval.eval_budget=$((2 * off)) "$@" > "$OUT/$name.log" 2>&1 || log "$name FAILED"
  echo "$name $(grep -a '\[summary\]' "$OUT/$name.log")" >> "$OUT/summary.txt"
  log "$name done: $(grep -a '\[summary\]' "$OUT/$name.log" | grep -oE 'success[^ ]*' | head -1)"
}
for seed in $SEEDS; do
  for off in $OFFS; do
    plan released_h5_d${off}_s$seed $seed $off
    plan relmin_h10_d${off}_s$seed $seed $off +metric_head=released +metric_agg=min plan_config.horizon=10
  done
done
log "PAA_BASE_DONE"
