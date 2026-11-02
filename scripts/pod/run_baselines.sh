#!/bin/bash
# Navigation baselines (2026-11-01), seed 0, same pairs and protocol as the graph anchor (offset d, budget 2d, 50 pairs).
#   gcbc / gcivl / hilp : offline goal-conditioned policies on frozen DINOv2 features (stable-worldmodel), STEPS batches
#                         per training phase (GCIVL/HILP: value then policy), evaluated with eval_wm.py +ff=true
#   pldm                : PLDM world model, STEPS_WM batches, planned with the same CEM as LeWM
#   dinowm              : released DINO-WM PushT checkpoint (pusht only), N_DINO pairs at d50/d100 (planning is slow)
# XB/XSEEDS: also evaluate on stitched goals (goals from other episodes) at those budgets and seeds
# usage: bash scripts/pod/run_baselines.sh <tworoom|pmm|pusht> "<methods>" [out=/root/out/bl_<env>]
set -u
ENV=$1; METHODS=$2; OUT=${3:-/root/out/bl_$ENV}; mkdir -p "$OUT"
R=${JEPAWM_HOME:-/root/jepa-wms-project}
SWM=$R/code/third_party/stable-worldmodel; P=$R/code/lewm_explorations/paa; PY=${SWM_PY:-/root/venv_swm/bin/python}
STEPS=${STEPS:-5000}; STEPS_WM=${STEPS_WM:-10000}; N_DINO=${N_DINO:-20}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ -f /usr/lib64-nvidia/libEGL_nvidia.so.0 ]; then
  V=/usr/share/glvnd/egl_vendor.d/10_nvidia.json; mkdir -p "$(dirname $V)"
  [ -f $V ] || printf '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib64-nvidia/libEGL_nvidia.so.0"}}\n' > $V
  export __EGL_VENDOR_LIBRARY_FILENAMES=$V LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
fi
case $ENV in
  tworoom) DATA=tworoom;          DSN=tworoom.h5;                       PD=2; CFG=tworoom;          OFFS=${OFFS-"25 50 75"} ;;
  pmm)     DATA=pointmaze_medium; DSN=d4rl_pointmaze_medium_expert.h5;  PD=4; CFG=pointmaze_medium; OFFS=${OFFS-"50 100 150 200"} ;;
  pusht)   DATA=pusht_lewm;       DSN=galilai-group/lewm-pusht;         PD=4; CFG=pusht;            OFFS=${OFFS-"50 75 100 150"} ;;
  *) echo "unknown env $ENV"; exit 1 ;;
esac
CK=${STABLEWM_HOME:-/root/.stable_worldmodel}/checkpoints
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$OUT/queue.log"; }
cd "$SWM"
if [ "$ENV" != pusht ] && [ ! -f "$OUT/FETCHED" ]; then
  $PY $P/fetch_lewm_data.py $ENV > "$OUT/fetch.log" 2>&1 && touch "$OUT/FETCHED" || { log "fetch FAILED"; exit 1; }
fi
LIM="+wandb.enable=false trainer.max_epochs=1 +trainer.limit_val_batches=20"
train() {  # method
  local m=$1 n=bl_${ENV}_$1 t0=$(date +%s)
  [ -f "$OUT/train_$m.done" ] && return 0
  log "train $m start"
  case $m in
    gcbc)  $PY scripts/train/gcbc.py data=$DATA wm.proprio_dim=$PD output_model_name=$n $LIM +trainer.limit_train_batches=$STEPS ;;
    gcivl) $PY scripts/train/gcivl.py dataset_name=$DSN dinowm.proprio_dim=$PD +subdir=bl output_model_name=$n $LIM +trainer.limit_train_batches=$STEPS ;;
    hilp)  $PY scripts/train/hilp.py dataset_name=$DSN dinowm.proprio_dim=$PD +subdir=bl output_model_name=$n $LIM +trainer.limit_train_batches=$STEPS ;;
    pldm)  $PY scripts/train/pldm.py data=$DATA output_model_name=$n trainer.max_epochs=1 +trainer.limit_train_batches=$STEPS_WM +trainer.limit_val_batches=20 ;;
  esac > "$OUT/train_$m.log" 2>&1
  local rc=$?; log "train $m rc=$rc $(( $(date +%s) - t0 ))s $(grep -aE '^[A-Za-z.]*(Error|Exception)' $OUT/train_$m.log | tail -1 | cut -c1-160)"
  [ $rc = 0 ] && touch "$OUT/train_$m.done"; return $rc
}
ev() {  # name offset extra...
  local name=$1 off=$2; shift 2
  [ -f "$OUT/$name.log" ] && grep -q '\[summary\]' "$OUT/$name.log" && return
  $PY scripts/plan/eval_wm.py --config-name=$CFG seed=0 eval.dataset_name=$DSN eval.goal_offset_steps=$off \
    eval.eval_budget=$((2 * off)) "$@" ${EVX:-} > "$OUT/$name.log" 2>&1 || log "$name FAILED"
  echo "$name $(grep -a '\[summary\]' "$OUT/$name.log")" >> "$OUT/summary.txt"
  log "$name: $(grep -a '\[summary\]' "$OUT/$name.log" | grep -oE 'success_rate=[0-9.]+')"
}
xev() {  # method policy-args...: stitched goals (+cross_goal) at budgets XB, seeds XSEEDS, same pairs as run_ga_env.sh cross
  local m=$1; shift
  for seed in ${XSEEDS:-}; do for bud in ${XB:-}; do
    local name=x${m}_b${bud}_s$seed
    [ -f "$OUT/$name.log" ] && grep -q '\[summary\]' "$OUT/$name.log" && continue
    $PY scripts/plan/eval_wm.py --config-name=$CFG seed=$seed eval.dataset_name=$DSN eval.goal_offset_steps=5 eval.eval_budget=$bud \
      +cross_goal=true +pairs_npz=$OUT/pairs_$name.npz "$@" ${EVX:-} > "$OUT/$name.log" 2>&1 || log "$name FAILED"
    echo "$name $(grep -a '\[summary\]' "$OUT/$name.log")" >> "$OUT/summary.txt"
    log "$name: $(grep -a '\[summary\]' "$OUT/$name.log" | grep -oE 'success_rate=[0-9.]+')"
  done; done
}
for m in $METHODS; do
  case $m in
    gcbc)       train gcbc  && { for d in $OFFS; do ev gcbc_d${d}_s0 $d policy=bl_${ENV}_gcbc +ff=true; done; xev gcbc policy=bl_${ENV}_gcbc +ff=true; } ;;
    gcivl|hilp) train $m    && { A="policy=bl_${ENV}_${m}_policy +ff=true +policy_object=$CK/bl_${ENV}_${m}_policy/model_object.pt"
                                 for d in $OFFS; do ev ${m}_d${d}_s0 $d $A; done; xev $m $A; } ;;
    pldm)       train pldm  && { for d in $OFFS; do ev pldm_d${d}_s0 $d policy=bl_${ENV}_pldm; done; xev pldm policy=bl_${ENV}_pldm; } ;;
    dinowm)
      [ "$ENV" = pusht ] || { log "dinowm: pusht only"; continue; }
      D=$CK/dinowm_pusht; mkdir -p $D
      [ -f $D/model_object.pt ] || $PY $P/convert_lightning_ckpt.py ${DINO_CFG:-/content/dinowm/config.yaml} \
          ${DINO_CKPT:-/content/dinowm/dinowm_noprop_pusht_weights.ckpt} $D/model_object.pt > "$OUT/dinowm_convert.log" 2>&1 \
          || { log "dinowm convert FAILED: $(tail -1 $OUT/dinowm_convert.log)"; continue; }
      for d in 50 100; do ev dinowm_d${d}_s0_n$N_DINO $d policy=dinowm_pusht +policy_object=$D/model_object.pt \
          objective=goal_mse_pixels eval.num_eval=$N_DINO; done ;;
  esac
done
log "BL_LANE_DONE $ENV $METHODS"
