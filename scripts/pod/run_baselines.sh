#!/bin/bash
# Navigation baselines (2026-11-01), seed 0, same pairs and protocol as the graph anchor (offset d, budget 2d, 50 pairs).
#   gcbc / gcivl / hilp : offline goal-conditioned policies on frozen DINOv2 features (stable-worldmodel), STEPS batches
#                         per training phase (GCIVL/HILP: value then policy), evaluated with eval_wm.py +ff=true
#   pldm                : PLDM world model, STEPS_WM batches, planned with the same CEM as LeWM
#   dinowm              : released DINO-WM PushT checkpoint (pusht only), N_DINO pairs at d50/d100 (planning is slow)
#   gciql               : offline goal-conditioned IQL on frozen DINOv2 features (stable-worldmodel), like gcivl
#   dwm                 : DINO-WM (frozen DINOv2 patches + predictor, scripts/train/prejepa.py) trained on this dataset for
#                         EPOCHS_DWM epochs of STEPS_DWM batches, planned with the same CEM (released settings, patch-MSE cost)
# gcbcg: graph waypoints + GCBC (needs GNODES, GWM); XB/XSEEDS: also evaluate on stitched goals (goals from other episodes) at those budgets and seeds
# EVSEEDS: seeds of the standard evaluations (default 0); EPOCHS / EPOCHS_WM: training epochs (default 1, STEPS / STEPS_WM
# batches each); MTAG: suffix of the trained model names (bl_<env>_<method><MTAG>), so longer trainings do not overwrite others
# GSPAN: waypoint span of gcbcg (default 6); GTAG: run-name suffix of gcbcg (e.g. _sp2)
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
  pml)     DATA=pointmaze_large;  DSN=pointmaze_large.h5;               PD=4; CFG=pointmaze_large;  OFFS=${OFFS-"50 100 200 400"} ;;
  pusht)   DATA=pusht_lewm;       DSN=galilai-group/lewm-pusht;         PD=4; CFG=pusht;            OFFS=${OFFS-"50 75 100 150"} ;;
  *) echo "unknown env $ENV"; exit 1 ;;
esac
CK=${STABLEWM_HOME:-/root/.stable_worldmodel}/checkpoints
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$OUT/queue.log"; }
cd "$SWM"
if [ "$ENV" != pusht ] && [ ! -f "$OUT/FETCHED" ]; then
  $PY $P/fetch_lewm_data.py $ENV > "$OUT/fetch.log" 2>&1 && touch "$OUT/FETCHED" || { log "fetch FAILED"; exit 1; }
fi
MT=${MTAG:-}
LIM="+wandb.enable=false trainer.max_epochs=${EPOCHS:-1} +trainer.limit_val_batches=20"
one_pt() {  # GCBC saves one .pt per epoch and its loader needs exactly one: keep the last, move the others away
  local K=$1
  if [ "$(ls "$K"/*.pt 2>/dev/null | wc -l)" -gt 1 ]; then
    mkdir -p "$K/older_epochs"; ls "$K"/*.pt | sort -V | head -n -1 | xargs -I{} mv {} "$K/older_epochs/"
    log "checkpoint $K: kept $(ls "$K"/*.pt)"
  fi
}
train() {  # method
  local m=$1 n=bl_${ENV}_$1$MT t0=$(date +%s)
  [ -f "$OUT/train_$m.done" ] && { [ $m = gcbc ] && one_pt "$CK/$n"; return 0; }
  log "train $m start"
  case $m in
    gcbc)  $PY scripts/train/gcbc.py data=$DATA wm.proprio_dim=$PD output_model_name=$n $LIM +trainer.limit_train_batches=$STEPS ;;
    gcivl) $PY scripts/train/gcivl.py dataset_name=$DSN dinowm.proprio_dim=$PD +subdir=bl output_model_name=$n $LIM +trainer.limit_train_batches=$STEPS ;;
    hilp)  $PY scripts/train/hilp.py dataset_name=$DSN dinowm.proprio_dim=$PD +subdir=bl output_model_name=$n $LIM +trainer.limit_train_batches=$STEPS ;;
    gciql) $PY scripts/train/gciql.py dataset_name=$DSN dinowm.proprio_dim=$PD +subdir=bl output_model_name=$n $LIM +trainer.limit_train_batches=$STEPS ;;
    dwm)   $PY scripts/train/prejepa.py dataset_name=$DSN output_model_name=$n trainer.max_epochs=${EPOCHS_DWM:-5} \
               +trainer.limit_train_batches=${STEPS_DWM:-5000} +trainer.limit_val_batches=20 trainer.strategy=auto num_workers=8 ;;
    pldm)  $PY scripts/train/pldm.py data=$DATA output_model_name=$n trainer.max_epochs=${EPOCHS_WM:-1} +trainer.limit_train_batches=$STEPS_WM +trainer.limit_val_batches=20 ;;
  esac > "$OUT/train_$m.log" 2>&1
  local rc=$?; log "train $m rc=$rc $(( $(date +%s) - t0 ))s $(grep -aE '^[A-Za-z.]*(Error|Exception)' $OUT/train_$m.log | tail -1 | cut -c1-160)"
  [ $rc = 0 ] && { touch "$OUT/train_$m.done"; [ $m = gcbc ] && one_pt "$CK/$n"; }; return $rc
}
ev() {  # name(ending in _s0) offset extra...: one evaluation per seed in EVSEEDS (default 0)
  local base=${1%_s0} off=$2 name sd; shift 2
  for sd in ${EVSEEDS:-0}; do
    name=${base}_s$sd
    [ -f "$OUT/$name.log" ] && grep -q '\[summary\]' "$OUT/$name.log" && continue
    $PY scripts/plan/eval_wm.py --config-name=$CFG seed=$sd eval.dataset_name=$DSN eval.goal_offset_steps=$off \
      eval.eval_budget=$((2 * off)) "$@" ${EVX:-} > "$OUT/$name.log" 2>&1 || log "$name FAILED"
    echo "$name $(grep -a '\[summary\]' "$OUT/$name.log")" >> "$OUT/summary.txt"
    log "$name: $(grep -a '\[summary\]' "$OUT/$name.log" | grep -oE 'success_rate=[0-9.]+')"
  done
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
    gcbc)       train gcbc  && { for d in $OFFS; do ev gcbc_d${d}_s0 $d policy=bl_${ENV}_gcbc$MT +ff=true; done; xev gcbc policy=bl_${ENV}_gcbc$MT +ff=true; } ;;
    dwm)        train dwm   && { n=bl_${ENV}_dwm$MT; O=$(find "$CK" -path "*$n*" -name model_object.pt 2>/dev/null | sort | tail -1)
                                 [ -n "$O" ] || { log "dwm: no model_object.pt for $n under $CK"; continue; }
                                 OBJ=objective=goal_mse_pixels_proprio
                                 for d in $OFFS; do ev dwm_d${d}_s0 $d policy=$n +policy_object=$O $OBJ; done
                                 xev dwm policy=$n +policy_object=$O $OBJ; } ;;
    gcivl|hilp|gciql) train $m && { A="policy=bl_${ENV}_${m}${MT}_policy +ff=true +policy_object=$CK/bl_${ENV}_${m}${MT}_policy/model_object.pt"
                                 for d in $OFFS; do ev ${m}_d${d}_s0 $d $A; done; xev $m $A; } ;;
    gcbcg)      # graph + GCBC: the GCBC policy is given graph waypoints (GNODES: LeWM nodes, GWM: that LeWM) as its goal
                train gcbc  && { A="policy=bl_${ENV}_gcbc$MT +ff=true +graph_waypoints=true +graph_nodes=${GNODES:?set GNODES} +graph_wm=${GWM:?set GWM} +paa_span=${GSPAN:-6}"
                                 for d in $OFFS; do ev gcbcg${GTAG:-}_d${d}_s0 $d $A; done; xev gcbcg${GTAG:-} $A; } ;;
    pldm)       train pldm  && { for d in $OFFS; do ev pldm_d${d}_s0 $d policy=bl_${ENV}_pldm$MT; done; xev pldm policy=bl_${ENV}_pldm$MT; } ;;
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
