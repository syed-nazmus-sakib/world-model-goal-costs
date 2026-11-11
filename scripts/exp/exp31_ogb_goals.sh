#!/bin/bash
# exp-31: PointMaze-Large on the official OGBench evaluation goals (the protocol of the published tables: 5 fixed start/goal
# cells with position noise, 50 episodes each, 1,000 steps, the environment's own goal tolerance).  No training: the LeWM of J1
# (lewm_pml) and the baselines already on disk.  Per task and seed, in this order (the important runs first):
#   ga        graph anchors (K=6, H=10, min-over-plan), graph over the whole training data
#   random    random actions: how much of a task is solved without a planner
#   released  LeWM as released (H=5, last step)      relmin  LeWM + min-over-plan (H=10)
#   gcbc gcivl gciql hilp pldm                        the policies / PLDM trained earlier on the same data, if their files exist
#   pldmga    graph anchors on the PLDM model (its own latents), if PLDM exists
#   DWM=1 adds DINO-WM (10 episodes per task, slow).  OGB_METHODS / OGB_SEEDS / OGB_TASKS choose a subset.
# Result: $D/table.txt (success per task and the mean over tasks, per method), $D/summary.txt.
# Finished runs are skipped, so the same command continues after a stop.  QUICK=1: 2 episodes, 100 steps (tests the code only).
# usage: bash scripts/exp/exp31_ogb_goals.sh <out dir>
set -u
D=$1; REPO=${JEPAWM_HOME:?}; R=$REPO/scripts/pod/run_ga_env.sh
SW=$REPO/code/third_party/stable-worldmodel; P=$REPO/code/lewm_explorations/paa; PY=${SWM_PY:-python}
SH=${STABLEWM_HOME:-$HOME/.stable_worldmodel}; CK=$SH/checkpoints; Q=${QUICK:-0}
q() { if [ "$Q" = 1 ]; then echo "$1"; else echo "$2"; fi; }
mkdir -p $D/graph
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$D/queue.log"; }
why() { grep -aE 'OutOfMemoryError|Error:|Error\b|Killed|Traceback' "$1" | grep -v '^ *File ' | tail -1 | cut -c1-200; }
[ -f "$SH/datasets/pointmaze_large.h5" ] || { log "exp31 FAILED: no $SH/datasets/pointmaze_large.h5 (the data of J1)"; exit 1; }
ls $CK/lewm_pml/*.pt > /dev/null 2>&1 || { log "exp31 FAILED: no LeWM in $CK/lewm_pml (trained by J1)"; exit 1; }
K=$CK/lewm_pml   # the loader needs exactly one .pt: keep the last epoch
if [ "$(ls $K/*.pt | wc -l)" -gt 1 ]; then mkdir -p $K/older_epochs; ls $K/*.pt | sort -V | head -n -1 | xargs -I{} mv {} $K/older_epochs/; fi
export PML_CKPT=lewm_pml MUJOCO_GL=egl PYOPENGL_PLATFORM=egl; echo pointmaze_large.h5 > $D/graph/dataset_name
OFFS=25 bash $R pml 0 $D/graph setup || { log "exp31 FAILED: graph setup"; exit 1; }
NODES=$D/graph/nodes.npz

E=$(q 2 50); BUD=$(q 100 1000); SEEDS="$(q 0 "${OGB_SEEDS:-0 1 2}")"; TASKS="${OGB_TASKS:-1 2 3 4 5}"; N=pointmaze_large_ogb
for s in $SEEDS; do
  [ -f "$SH/datasets/${N}_s${s}_t5.h5" ] && [ "$Q" != 1 ] && continue
  ( cd $SW && $PY $P/ogb_tasks_data.py --task large --out $N --episodes $E --seed $s --preview $D/goals_s$s.png ) > $D/tasks_s$s.log 2>&1 \
    || { log "exp31 FAILED: task files (seed $s): $(why $D/tasks_s$s.log)"; exit 1; }
  grep -a '^\[tasks\]' $D/tasks_s$s.log | head -1 | cut -c1-200 | while read -r l; do log "$l"; done
done

run() {  # name task seed overrides...
  local name=$1 k=$2 s=$3; shift 3
  grep -aq '\[summary\]' "$D/$name.log" 2>/dev/null && return
  log "$name start"
  ( cd $SW && $PY scripts/plan/eval_wm.py --config-name=pointmaze_large seed=$s eval.dataset_name=${N}_s${s}_t$k.h5 \
      eval.goal_offset_steps=1 eval.eval_budget=$BUD eval.num_eval=$E +stats_dataset=pointmaze_large.h5 "$@" ${EVX:-} ) > "$D/$name.log" 2>&1 \
    || log "$name FAILED: $(why "$D/$name.log")"
  grep -aq '\[summary\]' "$D/$name.log" && echo "$name $(grep -a '\[summary\]' "$D/$name.log")" >> "$D/summary.txt"
  log "$name: $(grep -a '\[summary\]' "$D/$name.log" | grep -oE 'success_rate=[0-9.]+')"
  $PY $P/ogb_table.py $D/summary.txt > $D/table.txt 2>/dev/null
}
GA="+paa_mode=fixed +paa_span=6 +metric_agg=min plan_config.horizon=10 +paa_noexclude=true"
pol() { echo "policy=bl_pml_$1_policy +ff=true +policy_object=$CK/bl_pml_$1_policy/model_object.pt"; }
method() {  # method task seed
  local m=$1 k=$2 s=$3 n=$1_t$2_s$3
  case $m in
    ga)       run $n $k $s policy=lewm_pml +paa_nodes=$NODES $GA ;;
    random)   run $n $k $s policy=random ;;
    released) run $n $k $s policy=lewm_pml ;;
    relmin)   run $n $k $s policy=lewm_pml +metric_head=released +metric_agg=min plan_config.horizon=10 ;;
    gcbc)     ls $CK/bl_pml_gcbc/*.pt > /dev/null 2>&1 && run $n $k $s policy=bl_pml_gcbc +ff=true ;;
    gcivl|gciql|hilp) [ -f $CK/bl_pml_${m}_policy/model_object.pt ] && run $n $k $s $(pol $m) ;;
    pldm)     ls $CK/bl_pml_pldm/*.pt > /dev/null 2>&1 && run $n $k $s policy=bl_pml_pldm ;;
    pldmga)   [ -f $D/graph_pldm/nodes.npz ] && run $n $k $s policy=bl_pml_pldm +paa_nodes=$D/graph_pldm/nodes.npz $GA ;;
    dwm)      O=$(find "$CK" -path '*bl_pml_dwm*' -name model_object.pt 2>/dev/null | sort | tail -1)
              [ -n "$O" ] && run $n $k $s policy=bl_pml_dwm +policy_object=$O objective=goal_mse_pixels_proprio eval.num_eval=$(q 2 10) solver.batch_size=${DINO_BS:-2} ;;
  esac
}
M1="${OGB_METHODS:-ga random released relmin}"; M2="${OGB_BASELINES:-gcbc gcivl gciql hilp pldm pldmga}"
for m in $M2; do case $m in
  gcbc|pldm) ls $CK/bl_pml_$m/*.pt > /dev/null 2>&1 || log "baseline $m: no checkpoint in $CK/bl_pml_$m, skipped" ;;
  gcivl|gciql|hilp) [ -f $CK/bl_pml_${m}_policy/model_object.pt ] || log "baseline $m: no $CK/bl_pml_${m}_policy/model_object.pt, skipped" ;;
esac; done
s0=${SEEDS%% *}
for m in $M1; do for k in $TASKS; do method $m $k $s0; done; done          # first seed, every task: the main numbers
if ls $CK/bl_pml_pldm/*.pt > /dev/null 2>&1 && echo " $M2 " | grep -q ' pldmga '; then   # graph on PLDM's own latents
  mkdir -p $D/graph_pldm; echo pointmaze_large.h5 > $D/graph_pldm/dataset_name
  WM=bl_pml_pldm OFFS=25 bash $R pml 0 $D/graph_pldm setup || log "pldmga: graph setup failed, skipped"
fi
for m in $M2; do for k in $TASKS; do method $m $k $s0; done; done
for s in $SEEDS; do [ "$s" = "$s0" ] && continue                             # the other noise seeds
  for m in $M1 $M2; do for k in $TASKS; do method $m $k $s; done; done
done
if [ "${DWM:-0}" = 1 ]; then source $REPO/scripts/exp/gpu_wait.sh; for k in $TASKS; do wait_vram 18; method dwm $k $s0; done; fi
$PY $P/ogb_table.py $D/summary.txt | tee $D/table.txt
echo "exp31 done"
