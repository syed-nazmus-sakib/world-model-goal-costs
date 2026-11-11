#!/bin/bash
# exp-30: a legged robot at long range.  OGBench AntMaze-Large from pixels (quadruped, 8 joint torques, goals up to 400 steps away).
#   1  train LeWM on the rendered data (10 epochs x 3,000 batches, as on PointMaze-Large)
#   2  kill test: LeWM at 25 and 50 steps (released planner, min-over-plan, graph anchors; random actions as the reference),
#      seed 0.  If the planner cannot reach goals 25 steps away (best of the two LeWM planners below KILL, default 30%, or less than
#      KILL_GAP, default 15, points above random actions), anchors
#      cannot help either and the experiment stops here.  FORCE=1 continues anyway.
#   3  LeWM, LeWM + min-over-plan, graph anchors at 100/200/400 steps and on stitched goals (budget 400), seeds 0/42
#   4  baselines trained on the same data, same pairs: GCBC, GCIVL, GCIQL, HILP, PLDM; DINO-WM with its own goal cost (pixels +
#      proprio, which holds the goal position) and with the pixel-only cost (20 pairs, seed 0); graph anchors on the PLDM model
# Every training prints [train] / [epoch] lines with the time left, every evaluation [progress] lines:
#   bash scripts/hpc/progress.sh <out dir>
# Finished trainings and runs are skipped, so the same command continues after a stop.
# Data first: bash scripts/exp/exp30_antmaze_data.sh.   usage: bash scripts/exp/exp30_antmaze.sh <out dir>
set -u
D=$1; REPO=${JEPAWM_HOME:?}; R=$REPO/scripts/pod/run_ga_env.sh; B=$REPO/scripts/pod/run_baselines.sh
SW=$REPO/code/third_party/stable-worldmodel; P=$REPO/code/lewm_explorations/paa; PY=${SWM_PY:-python}
SH=${STABLEWM_HOME:-$HOME/.stable_worldmodel}; CK=$SH/checkpoints; Q=${QUICK:-0}
q() { if [ "$Q" = 1 ]; then echo "$1"; else echo "$2"; fi; }
mkdir -p $D/train $D/ant $D/bl
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$D/queue.log"; }
[ -f "$SH/datasets/antmaze_large.h5" ] || { log "exp30 FAILED: no $SH/datasets/antmaze_large.h5; run  bash scripts/exp/exp30_antmaze_data.sh  first"; exit 1; }

# ---- 1. LeWM
if [ ! -f $D/train/train.done ]; then
  log "train lewm_antl start (progress: grep '\[train\]' $D/train/train_lewm.log | tail -1)"; t0=$(date +%s)
  ( cd $SW && $PY $P/train_progress.py lewm_antl scripts/train/lewm.py data=antmaze_large output_model_name=lewm_antl subdir=antl \
      trainer.max_epochs=$(q 1 ${EP:-10}) +trainer.limit_train_batches=$(q 30 ${NB:-3000}) +trainer.limit_val_batches=$(q 5 20) ) > $D/train/train_lewm.log 2>&1
  rc=$?; log "train lewm_antl rc=$rc $(( $(date +%s) - t0 ))s $(grep -aE '^[A-Za-z.]*(Error|Exception)' $D/train/train_lewm.log | tail -1 | cut -c1-160)"
  [ $rc = 0 ] || { log "exp30 FAILED: LeWM training"; exit 1; }
  touch $D/train/train.done
fi
K=$CK/lewm_antl   # one .pt per epoch; the loader needs exactly one: keep the last
if [ "$(ls $K/*.pt 2>/dev/null | wc -l)" -gt 1 ]; then mkdir -p $K/older_epochs; ls $K/*.pt | sort -V | head -n -1 | xargs -I{} mv {} $K/older_epochs/; fi
export ANTL_CKPT=lewm_antl; echo antmaze_large.h5 > $D/ant/dataset_name

# reference: random actions on the same pairs (no model), so that pairs solved by standing still are not read as planning
rnd() {  # offset
  local name=random_d$1_s0
  grep -aq '\[summary\]' "$D/ant/$name.log" 2>/dev/null && return
  ( cd $SW && $PY scripts/plan/eval_wm.py --config-name=antmaze_large policy=random seed=0 eval.dataset_name=antmaze_large.h5 \
      eval.goal_offset_steps=$1 eval.eval_budget=$((2 * $1)) ${EVX:-} ) > "$D/ant/$name.log" 2>&1
  grep -aq '\[summary\]' "$D/ant/$name.log" && echo "$name $(grep -a '\[summary\]' "$D/ant/$name.log")" >> "$D/ant/summary.txt"
  log "$name: $(grep -a '\[summary\]' "$D/ant/$name.log" | grep -oE 'success_rate=[0-9.]+')"
}

# ---- 2. kill test
OFFS="$(q 25 '25 50')" SPANS=6 bash $R antl 0 $D/ant all
for d in $(q 25 '25 50'); do rnd $d; done
rate() { grep -a "^$1 " $D/ant/summary.txt 2>/dev/null | grep -oE 'success_rate=[0-9.]+' | tail -1 | cut -d= -f2; }
a=$(rate released_h5_d25_s0); b=$(rate relmin_h10_d25_s0); g=$(rate ga_sp6_d25_s0); r0=$(rate random_d25_s0)
best=$(awk -v a="${a:-0}" -v b="${b:-0}" 'BEGIN {print (a > b) ? a : b}')
log "KILL TEST at 25 steps: LeWM ${a:-?}%, LeWM + min-over-plan ${b:-?}%, graph anchors ${g:-?}%, random actions ${r0:-?}% (needed: a LeWM planner at ${KILL:-30}% or more and at least ${KILL_GAP:-15} points above random actions)"
# both conditions: on 2026-11-10 the planners reached 24-32% and random actions 32%, which passed the old rule (30% alone)
if [ "$Q" != 1 ] && [ "${FORCE:-0}" != 1 ] && awk -v x="$best" -v k="${KILL:-30}" -v r="${r0:-0}" -v gp="${KILL_GAP:-15}" 'BEGIN {exit !(x < k || x < r + gp)}'; then
  log "KILL TEST NOT PASSED: the planner does not reach goals 25 steps away better than random actions, so the long-range runs are skipped (FORCE=1 runs them anyway)"
  touch $D/KILLED; echo "exp30 done (stopped at the kill test)"; exit 0
fi
rm -f $D/KILLED; log "KILL TEST PASSED"

# ---- 3. long range and stitched goals
S="$(q 0 '0 42')"; OF="$(q 50 "${ANT_OFFS:-100 200 400}")"; XBUD=$(q 100 400)
OFFS="$OF" SPANS=6 bash $R antl "$S" $D/ant all
for d in $OF; do rnd $d; done
XB=$XBUD XM="released relmin ga" bash $R antl "$S" $D/ant cross

# ---- 4. baselines on the same data and pairs
export TRAIN_WRAP=1
[ "$Q" = 1 ] && export STEPS=30 STEPS_WM=30
EVSEEDS="$S" XSEEDS="$S" XB=$XBUD OFFS="$OF" bash $B antl "${ANT_BL:-gcbc gcivl gciql hilp pldm}" $D/bl
if [ -f $D/bl/train_pldm.done ]; then   # graph anchors on PLDM's own latents
  mkdir -p $D/pldm_ant; echo antmaze_large.h5 > $D/pldm_ant/dataset_name
  WM=bl_antl_pldm OFFS="$(q 50 '200 400')" SPANS=6 bash $R antl 0 $D/pldm_ant all
fi
# DINO-WM: slow planning (patch tokens), so 20 pairs and seed 0.  Its own cost, then pixels only.  Planning 10 pairs at a time
# needs about 45 GB of GPU memory and is several times faster than 2 at a time (18 GB), so the batch follows the free memory.
source $REPO/scripts/exp/gpu_wait.sh
export EPOCHS_DWM=$(q 1 5) STEPS_DWM=$(q 20 5000)
FREE=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%d", $1 / 1024}')
DBS=${DINO_BS:-$([ "${FREE:-0}" -ge 50 ] && echo 10 || echo 2)}; DGB=${DINO_GB:-$([ "$DBS" -ge 10 ] && echo 45 || echo 18)}
log "DINO-WM: ${FREE:-?} GB free on the GPU -> $DBS pairs per planning batch (waits for $DGB GB free)"
DEV="eval.num_eval=$(q 2 20) solver.batch_size=$DBS"
for pass in 1 2 3; do   # finished runs are skipped: a pass only repeats what ran out of memory on a shared GPU
  wait_vram $DGB; EVX="$DEV" EVSEEDS=0 XSEEDS=0 XB=$XBUD OFFS="$(q 50 '200 400')" bash $B antl dwm $D/bl
  wait_vram $DGB; EVX="$DEV" EVSEEDS=0 XSEEDS=0 XB=$XBUD OFFS="$(q 50 400)" DWM_OBJ=objective=goal_mse_pixels GTAG=_pix bash $B antl dwm $D/bl
  n=$(grep -ac '^x\?dwm' $D/bl/summary.txt 2>/dev/null); [ "${n:-0}" -ge "$(q 4 5)" ] && break
  log "DINO-WM pass $pass: ${n:-0} runs finished, repeating the others"; sleep 300
done
echo "exp30 done"
