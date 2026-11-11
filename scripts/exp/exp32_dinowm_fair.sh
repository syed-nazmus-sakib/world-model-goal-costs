#!/bin/bash
# exp-32: is DINO-WM's 100% on PointMaze-Large a like-for-like comparison, and do anchors help DINO-WM there?
# DINO-WM's goal cost contains the agent's position; LeWM and Graph Anchors see the goal image only.  exp-29 (20 pairs, one
# seed) gave 70% with an image-only cost.  No training: the DINO-WM and LeWM models already on disk.  In this order:
#   A  PointMaze-Large d400, DINO-WM with the image-only goal cost, 50 pairs (the pairs of Table 1), seeds 0 42 1
#   C  PointMaze-Large d400, DINO-WM + graph anchors on its own pooled latents (K=6), and the same search with the goal as
#      target (span beyond every path): 20 pairs, seed 0
#   B  the goal-target control of the exp-26 anchor runs: Two-Room d75 and PointMaze-Medium d200, 20 pairs, seed 0
#   A2 as A on stitched goals (budget 400), seeds 0 42
# PARTS chooses a subset (default "A C B A2").  Finished runs are skipped, so the same command continues after a stop.
# QUICK=1: 2 pairs, short distances, 100 training batches where a model is missing (tests the code only).
# usage: bash scripts/exp/exp32_dinowm_fair.sh <out dir>
set -u
D=$1; REPO=${JEPAWM_HOME:?}; B=$REPO/scripts/pod/run_baselines.sh; E26=$REPO/scripts/exp/exp26_dinowm.sh
CK=${CK:-${STABLEWM_HOME:-$HOME/.stable_worldmodel}/checkpoints}; Q=${QUICK:-0}
q() { if [ "$Q" = 1 ]; then echo "$1"; else echo "$2"; fi; }
mkdir -p $D/pml; log() { echo "$(date -u +%F' '%T) $*" | tee -a "$D/queue.log"; }
source $REPO/scripts/exp/gpu_wait.sh
# planning 10 pairs at a time needs about 45 GB of GPU memory and is several times faster than 2 at a time (18 GB)
FREE=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%d", $1 / 1024}')
DBS=${DINO_BS:-$([ "${FREE:-0}" -ge 50 ] && echo 10 || echo 2)}; DGB=${DINO_GB:-$([ "$DBS" -ge 10 ] && echo 45 || echo 18)}
log "DINO-WM: ${FREE:-?} GB free on the GPU -> $DBS pairs per planning batch (waits for $DGB GB free)"
export DINO_BS=$DBS DINO_GB=$DGB
if [ "$Q" = 1 ]; then export STEPS_DWM=100 EPOCHS_DWM=1
else  # no training here: say so at once if a model is missing
  [ -n "$(find "$CK" -path '*bl_pml_dwm*' -name model_object.pt 2>/dev/null)" ] || { log "exp32 FAILED: no DINO-WM for PointMaze-Large under $CK (bl_pml_dwm, trained by J19)"; exit 1; }
  export EPOCHS_DWM=5 STEPS_DWM=5000
fi
[ -n "$(find "$CK" -path '*bl_pml_dwm*' -name model_object.pt 2>/dev/null)" ] && touch $D/pml/train_dwm.done
PIX="DWM_OBJ=objective=goal_mse_pixels GTAG=_pix50"; EV="eval.num_eval=$(q 2 50) solver.batch_size=$DBS"
n_ok() { cat "$@" 2>/dev/null | grep -c 'success_rate='; }
for part in ${PARTS:-A C B A2}; do case $part in
  A)  for pass in 1 2 3 4; do   # a pass repeats only what ran out of memory on a shared GPU
        wait_vram $DGB; env $PIX EVX="$EV" EVSEEDS="$(q 0 '0 42 1')" OFFS=$(q 50 400) bash $B pml dwm $D/pml
        [ "$(grep -c '^dwm_pix50_d' $D/pml/summary.txt 2>/dev/null)" -ge "$(q 1 3)" ] && break; sleep $(q 5 600)
      done
      log "A image-only cost, d$(q 50 400): $(grep -a '^dwm_pix50_d' $D/pml/summary.txt 2>/dev/null | grep -oE 'success_rate=[0-9.]+' | tr '\n' ' ')" ;;
  C)  SP="6 100000" bash $E26 pml $D/ga_pml >> $D/lane_ga_pml.log 2>&1
      log "C anchors on DINO-WM, PointMaze-Large: $(grep -a 'success_rate' $D/ga_pml/a/queue.log 2>/dev/null | cut -c21- | tr '\n' ';')" ;;
  B)  for e in tworoom pmm; do
        SP=100000 bash $E26 $e $D/ctl_$e >> $D/lane_ctl_$e.log 2>&1
        log "B goal-target control, $e: $(grep -a 'success_rate' $D/ctl_$e/a/queue.log 2>/dev/null | cut -c21- | tr '\n' ';')"
      done ;;
  A2) for pass in 1 2 3 4; do
        wait_vram $DGB; env $PIX EVX="$EV" XSEEDS="$(q 0 '0 42')" XB=$(q 50 400) OFFS="" bash $B pml dwm $D/pml
        [ "$(grep -c '^xdwm_pix50_b' $D/pml/summary.txt 2>/dev/null)" -ge "$(q 1 2)" ] && break; sleep $(q 5 600)
      done
      log "A2 image-only cost, stitched: $(grep -a '^xdwm_pix50_b' $D/pml/summary.txt 2>/dev/null | grep -oE 'success_rate=[0-9.]+' | tr '\n' ' ')" ;;
esac; done
echo "exp32 done"
