#!/bin/bash
# exp-26: does the anchor idea transfer to a frozen pretrained representation?  DINO-WM + graph anchors.
# Graph on 4x4-pooled DINOv2 patches; CEM on DINO-WM with the anchor cost on the same pooled latents.  Slow: run on the HPC.
# Needs a trained bl_<env>_dwm (run_baselines.sh dwm) unless TRAIN=1.  Compare with DINO-WM alone at the same distances.
# QUICK=1: 100 training batches, 3000 nodes, 2 pairs: tests the code only.
# Default: longest distance, seed 0, the setting of DINO-WM alone in Table 1 (about 12 h; a CEM solve takes minutes per pair).
# SP: anchor spans (default 6).  A span beyond every path (SP=100000) makes the anchor the goal itself: the same search and
# the same pooled latents with the goal as target, which is the min-over-plan control of the anchor run (exp-32).
# usage: [OFFS=..] [SEEDS=..] [SP=..] bash scripts/exp/exp26_dinowm.sh <pmm|tworoom|pml> <out dir>
set -u
ENV=$1; D=$2; Q=${QUICK:-0}; mkdir -p "$D"
REPO=${JEPAWM_HOME:?}; PY=${SWM_PY:-python}; R=$REPO/scripts/pod/run_ga_env.sh; B=$REPO/scripts/pod/run_baselines.sh
P=$REPO/code/lewm_explorations/paa; SW=$REPO/code/third_party/stable-worldmodel
CK=${CK:-${STABLEWM_HOME:-$HOME/.stable_worldmodel}/checkpoints}; N=bl_${ENV}_dwm
case $ENV in pmm) KEY=proprio; OBJ=objective=goal_mse_pixels_proprio; OFF=${OFFS:-200} ;;
             tworoom) KEY=pos_agent; OBJ=objective=goal_mse_pixels_proprio; OFF=${OFFS:-75} ;;
             pml) KEY=state; OBJ=objective=goal_mse_pixels_proprio; OFF=${OFFS:-400}; export PML_CKPT=${PML_CKPT:-lewm_pml} ;; *) exit 1 ;; esac
# batch 2 (pairs solved together; 300 CEM samples each, unchanged): estimated peak about 15 GB; at batch 10 it needed about 49 GB, which
# ran out of memory twice on a GPU shared with other jobs (exp-26b, 2026-11-08).  Only the order of random draws changes.
SD=${SEEDS:-0}; EV="eval.num_eval=20 solver.batch_size=${DINO_BS:-2}"   # 20 pairs, as DINO-WM alone in Table 1
if [ "$Q" = 1 ]; then export STEPS_DWM=100 EPOCHS_DWM=1; SD=0; EV=eval.num_eval=2; OFF=${OFFS:-25}; fi
if [ ! -d "$CK/$N" ] || [ "${TRAIN:-0}" = 1 ]; then OFFS="" bash $B $ENV dwm $D/train; fi
O=$(find "$CK" -path "*$N*" -name model_object.pt | sort | tail -1); [ -n "$O" ] || { echo "no model_object.pt for $N"; exit 1; }
if [ $ENV = pml ]; then   # our own LeWM and data (job J1): no download, and the loader needs exactly one .pt
  mkdir -p $D/lewm; echo pointmaze_large.h5 > $D/lewm/dataset_name; K=$CK/$PML_CKPT
  if [ "$(ls $K/*.pt 2>/dev/null | wc -l)" -gt 1 ]; then mkdir -p $K/older_epochs; ls $K/*.pt | sort -V | head -n -1 | xargs -I{} mv {} $K/older_epochs/; fi
fi
OFFS=25 bash $R $ENV 0 $D/lewm setup; DS=$(cat $D/lewm/dataset_name)
mkdir -p $D/a; echo $DS > $D/a/dataset_name
[ -f $D/a/nodes.npz ] || ( cd $SW; $PY $P/exp_encode.py --out $D/a/nodes.npz --model $O --dataset $DS --state_key $KEY $([ "$Q" = 1 ] && echo --max_nodes 3000) ) || exit 1   # kept from an earlier run
touch $D/a/READY
source $REPO/scripts/exp/gpu_wait.sh
# Each planning run claims its GPU memory as soon as it starts (eval_wm.py, RESERVE_GB), so other users of the shared GPU
# cannot take it during the minutes the model and graph load.  24 GB covers batch 2 with room to spare (estimate).
export RESERVE_GB=${RESERVE_GB:-24}
for pass in 1 2 3 4 5 6 7 8 9 10; do   # finished runs are skipped; a pass only repeats what ran out of memory on a shared GPU
  wait_vram ${DINO_GB:-26}
  WM=$N TAG=_dwm SPANS="${SP:-6}" EVX="+policy_object=$O $OBJ +paa_pool=true $EV" OFFS="$OFF" OKNODES=$D/lewm/nodes.npz bash $R $ENV "$SD" $D/a anchor
  [ "$(grep -c 'success_rate=' $D/a/summary.txt 2>/dev/null)" -ge "$(( $(echo $OFF | wc -w) * $(echo $SD | wc -w) * $(echo ${SP:-6} | wc -w) ))" ] && break
  sleep 600
done
echo "exp26 $ENV done"
