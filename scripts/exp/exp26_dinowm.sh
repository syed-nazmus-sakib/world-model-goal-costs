#!/bin/bash
# exp-26: does the anchor idea transfer to a frozen pretrained representation?  DINO-WM + graph anchors.
# Graph on 4x4-pooled DINOv2 patches; CEM on DINO-WM with the anchor cost on the same pooled latents.  Slow: run on the HPC.
# Needs a trained bl_<env>_dwm (run_baselines.sh dwm) unless TRAIN=1.  Compare with DINO-WM alone at the same distances.
# QUICK=1: 100 training batches, 3000 nodes, 2 pairs: tests the code only.
# usage: [OFFS=..] [SEEDS=..] bash scripts/exp/exp26_dinowm.sh <pmm|tworoom> <out dir>
set -u
ENV=$1; D=$2; Q=${QUICK:-0}; mkdir -p "$D"
REPO=${JEPAWM_HOME:?}; PY=${SWM_PY:-python}; R=$REPO/scripts/pod/run_ga_env.sh; B=$REPO/scripts/pod/run_baselines.sh
P=$REPO/code/lewm_explorations/paa; SW=$REPO/code/third_party/stable-worldmodel
CK=${CK:-${STABLEWM_HOME:-$HOME/.stable_worldmodel}/checkpoints}; N=bl_${ENV}_dwm
case $ENV in pmm) KEY=proprio; OBJ=objective=goal_mse_pixels_proprio; OFF=${OFFS:-"100 200"} ;;
             tworoom) KEY=pos_agent; OBJ=objective=goal_mse_pixels_proprio; OFF=${OFFS:-"50 75"} ;; *) exit 1 ;; esac
SD=${SEEDS:-"0 42"}; EV="eval.num_eval=20 solver.batch_size=${DINO_BS:-10}"   # 20 pairs, as DINO-WM alone in Table 1
if [ "$Q" = 1 ]; then export STEPS_DWM=100 EPOCHS_DWM=1; SD=0; EV=eval.num_eval=2; OFF=${OFFS:-25}; fi
if [ ! -d "$CK/$N" ] || [ "${TRAIN:-0}" = 1 ]; then OFFS="" bash $B $ENV dwm $D/train; fi
O=$(find "$CK" -path "*$N*" -name model_object.pt | sort | tail -1); [ -n "$O" ] || { echo "no model_object.pt for $N"; exit 1; }
OFFS=25 bash $R $ENV 0 $D/lewm setup; DS=$(cat $D/lewm/dataset_name)
mkdir -p $D/a; echo $DS > $D/a/dataset_name
( cd $SW; $PY $P/exp_encode.py --out $D/a/nodes.npz --model $O --dataset $DS --state_key $KEY $([ "$Q" = 1 ] && echo --max_nodes 3000) ) || exit 1
touch $D/a/READY
WM=$N TAG=_dwm SPANS=6 EVX="+policy_object=$O $OBJ +paa_pool=true $EV" OFFS="$OFF" OKNODES=$D/lewm/nodes.npz bash $R $ENV "$SD" $D/a anchor
echo "exp26 $ENV done"
