#!/bin/bash
# exp-27: appearance shift at test time.  Every frame gets a warm tint; the graph stays on the original latents.
# Does the nearest-node lookup still work?  Direct planner vs graph anchors at the longest distance, tint 0.1 0.2 0.4.
# Killed as a limitation if anchors stay above the direct planner at every strength.  QUICK=1: one strength, 2 pairs.
# usage: bash scripts/exp/exp27_shift.sh <pmm|tworoom> <out dir>
set -u
ENV=$1; D=$2; Q=${QUICK:-0}; mkdir -p "$D"
REPO=${JEPAWM_HOME:?}; R=$REPO/scripts/pod/run_ga_env.sh
case $ENV in pmm) OFF=200 ;; tworoom) OFF=75 ;; *) exit 1 ;; esac
if [ "$Q" = 1 ]; then Ss=0.2; SEEDS=0; EV=eval.num_eval=2; else Ss="0.1 0.2 0.4"; SEEDS=${SEEDS:-"0 42"}; EV=""; fi
OFFS=25 bash $R $ENV 0 $D setup
for s in $Ss; do
  TAG=_px$s EVX="+pix_shift=$s $EV" OFFS=$OFF SPANS=6 bash $R $ENV "$SEEDS" $D all
done
echo "exp27 $ENV done"
