#!/bin/bash
# exp-22: can more search fix the distant goal?  Direct goal cost (min over the plan), horizon x samples.
# Killed as an argument for anchors if the largest budget reaches the success of anchors at the default budget.
# H in 10 20, samples in 300 1000, elites 10%, at the longest distance.  QUICK=1: one setting, 2 pairs.
# usage: bash scripts/exp/exp22_search.sh <pmm|tworoom> <out dir>
set -u
ENV=$1; D=$2; Q=${QUICK:-0}; mkdir -p "$D"
REPO=${JEPAWM_HOME:?}; R=$REPO/scripts/pod/run_ga_env.sh
case $ENV in pmm) OFF=200 ;; tworoom) OFF=75 ;; *) echo "unknown env $ENV"; exit 1 ;; esac
if [ "$Q" = 1 ]; then Hs=10; Ss=300; SEEDS=0; EV=eval.num_eval=2; else Hs="10 20"; Ss="300 1000"; SEEDS=${SEEDS:-"0 42"}; EV=""; fi
OFFS=25 bash $R $ENV 0 $D setup
for h in $Hs; do for s in $Ss; do
  TAG=_srch${s} HA=$h EVX="solver.num_samples=$s solver.topk=$((s / 10)) $EV" OFFS=$OFF SK_REL=1 bash $R $ENV "$SEEDS" $D base
done; done
echo "exp22 $ENV done"
