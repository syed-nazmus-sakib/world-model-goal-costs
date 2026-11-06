#!/bin/bash
# exp-21: does the best anchor span K follow the planning horizon H?
# Hypothesis: the planner does not need the cost to rank well at the anchor, only that rollouts reach the region near it,
# so the best K grows with H.  Killed if the best K stays at 6 for H in {5, 10, 15}.
# K in 2 4 6 8 12, H in 5 10 15, seeds SEEDS, at the longest distance of the env.  QUICK=1: K 4 6, H 5 10, seed 0, 2 pairs.
# usage: bash scripts/exp/exp21_kxh.sh <pmm|tworoom> <out dir>
set -u
ENV=$1; D=$2; Q=${QUICK:-0}; mkdir -p "$D"
REPO=${JEPAWM_HOME:?}; R=$REPO/scripts/pod/run_ga_env.sh
case $ENV in pmm) OFF=200 ;; tworoom) OFF=75 ;; *) echo "unknown env $ENV"; exit 1 ;; esac
if [ "$Q" = 1 ]; then Ks="4 6"; Hs="5 10"; SEEDS=0; EV=eval.num_eval=2; else Ks="2 4 6 8 12"; Hs="5 10 15"; SEEDS=${SEEDS:-"0 42"}; EV=""; fi
OFFS=25 bash $R $ENV 0 $D setup
for h in $Hs; do
  TAG=_kh$h HA=$h EVX="$EV" OFFS=$OFF SPANS="$Ks" bash $R $ENV "$SEEDS" $D anchor
done
echo "exp21 $ENV done"
