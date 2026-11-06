#!/bin/bash
# Gated goal handoff (G=12) on stitched goals, seeds 0/42: Two-Room budget 200, PointMaze-Medium budget 400.
# Fills Table 1, "+ goal handoff" stitched cells.
source /content/head.sh
log "GA_CHAIN_START T2 handoff stitched"
HO="+paa_handoff=true +paa_handoff_hops=12"
OFFS=25 bash $R tworoom 0 $O/hg_tworoom setup >> $O/laneA.log 2>&1
TAG=_hg12 EVX="$HO" XB=200 XM=ga bash $R tworoom "0 42" $O/hg_tworoom cross >> $O/laneA.log 2>&1
OFFS=50 bash $R pmm 0 $O/hg_pmm setup >> $O/laneA.log 2>&1
TAG=_hg12 EVX="$HO" XB=400 XM=ga bash $R pmm "0 42" $O/hg_pmm cross >> $O/laneA.log 2>&1
log "T2: $(cat $O/hg_*/summary.txt 2>/dev/null | grep -c success_rate) evaluations"
log "GA_CHAIN_DONE"
