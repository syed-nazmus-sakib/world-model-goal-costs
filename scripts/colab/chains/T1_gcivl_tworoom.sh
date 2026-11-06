#!/bin/bash
# GCIVL on Two-Room (one epoch of 5,000 batches, as the other baselines): seeds 0/42/1 at d25/50/75, stitched b200 seeds 0/42.
# Reruns the November 1 evaluation whose result files were lost. Fills Table 1, Two-Room GCIVL.
source /content/head.sh
log "GA_CHAIN_START T1 gcivl tworoom"
EVSEEDS='0 42 1' XSEEDS='0 42' XB=200 OFFS='25 50 75' bash $B tworoom gcivl $O/bl_tworoom >> $O/laneA.log 2>&1
log "T1: $(grep -c success_rate $O/bl_tworoom/summary.txt 2>/dev/null) evaluations"
log "GA_CHAIN_DONE"
