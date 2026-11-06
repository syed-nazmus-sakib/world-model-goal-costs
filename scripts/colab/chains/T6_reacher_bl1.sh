#!/bin/bash
# Reacher baselines, part 1: GCBC, GCIQL, PLDM (one epoch, as elsewhere), seeds 0/42 at d25/50/100/150. Fills Table 1, Reacher.
source /content/head.sh
log "GA_CHAIN_START T6 reacher gcbc gciql pldm"
EVSEEDS='0 42' OFFS='25 50 100 150' bash $B reacher "gcbc gciql pldm" $O/bl_reacher >> $O/laneA.log 2>&1
log "T6: $(grep -c success_rate $O/bl_reacher/summary.txt 2>/dev/null) evaluations; $(grep -cE 'FAILED|rc=[1-9]' $O/bl_reacher/queue.log) failures"
log "GA_CHAIN_DONE"
