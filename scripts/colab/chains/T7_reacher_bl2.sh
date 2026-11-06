#!/bin/bash
# Reacher baselines, part 2: GCIVL and HILP (value then policy), seeds 0/42 at d25/50/100/150.
source /content/head.sh
log "GA_CHAIN_START T7 reacher gcivl hilp"
EVSEEDS='0 42' OFFS='25 50 100 150' bash $B reacher "gcivl hilp" $O/bl_reacher >> $O/laneA.log 2>&1
log "T7: $(grep -c success_rate $O/bl_reacher/summary.txt 2>/dev/null) evaluations; $(grep -cE 'FAILED|rc=[1-9]' $O/bl_reacher/queue.log) failures"
log "GA_CHAIN_DONE"
