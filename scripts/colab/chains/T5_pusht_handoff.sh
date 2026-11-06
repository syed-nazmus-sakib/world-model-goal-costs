#!/bin/bash
# Gated goal handoff (G=12) on Push-T d100/d150, seeds 0/42. Fills Table 1, "+ goal handoff" Push-T cells.
source /content/head.sh
log "GA_CHAIN_START T5 pusht handoff"
OFFS=50 bash $R pusht 0 $O/hg_pusht setup >> $O/laneA.log 2>&1
TAG=_hg12 EVX="+paa_handoff=true +paa_handoff_hops=12" OFFS="100 150" SPANS=6 bash $R pusht "0 42" $O/hg_pusht anchor >> $O/laneA.log 2>&1
log "T5: $(grep -c success_rate $O/hg_pusht/summary.txt 2>/dev/null) evaluations"
log "GA_CHAIN_DONE"
