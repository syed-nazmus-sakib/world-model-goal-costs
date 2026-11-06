#!/bin/bash
# Plumbing test before HPC job J19: HILP on Reacher (no proprio column) and DINO-WM with batched CEM (2 episodes at once).
source /content/head.sh
log "GA_CHAIN_START T4 reacher test"
STEPS=30 EVX="eval.num_eval=2" EVSEEDS=0 OFFS=25 bash $B reacher hilp $O/q_hilp >> $O/laneQ.log 2>&1
EPOCHS_DWM=1 STEPS_DWM=20 EVX="eval.num_eval=2 solver.batch_size=2" EVSEEDS=0 OFFS=25 bash $B reacher dwm $O/q_dwm >> $O/laneQ.log 2>&1
log "Q: $(cat $O/q_*/summary.txt 2>/dev/null | grep -c success_rate)/2 evals; failures: $(cat $O/q_*/queue.log 2>/dev/null | grep -cE 'FAILED|rc=[1-9]')"
log "GA_CHAIN_DONE"
