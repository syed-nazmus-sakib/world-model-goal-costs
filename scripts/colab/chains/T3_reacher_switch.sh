#!/bin/bash
# Episode-switch cost lambda=3 on Reacher, seed 42, d100/d150 (seed 0: 78/64). Graph nodes are uploaded (prebuilt).
# Then a quick plumbing test of the Reacher baselines (run_baselines.sh reacher) before J19 goes to the HPC.
source /content/head.sh
log "GA_CHAIN_START T3 reacher switch s42"
bash $R reacher 0 $O/hg_reacher setup >> $O/laneA.log 2>&1
TAG=_sw3 EVX="+paa_switch=3" OFFS="100 150" SPANS=6 bash $R reacher 42 $O/hg_reacher anchor >> $O/laneA.log 2>&1
log "T3: $(grep -c success_rate $O/hg_reacher/summary.txt 2>/dev/null) evaluations"
# plumbing test of the Reacher baseline port used by HPC job J19 (30 training batches, 2 episodes per evaluation)
STEPS=30 STEPS_WM=30 EVX="eval.num_eval=2" EVSEEDS=0 OFFS=25 bash $B reacher "gcbc gcivl gciql hilp pldm" $O/q_bl_reacher >> $O/laneQ.log 2>&1
EPOCHS_DWM=1 STEPS_DWM=20 EVX="eval.num_eval=2" EVSEEDS=0 OFFS=25 bash $B reacher dwm $O/q_dwm_reacher >> $O/laneQ.log 2>&1
log "Q: $(cat $O/q_*/summary.txt 2>/dev/null | grep -c success_rate)/6 reacher baseline evals ran; failures: $(cat $O/q_*/queue.log 2>/dev/null | grep -cE 'FAILED|rc=[1-9]')"
log "GA_CHAIN_DONE"
