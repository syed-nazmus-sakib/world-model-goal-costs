#!/bin/bash
# Plumbing test of exp-30 (AntMaze) before it goes to the cluster: 64 episodes of data, then the whole job in QUICK mode
# (30 training batches per model, 2 start/goal pairs per evaluation).  Checks that every stage starts, logs progress and ends;
# the numbers mean nothing.  Results: /content/out/data (log + preview.png) and /content/out/exp-30.
source /content/head.sh
log "GA_CHAIN_START T10 antmaze test"
export GWM_NOCONDA=1 STABLEWM_HOME=/root/.stable_worldmodel PROGRESS_EVERY=5
cd $JEPAWM_HOME
EPISODES=64 WORKERS=${WORKERS:-2} OUT=$O/data bash scripts/exp/exp30_antmaze_data.sh > $O/data.out 2>&1
log "data exit $? | $(grep -a '^\[render\]' $O/data.out | tail -1 | cut -c1-150) | $(grep -a '^\[check\]' $O/data.out | cut -c1-110)"
QUICK=1 OUT_ROOT=$O DINO_GB=4 bash scripts/hpc/run_job.sh exp-30 > $O/exp30.out 2>&1
log "exp-30 exit $? | $(grep -a 'JOB exp-30' $O/exp-30/job.log | tail -1 | cut -c21-)"
ONCE=1 bash scripts/hpc/progress.sh $O/exp-30 > $O/progress_view.txt 2>&1
log "GA_CHAIN_DONE"
