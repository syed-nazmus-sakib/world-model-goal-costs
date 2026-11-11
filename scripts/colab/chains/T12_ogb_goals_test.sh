#!/bin/bash
# Plumbing test of exp-31 (PointMaze-Large on the official OGBench goals) before it goes to the cluster.  The trained model is
# only on the cluster, so this trains 30 batches of LeWM, GCBC, GCIVL and PLDM and runs the job in QUICK mode (2 episodes per
# task, 100 steps).  Checks that the task files are built from the installed OGBench, that every method starts on them and that
# the table is written; the numbers mean nothing.  Results: /content/out/exp-31/ogb (table.txt, goals_s0.png, logs).
source /content/head.sh
log "GA_CHAIN_START T12 ogb goals test"
export GWM_NOCONDA=1 STABLEWM_HOME=/root/.stable_worldmodel PROGRESS_EVERY=5 MUJOCO_GL=egl
cd $JEPAWM_HOME; source scripts/hpc/egl.sh; P=$JEPAWM_HOME/code/lewm_explorations/paa
( cd code/third_party/stable-worldmodel && python3 $P/fetch_lewm_data.py pml > $O/fetch.log 2>&1
  log "fetch exit $? | $(ls -la $STABLEWM_HOME/datasets/ | grep -c pointmaze_large) file(s)"
  python3 $P/train_progress.py lewm_pml scripts/train/lewm.py data=pointmaze_large output_model_name=lewm_pml subdir=pml \
      trainer.max_epochs=1 +trainer.limit_train_batches=30 +trainer.limit_val_batches=5 > $O/train_lewm.log 2>&1
  log "train lewm_pml exit $?" )
TRAIN_WRAP=1 STEPS=30 STEPS_WM=30 EVX="eval.num_eval=2" EVSEEDS=0 OFFS=50 bash $B pml "gcbc gcivl pldm" $O/bl > $O/bl.out 2>&1
log "baselines exit $? | $(grep -c 'success_rate=' $O/bl/summary.txt 2>/dev/null) results | $(grep -c 'rc=[1-9]\|FAILED' $O/bl/queue.log) failures"
QUICK=1 OUT_ROOT=$O bash scripts/hpc/run_job.sh exp-31 > $O/exp31.out 2>&1
log "exp-31 exit $? | $(grep -a 'JOB exp-31' $O/exp-31/job.log | tail -1 | cut -c21-)"
log "tasks: $(grep -a '^\[tasks\]' $O/exp-31/ogb/tasks_s0.log | head -1 | cut -c1-220)"
ONCE=1 bash scripts/hpc/progress.sh $O/exp-31 > $O/progress_view.txt 2>&1
log "GA_CHAIN_DONE"
