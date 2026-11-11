#!/bin/bash
# Plumbing test of exp-32 (DINO-WM with an image-only goal cost, anchors on DINO-WM on PointMaze-Large, goal-target controls)
# before it goes to the cluster.  The trained models are only on the cluster, so this trains 30 batches of LeWM and, inside the
# job (QUICK=1), 100 batches of DINO-WM per environment, and plans 2 pairs at short distances.  The numbers mean nothing.
# Results: /content/out/exp-32 (queue.log, pml/summary.txt, ga_pml/a, ctl_tworoom/a, ctl_pmm/a).
source /content/head.sh
log "GA_CHAIN_START T13 dinowm fair test"
export GWM_NOCONDA=1 STABLEWM_HOME=/root/.stable_worldmodel PROGRESS_EVERY=5 MUJOCO_GL=egl RESERVE_GB=2
cd $JEPAWM_HOME; source scripts/hpc/egl.sh; P=$JEPAWM_HOME/code/lewm_explorations/paa
( cd code/third_party/stable-worldmodel && python3 $P/fetch_lewm_data.py pml > $O/fetch.log 2>&1
  log "fetch exit $? | $(ls -la $STABLEWM_HOME/datasets/ | grep -c pointmaze_large) file(s)"
  python3 $P/train_progress.py lewm_pml scripts/train/lewm.py data=pointmaze_large output_model_name=lewm_pml subdir=pml \
      trainer.max_epochs=1 +trainer.limit_train_batches=30 +trainer.limit_val_batches=5 > $O/train_lewm.log 2>&1
  log "train lewm_pml exit $?" )
QUICK=1 OUT_ROOT=$O DINO_GB=4 DINO_BS=2 bash scripts/hpc/run_job.sh exp-32 > $O/exp32.out 2>&1
log "exp-32 exit $? | $(grep -a 'JOB exp-32' $O/exp-32/job.log | tail -1 | cut -c21-)"
for f in pml/summary.txt ga_pml/a/summary.txt ctl_tworoom/a/summary.txt ctl_pmm/a/summary.txt; do
  log "$f: $(grep -c 'success_rate=' $O/exp-32/$f 2>/dev/null) result(s) | $(cut -d' ' -f1 $O/exp-32/$f 2>/dev/null | tr '\n' ' ')"
done
log "GA_CHAIN_DONE"
