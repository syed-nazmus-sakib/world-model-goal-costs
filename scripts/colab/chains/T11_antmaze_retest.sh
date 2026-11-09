#!/bin/bash
# Second plumbing test of exp-30 (AntMaze) after the fix of the policies' action size: 64 episodes of data, then only the parts
# the first test (T10) did not pass or did not have: GCBC, GCIVL, GCIQL, HILP in QUICK mode, and the random-action reference.
source /content/head.sh
log "GA_CHAIN_START T11 antmaze retest"
export GWM_NOCONDA=1 STABLEWM_HOME=/root/.stable_worldmodel PROGRESS_EVERY=5 MUJOCO_GL=egl
cd $JEPAWM_HOME
EPISODES=64 WORKERS=${WORKERS:-2} OUT=$O/data bash scripts/exp/exp30_antmaze_data.sh > $O/data.out 2>&1
log "data exit $?"
source scripts/hpc/egl.sh
TRAIN_WRAP=1 STEPS=30 STEPS_WM=30 EVX="eval.num_eval=2" EVSEEDS=0 XSEEDS=0 XB=100 OFFS=50 bash $B antl "gcbc gcivl gciql hilp" $O/bl > $O/bl.out 2>&1
log "baselines exit $? | $(grep -c 'success_rate=' $O/bl/summary.txt 2>/dev/null) results | $(grep -c 'rc=[1-9]\|FAILED' $O/bl/queue.log) failures"
( cd code/third_party/stable-worldmodel && python3 scripts/plan/eval_wm.py --config-name=antmaze_large policy=random seed=0 \
    eval.dataset_name=antmaze_large.h5 eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=2 ) > $O/random_d25.log 2>&1
log "random exit $? | $(grep -a '\[summary\]' $O/random_d25.log | grep -oE 'success_rate=[0-9.]+')"
ONCE=1 bash scripts/hpc/progress.sh $O > $O/progress_view.txt 2>&1
log "GA_CHAIN_DONE"
