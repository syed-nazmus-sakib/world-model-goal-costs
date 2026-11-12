#!/bin/bash
# Plumbing test of exp-33 (Metro-WM-style sub-goals, our re-implementation) before it goes to the cluster: Two-Room and
# PointMaze-Medium with the released LeWM models, 2 pairs, two (h, H) settings at a short distance, then the best one and the
# Metro graph with our search.  Checks that the graph is built on the real nodes, that every part runs and that the best setting
# is chosen from the sweep; the numbers mean nothing.  Results: /content/out/exp-33/<env> (queue.log, summary.txt, logs).
# Run on 2026-11-12 (T4, 12 GB host memory): Two-Room passed every part; PointMaze-Medium stopped in the node encoding, which was
# killed for memory while both environments were encoded at once.  The job now prepares one environment at a time.
source /content/head.sh
log "GA_CHAIN_START T14 metro test"
export GWM_NOCONDA=1 STABLEWM_HOME=/root/.stable_worldmodel PROGRESS_EVERY=5 MUJOCO_GL=egl
cd $JEPAWM_HOME; source scripts/hpc/egl.sh
QUICK=1 OUT_ROOT=$O bash scripts/hpc/run_job.sh exp-33 > $O/exp33.out 2>&1
log "exp-33 exit $? | $(grep -a 'JOB exp-33' $O/exp-33/job.log | tail -1 | cut -c21-)"
for e in tworoom pmm; do
  log "$e graph: $(grep -ah 'metro graph' $O/exp-33/$e/metro_*.log 2>/dev/null | tail -1 | cut -c1-330)"
  log "$e: best $(cat $O/exp-33/$e/best_cfg 2>/dev/null) | $(grep -c 'success_rate=' $O/exp-33/$e/summary.txt 2>/dev/null) result(s) | $(cut -d' ' -f1 $O/exp-33/$e/summary.txt 2>/dev/null | tr '\n' ' ')"
  log "$e timing: $(grep -ahoE 'ms_per_decision=[0-9.]+ decisions=[0-9]+' $O/exp-33/$e/metro_*.log 2>/dev/null | tr '\n' ';' | cut -c1-200) | $(grep -ah 'graph_timing' $O/exp-33/$e/metro_*.log 2>/dev/null | tail -1 | cut -c1-120)"
  log "$e errors: $(grep -alE 'Traceback|Error' $O/exp-33/$e/metro_*.log 2>/dev/null | tr '\n' ' ')"
done
log "GA_CHAIN_DONE"
