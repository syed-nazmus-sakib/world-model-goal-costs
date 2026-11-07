#!/bin/bash
# Rollout traces for the drift test on Push-T (d150) and Two-Room (d75): released, min-over-plan, anchors, and anchors + switch cost on Push-T.
# Hypothesis: anchors make more decisions progress than the plain goal target, far from the goal, as on PM-M.
# Kill: on an environment where the Table 1 gain is real, anchors do not beat released on share of decisions closer: no such claim for that env.
# pull() skips *.npz, so traces and the node index (ep, step, state: anchor_log indices refer to it) travel in traces.tar.
source /content/head.sh
log "GA_CHAIN_START T8 traces"
OFFS=50 bash $R pusht 0 $O/tr_pusht setup >> $O/laneA.log 2>&1
OFFS=25 bash $R tworoom 0 $O/tr_tworoom setup >> $O/laneB.log 2>&1
python3 -c "
import numpy as np
for e in ('pusht', 'tworoom'):
    d = np.load(f'$O/tr_{e}/nodes.npz'); np.savez_compressed(f'$O/tr_{e}/trace_nodeidx.npz', ep=d['ep'], step=d['step'], state=d['state'])
"
pack() { (cd $O && tar cf traces.tmp tr_*/trace_*.npz 2>/dev/null && mv traces.tmp traces.tar); }
( while sleep 600; do pack; done ) & PK=$!
pt() { TRACE=1 OFFS=150 SPANS=6 bash $R pusht "$1" $O/tr_pusht all >> $O/lane$2.log 2>&1
       TRACE=1 TAG=_sw3 EVX="+paa_switch=3" OFFS=150 SPANS=6 bash $R pusht "$1" $O/tr_pusht anchor >> $O/lane$2.log 2>&1; }
pt 0 A & P1=$!
pt 42 C & P2=$!
TRACE=1 OFFS=75 SPANS=6 bash $R tworoom "0 42 1" $O/tr_tworoom all >> $O/laneB.log 2>&1 & P3=$!
wait $P1 $P2 $P3
kill $PK; pack
log "T8: $(cat $O/tr_*/summary.txt 2>/dev/null | grep -c success_rate) evaluations, $(tar tf $O/traces.tar | wc -l) files"
log "GA_CHAIN_DONE"
