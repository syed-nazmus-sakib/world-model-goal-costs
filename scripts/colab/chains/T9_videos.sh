#!/bin/bash
# Planner-view videos (Prof. Cao's question): LeWM (released) vs Graph Anchors on the same pairs, seed 0, with per-step traces
# (agent state, anchors) so markers can be drawn on the frames.  Push-T d150, Two-Room d75, Reacher d150.
source /content/head.sh
log "GA_CHAIN_START T9 videos"
OFFS=50 bash $R pusht 0 $O/v_pusht setup >> $O/laneA.log 2>&1 &
OFFS=25 bash $R tworoom 0 $O/v_tworoom setup >> $O/laneB.log 2>&1 &
OFFS=50 bash $R reacher 0 $O/v_reacher setup >> $O/laneC.log 2>&1 &
wait
pack() { (cd $O && tar cf vids.tmp v_*/trace_*.npz v_*/vid_* 2>/dev/null && mv vids.tmp vids.tar); }
( while sleep 600; do pack; done ) & PK=$!
V="VID=1 TRACE=1 TRACE_EVERY=1 SK_MIN=1 SPANS=6"
env $V OFFS=150 bash $R pusht 0 $O/v_pusht all >> $O/laneA.log 2>&1 & P1=$!
( env $V OFFS=75 bash $R tworoom 0 $O/v_tworoom all >> $O/laneB.log 2>&1 ) & P2=$!
env $V OFFS=150 bash $R reacher 0 $O/v_reacher all >> $O/laneC.log 2>&1 & P3=$!
wait $P1 $P2 $P3
kill $PK; pack
log "T9: $(cat $O/v_*/summary.txt 2>/dev/null | grep -c success_rate) evaluations, $(find $O -name '*.mp4' | wc -l) videos"
log "GA_CHAIN_DONE"
