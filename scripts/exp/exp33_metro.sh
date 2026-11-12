#!/bin/bash
# exp-33: Metro-WM-style sub-goals (Lee et al., arXiv 2609.31868) on our tasks, next to graph anchors.
# Metro-WM also retrieves sub-goals for a frozen LeWM planner from a graph of recorded frames, and reports goals up to 75 steps
# away.  This runs our re-implementation of its sub-goal rule (code/lewm_explorations/paa/metro_graph.py: bridges to the 4 nearest
# frames of other episodes inside a calibrated radius, route costs in frames, the farthest route vertex within a budget of H
# frames) with its planner (a plan of h action blocks scored at its last step, all h blocks executed before the next plan) on the
# same nodes, pairs and seeds as our runs.  No training: the models and data already on disk.
# Metro-WM tunes (h, H) per benchmark on the evaluation pairs; it gets the same here:
#   S  sweep of (h, H) at one distance of the env, seed 0:  MCFG (default "1:5 1:10 3:15 3:20 5:25 5:30"; their grid is
#      h in 1..5 and H in [5h-5, 5h+5])
#   M  the best (h, H) of S (ties: the shorter plan), seeds 0 42 1, every distance of the env
#   X  the Metro graph and sub-goal rule with OUR search (10 blocks, minimum over the plan), budget 30 frames = six nodes,
#      seeds 0 42 at the sweep distance: separates the graph and sub-goal rule from the search
# PARTS chooses a subset (default "S M X").  Finished runs are skipped, so the same command continues after a stop.
# QUICK=1: 2 pairs, two (h, H) settings, short distances, seed 0 (tests the code only).
# usage: bash scripts/exp/exp33_metro.sh <pmm|tworoom|pml|pusht> <out dir>
set -u
ENV=$1; D=$2; Q=${QUICK:-0}; mkdir -p "$D"
REPO=${JEPAWM_HOME:?}; R=$REPO/scripts/pod/run_ga_env.sh
q() { if [ "$Q" = 1 ]; then echo "$1"; else echo "$2"; fi; }
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$D/queue.log"; }
case $ENV in   # SW: distance of the sweep; ALL: distances of the main runs
  pmm)     SW=$(q 50 200); ALL=$(q 50 "50 100 150 200") ;;
  tworoom) SW=$(q 25 75);  ALL=$(q 25 "25 50 75") ;;
  pml)     SW=$(q 50 400); ALL=$(q 50 "200 400") ;;
  pusht)   SW=$(q 50 75);  ALL=$(q 50 "50 75 150") ;;   # 75: the distance Metro-WM reports
  *) echo "unknown env $ENV"; exit 1 ;;
esac
CFGS=${MCFG:-$(q "1:10 3:15" "1:5 1:10 3:15 3:20 5:25 5:30")}; EV=$(q eval.num_eval=2 "")
case " ${EVX:-} " in *" eval.num_eval="*) EV="" ;; esac   # the caller already set the number of pairs
OFFS=25 bash $R $ENV 0 $D setup || { log "exp33 $ENV: setup FAILED"; exit 1; }
best() {  # the (h, H) with the highest success in the sweep; ties: the shorter plan, then the smaller budget
  grep -a "^metro_h[0-9]*_b[0-9]*_d${SW}_s0 " "$D/summary.txt" 2>/dev/null \
    | sed -E 's/^metro_h([0-9]+)_b([0-9]+)_.*success_rate=([0-9.]+).*/\3 \1 \2/' | sort -k1,1nr -k2,2n -k3,3n | head -1 | awk '{print $2 ":" $3}'
}
for part in ${PARTS:-S M X}; do case $part in
  S) MCFG="$CFGS" EVX="${EVX:-} $EV" OFFS=$SW bash $R $ENV 0 $D metro
     log "S sweep at d$SW (h:H=success): $(grep -a "^metro_h[0-9]*_b[0-9]*_d${SW}_s0 " "$D/summary.txt" 2>/dev/null \
        | sed -E 's/^metro_h([0-9]+)_b([0-9]+)_.*success_rate=([0-9.]+).*/\1:\2=\3/' | tr '\n' ' ')" ;;
  M) B=${MBEST:-$(best)}
     [ -n "$B" ] || { log "M skipped: no finished sweep run in $D/summary.txt (run part S first, or set MBEST=h:H)"; continue; }
     echo "$B" > "$D/best_cfg"; log "M best (h:H) = $B"
     MCFG="$B" EVX="${EVX:-} $EV" OFFS="$ALL" bash $R $ENV "$(q 0 '0 42 1')" $D metro
     log "M Metro-WM-style h:H=$B: $(grep -a "^metro_h${B%%:*}_b${B##*:}_d" "$D/summary.txt" 2>/dev/null \
        | sed -E 's/^metro_h[0-9]+_b[0-9]+_(d[0-9]+_s[0-9]+) .*success_rate=([0-9.]+).*/\1=\2/' | tr '\n' ' ')" ;;
  X) MCFG="ours:30" EVX="${EVX:-} $EV" OFFS=$SW bash $R $ENV "$(q 0 '0 42')" $D metro
     log "X Metro graph + our search: $(grep -a '^metro_hours_b30_d' "$D/summary.txt" 2>/dev/null \
        | sed -E 's/^metro_hours_b30_(d[0-9]+_s[0-9]+) .*success_rate=([0-9.]+).*/\1=\2/' | tr '\n' ' ')" ;;
esac; done
echo "exp33 $ENV done"
