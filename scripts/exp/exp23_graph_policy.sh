#!/bin/bash
# exp-23 / 24 / 25: graph methods without a world model, on our pairs.  Same scorer as Table 1.
#   gcbch gcivlh hilph : our latent graph, a nearer waypoint held until reached (exp-23)
#   ttgs               : TTGS-style, graph + subgoals from GCIVL's value, GCIVL follows (exp-24)
#   gas                : GAS-style, graph + subgoals from HILP's phi, HILP follows (exp-25)
# Compare with graph anchors at the same distances.  Killed if none beats the same policy without a graph.
# QUICK=1: 200 training batches, one distance, 2 pairs: tests the code only, its numbers mean nothing.
# usage: [METHODS="gcbch ttgs"] [OFFS=...] bash scripts/exp/exp23_graph_policy.sh <pmm|tworoom|pml> <out dir>
set -u
ENV=$1; D=$2; Q=${QUICK:-0}; mkdir -p "$D"
REPO=${JEPAWM_HOME:?}; R=$REPO/scripts/pod/run_ga_env.sh; B=$REPO/scripts/pod/run_baselines.sh
case $ENV in pmm) GWM=yjang43/lp2-pointmaze-medium ;; tworoom) GWM=quentinll/lewm-tworooms ;; pml) GWM=${PML_CKPT:?set PML_CKPT} ;; esac
OFFS=25 bash $R $ENV 0 $D/graph setup
export GNODES=$D/graph/nodes.npz GWM METHODS=${METHODS:-gcbch gcivlh hilph ttgs gas}
NV=""; V=""; for m in $METHODS; do case $m in ttgs|gas) V="$V $m" ;; *) NV="$NV $m" ;; esac; done
if [ "$Q" = 1 ]; then
  export STEPS=200 EVX=eval.num_eval=2 VGN=300 EVSEEDS=0
  OFFS=${OFFS:-50} bash $B $ENV "$METHODS" $D
else
  export EVSEEDS=${EVSEEDS:-"0 42 1"} XSEEDS=${XSEEDS:-"0 42"} VGN=${VGN:-3000}
  case $ENV in tworoom) export XB=200 ;; *) export XB=400 ;; esac
  # On a shared GPU a run can die when other jobs fill the memory.  Finished runs are skipped, so each pass repeats only
  # what failed; a pass waits until the GPU has EXP23_GB free (default 10) and the loop stops after a pass with no failure.
  source $REPO/scripts/exp/gpu_wait.sh
  for pass in 1 2 3 4 5 6; do
    n0=$(cat $D/queue.log 2>/dev/null | wc -l)
    wait_vram ${EXP23_GB:-10}
    [ -z "$NV" ] || bash $B $ENV "$NV" $D
    # the edge limit tau of TTGS and GAS is tuned per environment: pick by seed 0 at the middle distance, report the other seeds
    [ -z "$V" ] || for t in ${VGTAUS:-2 4 8}; do GTAG=_t$t VGTAU=$t bash $B $ENV "$V" $D; done
    tail -n +$((n0 + 1)) $D/queue.log 2>/dev/null | grep -q FAILED || break
    echo "exp23 $ENV: pass $pass had failures, repeating them in 10 min"; sleep 600
  done
fi
echo "exp23 $ENV done"
