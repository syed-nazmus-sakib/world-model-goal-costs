#!/bin/bash
# exp-28: the cost alone, with exact dynamics and exact search (exp_exact.py).  CPU after the nodes exist; uses exp-20's output.
# usage: bash scripts/exp/exp28_exact.sh <exp-20 out dir of the env>
set -u
D=$1; REPO=${JEPAWM_HOME:?}; PY=${SWM_PY:-python}; P=$REPO/code/lewm_explorations/paa
L=""; for m in lewm pldm dinov2 dwm; do [ -f $D/${m}_e.npz ] && L="$L $m=$D/${m}_e.npz"; done
$PY $P/exp_exact.py --out $D/exact.json --pairs ${PAIRS:-300} $L
