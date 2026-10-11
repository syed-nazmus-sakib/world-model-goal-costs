#!/usr/bin/env bash
# Reproduce the ctrlgeo kill tests. Run from the LeWM study folder.
#   bash run_all.sh            # everything (~1.5 h on RTX 5090 + 32 cores)
#   bash run_all.sh exp1       # ranking diagnostic only (~25 min)
#   bash run_all.sh exp2       # planning sweeps only
#   bash run_all.sh confirm    # held-out confirmation of the far-goal result (~2 h)
set -euo pipefail
cd "$(dirname "$0")"
source ../../env/lewm.sh || exit 1
PY="$(command -v python)"
export PYTHONPATH=$PWD
mkdir -p logs results/exp2
what=${1:-all}

if [[ $what == all || $what == exp1 ]]; then
  # Kill test 1: rank simulator-labelled candidate states by true continuation value
  "$PY" scripts/exp1_ranking.py --out results/exp1 --groups 300 --workers 30 2>&1 | tee logs/exp1.log
fi

if [[ $what == all || $what == exp2 ]]; then
  # Kill test 2a: standard LeWM PushT protocol (goal +25 steps, budget 50), matched WM compute
  "$PY" scripts/exp2_planning.py --budgets 200 400 1500 --seeds 0 1 2 \
    --methods l2 l2:H6 l2:H7 drift:K1 probemin:K1:M4 probemin:K1:M8 \
      cf_ridge:K1:M8:rho0.1 cf_ridge:K1:M8:rho0.01 cf_ridge:K2:M8:rho0.1 \
      l2+cf_perp:K1:M8:beta1 jac_ridge:K1:rho0.1 jac_ridge:K1:rho0.01 \
    --out results/exp2/sweep1.jsonl 2>&1 | tee logs/exp2_sweep1.log
  # Kill test 2b: far goals (goal +50 steps, budget 100), incl. top-R rerank variants
  "$PY" scripts/exp2_planning.py --offset 50 --eval_budget 100 --budgets 400 1500 --seeds 0 1 2 \
    --methods l2 l2:H8 drift:K1 probemin:K1:M8 cf_ridge:K1:M8:rho0.1 l2+cf_perp:K1:M8:beta1 \
      probemin:K1:M8:R10 cf_ridge:K1:M8:rho0.1:R10 probemin:K2:M16:R10 cf_ridge:K2:M16:rho0.1:R10 \
    --out results/exp2/sweep2_far.jsonl 2>&1 | tee logs/exp2_sweep2.log
  "$PY" scripts/summarize_exp2.py results/exp2/sweep1.jsonl results/exp2/sweep2_far.jsonl
fi

if [[ $what == confirm ]]; then
  # Held-out confirmation: fresh seeds 10-14 (never used during development), 100 episodes each,
  # far goals (+50 steps, budget 100), matched WM transitions per CEM iteration.
  "$PY" scripts/exp2_planning.py --offset 50 --eval_budget 100 --episodes 100 \
    --budgets 1500 3000 --seeds 10 11 12 13 14 \
    --methods l2 drift:K1 probemin:K1:M8 \
      cf_ridge:K1:M8:rho0.03 cf_ridge:K1:M8:rho0.1 cf_ridge:K1:M8:rho0.3 \
      cf_ridge:K1:M4:rho0.1 cf_ridge:K2:M8:rho0.1 \
    --out results/exp2/confirm_far.jsonl 2>&1 | tee logs/exp2_confirm.log
  "$PY" scripts/summarize_exp2.py results/exp2/confirm_far.jsonl
fi
