#!/usr/bin/env bash
# Idea 3 step 2 pipeline: train fixed-budget LeWMs (resume + retry on crash), then evals.
#   SEEDS="0 1" bash scripts/exp5_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source ../../env/lewm.sh || exit 1
PY="$(command -v python)"
export PYTHONPATH=$PWD
mkdir -p logs

train() {  # name seed
  local d=results/exp5/models/$1_s$2
  for try in 1 2 3 4; do
    [ -f $d/weights.pt ] && return 0
    "$PY" -m ctrlgeo.train --data results/exp5/data/$1.npz --out $d --seed $2 >> logs/train_$1_s$2.log 2>&1 \
      || echo "train $1 s$2 attempt $try failed (exit $?)" >> logs/exp5_pipeline.log
  done
}

evals() {  # name seed
  local m=results/exp5/models/$1_s$2
  [ -f $m/weights.pt ] || { echo "missing $m" >> logs/exp5_eval.log; return; }
  grep -q "\"$m\"" results/exp5/cf_eval.jsonl 2>/dev/null \
    || "$PY" scripts/exp5_cf_eval.py --models $m >> logs/exp5_eval.log 2>&1
  grep -q "\"$m\"" results/exp5/planning.jsonl 2>/dev/null \
    || "$PY" scripts/exp2_planning.py --model $m --methods l2 --budgets 1500 --seeds 0 1 2 \
         --out results/exp5/planning.jsonl >> logs/exp5_eval.log 2>&1
}

while pgrep -f "ctrlgeo.train" > /dev/null; do sleep 20; done   # let running jobs finish
for s in ${SEEDS:-0 1}; do
  train EXP $s & train RND1 $s & wait
  train RND4 $s & train RND8 $s & wait
  for n in EXP RND1 RND4 RND8; do evals $n $s; done
  echo "EVAL_DONE seed $s" >> logs/exp5_eval.log
done
echo PIPELINE_DONE >> logs/exp5_eval.log
