#!/usr/bin/env bash
# Direction A K1 (Cube): data -> train (resume/retry) -> CF + planning eval.
set -uo pipefail
cd "$(dirname "$0")/.."
source ../../env/lewm.sh || exit 1
PY="$(command -v python)"
export PYTHONPATH=$PWD MUJOCO_GL=egl
mkdir -p logs
L=logs/exp6_pipeline.log
[ -f results/exp6/data/RND4.npz ] || "$PY" scripts/exp6_cube.py data >> $L 2>&1
[ -f results/exp6/cf_groundtruth.npz ] || "$PY" scripts/exp6_cube.py cfbuild >> $L 2>&1
train() {
  local d=results/exp6/models/$1_s$2
  for try in 1 2 3 4; do
    [ -f $d/weights.pt ] && return 0
    "$PY" -m ctrlgeo.train --data results/exp6/data/$1.npz --out $d --seed $2 >> logs/train6_$1_s$2.log 2>&1 \
      || echo "train $1 s$2 attempt $try failed" >> $L
  done
}
for s in 0 1; do
  train EXP $s & train RND1 $s & wait
  train RND4 $s
  for n in EXP RND1 RND4; do
    m=results/exp6/models/${n}_s$s
    "$PY" scripts/exp6_cube.py cf --models $m >> $L 2>&1
    "$PY" scripts/exp6_cube.py plan --models $m --seeds 0 1 2 >> $L 2>&1
  done
  echo "EVAL_DONE seed $s" >> $L
done
echo PIPELINE_DONE >> $L
