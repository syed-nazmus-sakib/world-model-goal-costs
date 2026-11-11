#!/usr/bin/env bash
# exp-31: PointMaze-Large on the official OGBench evaluation goals, on GPU 0 only.  No training and no download: it uses the
# PointMaze-Large data and the models already on this machine (J1 and the baselines); see scripts/exp/exp31_ogb_goals.sh.
#
#   bash scripts/hpc/run_ogb_goals.sh           # starts detached, the terminal can be closed
#   bash scripts/hpc/progress.sh out/exp31      # live view: step and time left of every evaluation; finished results
#   cat out/exp31/exp-31/ogb/table.txt          # the result so far: success per task and the mean, per method
#
# Run the same command again after a time limit or crash: finished runs are skipped.
# Result to send back: out/exp31/ALL_results.tgz.     QUICK=1 bash scripts/hpc/run_ogb_goals.sh = a short plumbing test.
set -uo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$H/../.." && pwd)"; cd "$ROOT"
SH="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
[[ -f "$SH/datasets/pointmaze_large.h5" ]] || { echo "No $SH/datasets/pointmaze_large.h5 (the PointMaze-Large data of job J1)."; exit 1; }
ls "$SH"/checkpoints/lewm_pml/*.pt > /dev/null 2>&1 || { echo "No model in $SH/checkpoints/lewm_pml (trained by job J1)."; exit 1; }
OUT="${OUT:-out/exp31}"
GPU=0 PAIRS="exp-31:$OUT" NEED_GB="${NEED_GB:-12}" bash "$H/run_j17_j19.sh" || exit 1
echo
echo "Live progress:  bash scripts/hpc/progress.sh $OUT"
echo "Result so far:  cat $OUT/exp-31/ogb/table.txt"
