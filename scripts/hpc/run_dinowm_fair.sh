#!/usr/bin/env bash
# exp-32: DINO-WM with an image-only goal cost on PointMaze-Large (50 pairs, three seeds), graph anchors on DINO-WM there, and
# the goal-target control of the earlier DINO-WM anchor runs.  GPU 0 only, no training; see scripts/exp/exp32_dinowm_fair.sh.
#
#   bash scripts/hpc/run_dinowm_fair.sh         # starts detached, the terminal can be closed
#   bash scripts/hpc/progress.sh out/exp32      # live view
#   grep -a success_rate out/exp32/exp-32/queue.log     # results so far
#
# Run the same command again after a time limit or crash: finished runs are skipped.
# Result to send back: out/exp32/ALL_results.tgz.     PARTS="A" bash scripts/hpc/run_dinowm_fair.sh = the first part only.
set -uo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$H/../.." && pwd)"; cd "$ROOT"
SH="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
[[ -f "$SH/datasets/pointmaze_large.h5" ]] || { echo "No $SH/datasets/pointmaze_large.h5 (the PointMaze-Large data of job J1)."; exit 1; }
[[ -n "$(find "$SH/checkpoints" -path '*bl_pml_dwm*' -name model_object.pt 2>/dev/null)" ]] || { echo "No DINO-WM for PointMaze-Large in $SH/checkpoints (bl_pml_dwm, trained by job J19)."; exit 1; }
OUT="${OUT:-out/exp32}"
GPU=0 PAIRS="exp-32:$OUT" NEED_GB="${NEED_GB:-18}" bash "$H/run_j17_j19.sh" || exit 1
echo
echo "Live progress:  bash scripts/hpc/progress.sh $OUT"
echo "Results so far: grep -a success_rate $OUT/exp-32/queue.log"
