#!/usr/bin/env bash
# exp-33: Metro-WM-style sub-goals (arXiv 2609.31868, our re-implementation) on our tasks, next to graph anchors: PointMaze-Medium,
# Two-Room, PointMaze-Large and Push-T, one lane per environment on GPU 0.  No training and no download: the models and data
# already on this machine (the earlier jobs).  See scripts/exp/exp33_metro.sh.
#
#   bash scripts/hpc/run_metro.sh               # starts detached, the terminal can be closed
#   bash scripts/hpc/progress.sh out/exp33      # live view
#   grep -a " S sweep\| M \| X " out/exp33/exp-33/*/queue.log    # results so far
#
# Run the same command again after a time limit or crash: finished runs are skipped.
# Result to send back: out/exp33/ALL_results.tgz.
# Options:  METRO_ENVS="pmm tworoom" (a subset of pmm tworoom pml pusht)   PARTS="S" (sweep only; default "S M X")
#           QUICK=1 bash scripts/hpc/run_metro.sh = a short plumbing test (2 pairs, Two-Room and PointMaze-Medium).
set -uo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$H/../.." && pwd)"; cd "$ROOT"
OUT="${OUT:-out/exp33}"
# RERUN=1: the job may be started again with more environments or parts after an earlier start has ended
RERUN=1 GPU=0 PAIRS="exp-33:$OUT" NEED_GB="${NEED_GB:-30}" bash "$H/run_j17_j19.sh" || exit 1
echo
echo "Live progress:  bash scripts/hpc/progress.sh $OUT"
echo "Results so far: grep -a ' S sweep\| M \| X ' $OUT/exp-33/*/queue.log"
