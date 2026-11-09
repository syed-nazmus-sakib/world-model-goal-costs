#!/usr/bin/env bash
# exp-30, step 2 of 2: AntMaze-Large (legged robot) on GPU 0 only.  Trains LeWM, runs the kill test, then the long-range runs and
# every baseline (PLDM, DINO-WM, GCBC, GCIVL, GCIQL, HILP); see scripts/exp/exp30_antmaze.sh.
#
#   bash scripts/exp/exp30_antmaze_data.sh     # step 1, once: download + render the data (separate script)
#   bash scripts/hpc/run_antmaze.sh            # step 2: starts detached, the terminal can be closed
#   bash scripts/hpc/progress.sh out/exp30     # live view: epoch, step, loss and time left of the training; step and time
#                                              # left of every evaluation; finished results
#
# Run the same command again after a time limit or crash: finished trainings and runs are skipped.
# Result to send back: out/exp30/ALL_results.tgz.     QUICK=1 bash scripts/hpc/run_antmaze.sh = a short plumbing test.
set -uo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$H/../.." && pwd)"; cd "$ROOT"
SH="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
[[ -f "$SH/datasets/antmaze_large.h5" ]] || { echo "No $SH/datasets/antmaze_large.h5. Run first:  bash scripts/exp/exp30_antmaze_data.sh"; exit 1; }
OUT="${OUT:-out/exp30}"
GPU=0 PAIRS="exp-30:$OUT" NEED_GB="${NEED_GB:-30}" bash "$H/run_j17_j19.sh" || exit 1
echo
echo "Live progress:  bash scripts/hpc/progress.sh $OUT"
