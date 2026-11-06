#!/usr/bin/env bash
# The exp-NN experiments, each in its own output folder, on whichever GPUs have free memory right now.
#
#   bash scripts/hpc/run_exps.sh                       # exp-21, 22, 23, 26, 27
#   PLAN_ONLY=1 bash scripts/hpc/run_exps.sh           # only show which experiment would go to which GPU
#   PAIRS="exp-23:out/exp23" bash scripts/hpc/run_exps.sh   # a subset
#
# Runs detached: the terminal can be closed. Run the same command again after a time limit or crash, finished runs are
# skipped. Progress: tail -f out/exp23/run_all.log    Results: out/expNN/ALL_results.tgz, one file per experiment.
# GPU assignment and restart behaviour are those of run_j17_j19.sh; NEED_GB is the memory reserved per experiment.
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PAIRS="${PAIRS:-exp-21:out/exp21 exp-22:out/exp22 exp-23:out/exp23 exp-26:out/exp26 exp-27:out/exp27}" NEED_GB="${NEED_GB:-30}"
exec bash "$H/run_j17_j19.sh"
