#!/bin/bash
set -euo pipefail
# Candidate collection for the Metaworld metric-conflict pilot, 4 tasks in parallel.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/mw_metric_conflict/data
mkdir -p "$D"
rm -f "$D/DONE"
for t in mw-reach mw-push mw-drawer-open mw-door-open; do
  python -m evals.mw_metric_conflict.collect --task $t --out_dir "$D" --num_episodes 60 --seed 0 > "$D/collect_$t.log" 2>&1 &
done
wait
touch "$D/DONE"
