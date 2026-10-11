#!/bin/bash
set -euo pipefail
# Failure attribution on the 96 released Wall episodes, JEPA-WM then DINO-WM.
# Resumable: rerunning continues after the last finished episode of each model.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/failure_attribution
mkdir -p "$D"
for m in jepa dino; do
  python -m evals.failure_attribution.run --fname "$ROOT/configs/wall_${m}-wm_repro.yaml" \
    --out_dir "$D/wall_${m}_wm" --episodes 96 >> "$D/wall_${m}_wm.log" 2>&1
  python -m evals.failure_attribution.summarize --run_dir "$D/wall_${m}_wm" > "$D/wall_${m}_wm_summary.txt" 2>&1
done
