#!/bin/bash
set -euo pipefail
# Failure attribution on the 96 released Push-T episodes, JEPA-WM then DINO-WM.
# Resumable: rerunning continues after the last finished episode of each model.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/pusht_failure_attribution
mkdir -p "$D"
for m in jepa dino; do
  python -m evals.pusht_failure_attribution.run --fname "$ROOT/configs/pt_${m}-wm_repro.yaml" \
    --out_dir "$D/${m}_wm" --episodes 96 >> "$D/${m}_wm.log" 2>&1
  python -m evals.pusht_failure_attribution.summarize --run_dir "$D/${m}_wm" > "$D/${m}_wm_summary.txt" 2>&1
done
