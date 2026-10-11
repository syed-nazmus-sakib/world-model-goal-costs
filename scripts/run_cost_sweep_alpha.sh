#!/usr/bin/env bash
set -euo pipefail
# Push-T, simulator dynamics: does top-8 add on top of a tuned proprio weight?
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
mkdir -p "$JEPAWM_LOGS/pusht_failure_attribution" "$JEPAWM_LOGS/failure_attribution"
for a in 1 10; do
  python -m evals.pusht_failure_attribution.cost_sweep --fname "$ROOT/configs/pt_jepa-wm_repro.yaml" \
    --run_dir "$JEPAWM_LOGS/pusht_failure_attribution/jepa_wm" --costs topk:8 --alpha $a \
    >> "$JEPAWM_LOGS/pusht_failure_attribution/cost_sweep_alpha_jepa.log" 2>&1
done
