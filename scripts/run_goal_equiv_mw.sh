#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
mkdir -p "$JEPAWM_LOGS/failure_attribution"
python -m evals.failure_attribution.goal_equiv_mw --fname "$ROOT/configs/mw_jepa-wm_push.yaml" \
  --out_dir "$JEPAWM_LOGS/failure_attribution/goal_equiv_mw_push" --episodes 24 > "$JEPAWM_LOGS/failure_attribution/goal_equiv_mw_push.log" 2>&1
