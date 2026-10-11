#!/usr/bin/env bash
set -euo pipefail
# Push-T, simulator dynamics, released episodes: area-invariant patch aggregations of the latent cost
# (vision only, training-free) vs the released mean-over-patches L2 (= sim_latent of run.py, 62/96).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
mkdir -p "$JEPAWM_LOGS/pusht_failure_attribution" "$JEPAWM_LOGS/failure_attribution"
python -m evals.pusht_failure_attribution.cost_sweep --fname "$ROOT/configs/pt_jepa-wm_repro.yaml" \
  --run_dir "$JEPAWM_LOGS/pusht_failure_attribution/jepa_wm" --costs topk:8 topk:4 pmean:4 \
  >> "$JEPAWM_LOGS/pusht_failure_attribution/cost_sweep_jepa.log" 2>&1
