#!/usr/bin/env bash
set -euo pipefail
# Resume the two runs interrupted by the move (both skip finished episodes / alphas).
#   1) Push-T alpha sweep with the simulator as dynamics (JEPA-WM encoding)
#   2) Wall failure attribution, JEPA-WM then DINO-WM (96 episodes each)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
mkdir -p "$JEPAWM_LOGS/pusht_failure_attribution" "$JEPAWM_LOGS/failure_attribution"
nohup python -m evals.pusht_failure_attribution.alpha_sweep --fname "$ROOT/configs/pt_jepa-wm_repro.yaml" \
  --run_dir "$JEPAWM_LOGS/pusht_failure_attribution/jepa_wm" --alphas 0 0.1 1 10 100 \
  >> "$JEPAWM_LOGS/pusht_failure_attribution/alpha_sweep_jepa.log" 2>&1 &
nohup bash "$ROOT/scripts/run_attribution_wall.sh" > "$JEPAWM_LOGS/failure_attribution/wall_driver.log" 2>&1 &
echo "launched; follow with:"
echo "  tail -f $JEPAWM_LOGS/pusht_failure_attribution/alpha_sweep_jepa.log"
echo "  tail -f $JEPAWM_LOGS/failure_attribution/wall_jepa_wm.log"
