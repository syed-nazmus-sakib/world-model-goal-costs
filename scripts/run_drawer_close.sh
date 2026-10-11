#!/usr/bin/env bash
set -euo pipefail
# Metaworld drawer-close (large articulated object), simulator dynamics, 24 episodes each.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/failure_attribution/drawer_close
mkdir -p "$D"
C=$ROOT/configs/mw_jepa-wm_drawer-close.yaml
timeout 3600 python -m evals.failure_attribution.goal_equiv_mw --fname "$C" --out_dir "$D/goal_equiv" --episodes 24 > "$D/goal_equiv.log" 2>&1
run() {  # name extra-args
  timeout 5400 python -m evals.failure_attribution.run_closed --fname "$C" --out_dir "$D/$1" --episodes 24 $2 >> "$D/$1.log" 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir "$D/$1" > "$D/$1_summary.txt" 2>&1
}
run released "--planners sim_latent sim_true"
run robot_free_a0 "--planners sim_latent --robot_free --alpha 0"
