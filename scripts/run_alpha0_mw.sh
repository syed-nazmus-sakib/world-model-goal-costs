#!/usr/bin/env bash
set -euo pipefail
# Metaworld push, simulator dynamics, latent cost WITHOUT the proprio goal term (alpha 0), 24 episodes:
#   robot_free_a0: goal + candidates rendered without the robot | robot_a0: robot visible
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/failure_attribution/goal_equiv_closed
mkdir -p "$D"
for name in robot_free_a0 robot_a0; do
  extra=""; [ $name = robot_free_a0 ] && extra="--robot_free"
  timeout 3000 python -m evals.failure_attribution.run_closed --fname "$ROOT/configs/mw_jepa-wm_push.yaml" --out_dir "$D/$name" \
    --episodes 24 --planners sim_latent --alpha 0 $extra >> "$D/$name.log" 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir "$D/$name" > "$D/${name}_summary.txt" 2>&1
done
