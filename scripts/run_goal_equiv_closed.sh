#!/usr/bin/env bash
set -euo pipefail
# Metaworld push, simulator dynamics, released latent cost, 24 episodes each:
#   L4: same puck goal, arm moved to pose k=1..4 (variant 0 = eval goal = mw_push_costs/l2_a0.1: 2/24)
#   oracle: goal and candidate frames rendered without the robot
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/failure_attribution/goal_equiv_closed
mkdir -p "$D"
run() {  # name extra-args
  python -m evals.failure_attribution.run_closed --fname "$ROOT/configs/mw_jepa-wm_push.yaml" --out_dir "$D/$1" \
    --episodes ${MW_EPISODES:-24} --planners sim_latent $2 >> "$D/$1.log" 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir "$D/$1" > "$D/$1_summary.txt" 2>&1
}
run robot_free "--robot_free"
for k in 1 2 3 4; do run variant$k "--goal_variant $k"; done
