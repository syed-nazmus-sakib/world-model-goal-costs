#!/usr/bin/env bash
set -euo pipefail
# Metaworld push (small puck, not the robot): simulator dynamics, same 24 episodes, only the cost changes.
#   l2 (released, alpha 0.1) + true-state oracle | l2 with alpha 10 | top-8 | top-8 with alpha 10
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/failure_attribution/mw_push_costs
mkdir -p "$D"
run() {  # name planners cost alpha
  python -m evals.failure_attribution.run_closed --fname "$ROOT/configs/mw_jepa-wm_push.yaml" --out_dir "$D/$1" \
    --episodes ${MW_EPISODES:-24} --planners $2 --cost $3 --alpha $4 >> "$D/$1.log" 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir "$D/$1" > "$D/$1_summary.txt" 2>&1
}
run l2_a0.1 "sim_latent sim_true" l2 0.1
run top8_a0.1 sim_latent topk:8 0.1
run l2_a10 sim_latent l2 10
run top8_a10 sim_latent topk:8 10
