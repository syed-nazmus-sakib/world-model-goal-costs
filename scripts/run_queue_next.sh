#!/usr/bin/env bash
set -euo pipefail
# Queue for the next GPU work, one job at a time (two world-model jobs do not fit in 32 GB together).
#   1) waits for the current runs (Wall DINO-WM attribution, Push-T alpha sweep) to finish
#   2) Push-T, real world-model planner at the best alpha, JEPA-WM and DINO-WM   (run_alpha_wm.sh, ~4 h)
#   3) Metaworld reach-wall closed-loop attribution, JEPA-WM then DINO-WM, 24 episodes each (~9 h)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/failure_attribution
mkdir -p "$D"
while pgrep -f "failure_attribution.run |alpha_sweep|run_attribution_wall" >/dev/null; do sleep 60; done
echo "$(date) queue: alpha world-model runs"
bash "$ROOT/scripts/run_alpha_wm.sh"
for m in jepa dino; do
  echo "$(date) queue: metaworld $m"
  python -m evals.failure_attribution.run_closed --fname "$ROOT/configs/mw_${m}-wm.yaml" \
    --out_dir "$D/mw_reach-wall_${m}_wm" --episodes ${MW_EPISODES:-24} >> "$D/mw_reach-wall_${m}_wm.log" 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir "$D/mw_reach-wall_${m}_wm" > "$D/mw_reach-wall_${m}_wm_summary.txt" 2>&1
done
echo "$(date) queue: done"
