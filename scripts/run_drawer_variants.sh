#!/usr/bin/env bash
set -euo pipefail
# drawer-close level 4: same drawer goal, arm moved to pose k (released latent cost, simulator dynamics).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/failure_attribution/drawer_close
mkdir -p "$D"
C=$ROOT/configs/mw_jepa-wm_drawer-close.yaml
for k in 2 3 4; do
  timeout 5400 python -m evals.failure_attribution.run_closed --fname "$C" --out_dir "$D/variant$k" --episodes 24 \
    --planners sim_latent --goal_variant $k >> "$D/variant$k.log" 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir "$D/variant$k" > "$D/variant${k}_summary.txt" 2>&1
done
