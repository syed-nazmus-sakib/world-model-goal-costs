#!/bin/bash
set -euo pipefail
# Released Push-T planning eval (CEM L2, alpha=0.1) on the first 24 episodes, for both models.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
mkdir -p "$JEPAWM_LOGS/pt_repro"
rm -f "$JEPAWM_LOGS/pt_repro/DONE"
for m in jepa-wm dino-wm; do
  python -m evals.main --fname "$ROOT/configs/pt_${m}_repro_ep24.yaml" --debug > "$JEPAWM_LOGS/pt_repro/${m}_ep24.log" 2>&1
  echo "$m exit $?" >> $JEPAWM_LOGS/pt_repro/status.txt
done
touch "$JEPAWM_LOGS/pt_repro/DONE"
