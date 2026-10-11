#!/bin/bash
set -euo pipefail
# Archival check: higher-capacity global vs per-task metrics.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/mw_metric_conflict
mkdir -p "$D"
rm -f "$D/HIGHCAP_DONE"
until [ -f "$JEPAWM_LOGS/pt_repro/DONE" ]; do sleep 20; done
python -m evals.mw_metric_conflict.fit_highcap --data_dir "$D/data" --emb_dir "$D/emb_jepa_wm" --out "$D/fit_highcap_jepa_wm.json" > "$D/fit_highcap_jepa.log" 2>&1
echo "exit $?" > $D/highcap_status.txt
touch "$D/HIGHCAP_DONE"
