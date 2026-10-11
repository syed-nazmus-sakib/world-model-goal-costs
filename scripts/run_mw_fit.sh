#!/bin/bash
set -euo pipefail
# Encode Metaworld candidates and fit E0-E3 metrics, for both released encodings.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
D=$JEPAWM_LOGS/mw_metric_conflict
mkdir -p "$D"
rm -f "$D/FIT_DONE"
for m in jepa dino; do
  [ -f "$D/emb_${m}_wm/reach.npz" ] || python -m evals.mw_metric_conflict.encode --config "$ROOT/configs/mw_${m}-wm.yaml" --model_name ${m}_wm_metaworld --data_dir "$D/data" --out_dir "$D/emb_${m}_wm" > "$D/encode_${m}.log" 2>&1
  python -m evals.mw_metric_conflict.fit --data_dir "$D/data" --emb_dir "$D/emb_${m}_wm" --out "$D/fit_${m}_wm.json" --n_splits 3 > "$D/fit_${m}.log" 2>&1
  echo "$m exit $?" >> $D/fit_status.txt
done
touch "$D/FIT_DONE"
