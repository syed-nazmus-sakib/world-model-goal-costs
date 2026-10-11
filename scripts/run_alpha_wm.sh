#!/usr/bin/env bash
set -euo pipefail
# Run 1 of the plan: does the cost reweighting found with simulator dynamics also help the REAL
# world-model planner? Released Push-T eval (96 episodes) with --alpha, JEPA-WM and DINO-WM.
# Usage: bash run_alpha_wm.sh 10        (alpha; default: best of the finished sweep)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/env.sh"
cd "$ROOT/code/jepa-wms"
ALPHA=${1:-$(python - <<'PY'
import glob, os, numpy as np
best = None
for f in glob.glob(os.environ['JEPAWM_LOGS'] + '/pusht_failure_attribution/jepa_wm/alpha_sweep/alpha_*.npz'):
    a = os.path.basename(f)[6:-4]
    d = np.load(f, allow_pickle=True)
    k = [x for x in d.files if 'succ' in x.lower()][0]
    s = float(np.mean(d[k]))
    if best is None or s > best[1]: best = (a, s)
print(best[0])
PY
)}
echo "alpha = $ALPHA"
D=$JEPAWM_LOGS/failure_attribution
mkdir -p "$D"
for m in jepa dino; do
  python -m evals.failure_attribution.run --fname "$ROOT/configs/pt_${m}-wm_repro.yaml" \
    --out_dir "$D/pusht_${m}_wm_alpha${ALPHA}" --episodes 96 --alpha $ALPHA >> "$D/pusht_${m}_wm_alpha${ALPHA}.log" 2>&1
  python -m evals.failure_attribution.summarize --run_dir "$D/pusht_${m}_wm_alpha${ALPHA}" > "$D/pusht_${m}_wm_alpha${ALPHA}_summary.txt" 2>&1
done
