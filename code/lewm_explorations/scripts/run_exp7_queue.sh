#!/usr/bin/env bash
set -euo pipefail
# Cube cost test (exp7): starts once the Push-T cost sweeps have released the CPU cores.
cd "$(dirname "$0")/.."
while pgrep -f "pusht_failure_attribution.cost_sweep" > /dev/null; do sleep 60; done
mkdir -p logs
export MUJOCO_GL=egl PYTHONPATH=.
source ../../env/lewm.sh || exit 1
PY="$(command -v python)"
"$PY" scripts/exp7_cube_cost.py --costs true mean topk:8 topk:4 --episodes 50 >> logs/exp7_cube_cost.log 2>&1
"$PY" scripts/exp7_cube_cost.py --costs mean topk:8 --alpha 10 --episodes 50 >> logs/exp7_cube_cost.log 2>&1
