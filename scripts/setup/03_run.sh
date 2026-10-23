#!/usr/bin/env bash
# Step 3 of 3: run the experiments (training of the goal-cost heads + closed-loop planning evaluation).
# Results go to artifacts/raw_logs/<experiment>/; each run ends with a one-line success summary in queue.log.
# Background and findings: docs/10_METHOD_C_D_RESULTS.md, docs/11_DIRECTIONS.md.
#
#   smoke        1 Metaworld episode of data generation + 1 closed-loop episode (about 5 min): checks GPU, EGL, data
#   mw_state     criterion-aligned head on 9 Metaworld tasks: simulator-labelled encodings, head training, closed loop
#                (12 episodes) for the head and the released cost. Several hours; CPU-heavy (MuJoCo workers).
#   lewm_cube    LeWM Cube: latents, state / time heads, planning runs at goal offsets 50 and 75 for seeds 0 1 2
#   lewm_pusht   the same on LeWM PushT
#
# Usage:  bash scripts/setup/03_run.sh smoke
#         bash scripts/setup/03_run.sh mw_state            (or several: mw_state lewm_cube)
# On a batch system, submit this script as the job command with one GPU, >= 16 CPU cores and >= 48 GB RAM.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/env/conda.sh"
CONDA_BASE="$(project_conda_root)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
export JEPAWM_HOME="$ROOT"
OUT="$ROOT/artifacts/raw_logs"; mkdir -p "$OUT"
[[ $# -gt 0 ]] || { sed -n 2,15p "$0"; exit 1; }

check_gpu() {
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no GPU visible"; exit 1; }
  bash "$ROOT/scripts/pod/check_egl.sh"
}

for what in "$@"; do
  case $what in
    smoke)
      conda activate jepa-wms; check_gpu
      source "$ROOT/scripts/pod/jwm_env.sh"
      python -m evals.failure_attribution.mw_pilot_data --fname "$ROOT/configs/mw_jepa-wm_drawer-open.yaml" \
        --out_dir "$OUT/smoke/data" --ep_start 0 --episodes 1 --num_workers 8 && echo "[smoke] data OK"
      python -m evals.failure_attribution.run_closed --fname "$ROOT/configs/mw_jepa-wm_drawer-open.yaml" \
        --out_dir "$OUT/smoke/closed" --episodes 1 --planners sim_latent --num_workers 8 && echo "[smoke] closed loop OK"
      echo "[smoke] one closed-loop episode should take about 2 min; much longer means CPU rendering (see check_egl.sh)"
      ;;
    mw_state)
      conda activate jepa-wms; check_gpu
      bash "$ROOT/scripts/pod/run_mw_state_head.sh" "$OUT/mw_state_head"
      grep closed "$OUT/mw_state_head/queue.log"
      ;;
    lewm_cube|lewm_pusht)
      conda activate swm; check_gpu
      env=${what#lewm_}
      for seed in 0 1 2; do
        SWM_PY="$(command -v python)" bash "$ROOT/scripts/pod/run_lewm_methodC.sh" "$env" "$seed" "$OUT/lewm_methodC_$env"
      done
      cat "$OUT/lewm_methodC_$env/summary.txt"
      ;;
    *) echo "unknown experiment $what (smoke | mw_state | lewm_cube | lewm_pusht)"; exit 1 ;;
  esac
done
