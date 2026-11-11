#!/usr/bin/env bash
# exp-30, step 1 of 2: the AntMaze data.  Downloads OGBench antmaze-large-navigate-v0 (243 MB) and renders its 1,000 episodes at
# 224 px into $STABLEWM_HOME/datasets/antmaze_large.h5 (roughly 40-80 GB; the log states the expected size after the first
# episodes).  Needs no trained model.  Rendering uses GPU 0 only.
#
#   bash scripts/exp/exp30_antmaze_data.sh            # all 1,000 episodes, about 1-2 h
#   EPISODES=20 bash scripts/exp/exp30_antmaze_data.sh   # a small file for a test
#
# Progress is printed here and in out/exp30_data/data.log:
#   [download] 37% | 90/243 MB | 6.2 MB/s | left 0:00:25
#   [render] episode 120/1000 (12%) | 415 frames/s | elapsed 0:05:10 | left 0:38:00 | file 5.1 GB (about 43 GB when done)
# It can be stopped and started again: the download resumes and rendering continues after the last finished episode.
# Then run step 2:  bash scripts/hpc/run_antmaze.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${GWM_NOCONDA:-0}" != 1 ]]; then
    source "$ROOT/env/conda.sh"; CONDA_BASE="$(project_conda_root)" || exit 1
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    EPY="$(bash "$ROOT/scripts/hpc/env_setup.sh" python)" || { echo "conda env ${GWM_ENV:-gwm} missing: run scripts/hpc/env_setup.sh"; exit 1; }
    conda activate "${GWM_ENV:-gwm}" || echo "conda activate ${GWM_ENV:-gwm} returned an error; using $EPY by its path"
    # if the activation did not take effect, put the environment first on PATH by hand
    [[ "$(command -v python)" == "$EPY" ]] || export PATH="$(dirname "$EPY"):$PATH" CONDA_PREFIX="$(dirname "$(dirname "$EPY")")"
fi
command -v python > /dev/null || python() { python3 "$@"; }
export PYTHONPATH="$ROOT/code/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH}"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable_worldmodel}" MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0     # GPU 0 only
source "$ROOT/scripts/hpc/egl.sh"
O="${OUT:-$ROOT/out/exp30_data}"; mkdir -p "$O" "$STABLEWM_HOME"
echo "$(date -u +%F' '%T) exp-30 data start on $(hostname), git $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null), free disk at $STABLEWM_HOME: $(df -h "$STABLEWM_HOME" | awk 'NR==2 {print $4}')" | tee -a "$O/data.log"
python -u "$ROOT/code/lewm_explorations/paa/antmaze_data.py" --episodes "${EPISODES:-0}" --workers "${WORKERS:-8}" \
    --preview "$O/preview.png" 2>&1 | tee -a "$O/data.log"
if grep -q '^DATA_DONE' "$O/data.log"; then echo "exp-30 data: PASS. Next: bash scripts/hpc/run_antmaze.sh"
else echo "exp-30 data: NOT finished (see $O/data.log); run the same command again to continue"; exit 1; fi
