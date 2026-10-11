# Shared paths; source this without activating a Python environment.
export JEPAWM_HOME="${JEPAWM_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export JEPAWM_DSET="${JEPAWM_DSET:-$JEPAWM_HOME/artifacts/datasets}"
export JEPAWM_LOGS="${JEPAWM_LOGS:-$JEPAWM_HOME/artifacts/raw_logs}"
export JEPAWM_CKPT="${JEPAWM_CKPT:-$JEPAWM_HOME/artifacts/checkpoints}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
