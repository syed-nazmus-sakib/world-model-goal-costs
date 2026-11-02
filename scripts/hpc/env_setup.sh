#!/usr/bin/env bash
# Conda environment "gwm" for the graph-anchor planning experiments (scripts/hpc/run_job.sh).
#
#   bash scripts/hpc/env_setup.sh          # create (or update) the environment, then run the GPU / rendering check
#   SKIP_GPU_TEST=1 bash scripts/hpc/env_setup.sh   # on a login node without a GPU; run `bash scripts/hpc/env_setup.sh check`
#                                                   # later on a GPU node
#   bash scripts/hpc/env_setup.sh check    # only the check
#
# Options (environment variables):
#   GWM_ENV=gwm            environment name
#   TORCH_CUDA=cu126       PyTorch CUDA build (cu126 needs NVIDIA driver >= 525; use cu128 on RTX 50xx / B200)
#   CONDA_ROOT=/path       Conda installation, if `conda` is not on PATH
#   CONDA_ENVS_PATH=/path  where Conda puts environments (if $HOME has a small quota)
#   STABLEWM_HOME=/path    data and checkpoint cache used by the jobs (default ~/.stable_worldmodel, needs ~40 GB)
#
# Installs: Python 3.12, PyTorch + torchvision (CUDA wheels), the pinned package set in scripts/hpc/requirements-lock.txt
# (falls back to scripts/hpc/requirements.txt), MuJoCo / dm_control / OGBench / gymnasium-robotics, and the repository's
# stable-worldmodel in editable mode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/env/conda.sh"
CONDA_BASE="$(project_conda_root)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="${GWM_ENV:-gwm}"
TORCH_CUDA="${TORCH_CUDA:-cu126}"
H="$ROOT/scripts/hpc"

env_python() { conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)'; }

check() {
    local py; py="$(env_python)"
    export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" SDL_VIDEODRIVER=dummy
    source "$H/egl.sh"
    PYTHONPATH="$ROOT/code/third_party/stable-worldmodel" "$py" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available: check the NVIDIA driver and TORCH_CUDA"
x = torch.randn(512, 512, device="cuda")
print("[check] cuda ok:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| matmul", round(float((x @ x).abs().sum()), 1))
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m))
print("[check] headless MuJoCo render ok:", r.render().shape, "| MUJOCO_GL =", __import__("os").environ.get("MUJOCO_GL"))
import gymnasium as gym, stable_worldmodel as swm  # noqa: F401  (registers the swm/* environments)
for name, kw in (("swm/TwoRoom-v1", {}), ("swm/PointMaze_Medium-v3", {}), ("swm/PointMaze-v0", {"task": "large", "resolution": 224})):
    env = gym.make(name, **kw); env.reset(seed=0); frame = env.render(); env.close()
    print(f"[check] {name} reset + render ok:", getattr(frame, "shape", None))
print("[check] PASSED")
EOF
}

if [[ "${1:-}" == check ]]; then check; exit 0; fi

if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    echo "[$ENV_NAME] exists, updating it"
else
    # conda-forge only (no Anaconda 'defaults' terms-of-service prompt); zstd unpacks the PointMaze-Medium data
    conda create -n "$ENV_NAME" python=3.12 zstd -c conda-forge --override-channels -y
fi
PY="$(env_python)"
"$PY" -m pip install -U pip
"$PY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
if [[ -f "$H/requirements-lock.txt" ]]; then REQ="$H/requirements-lock.txt"; else REQ="$H/requirements.txt"; fi
echo "[$ENV_NAME] installing $REQ"
"$PY" -m pip install -r "$REQ"
# dm_control / OGBench pin older MuJoCo versions than the rest of the stack; install them without their pins (as in our runs)
"$PY" -m pip install --no-deps dm_control ogbench
"$PY" -m pip install --no-deps -e "$ROOT/code/third_party/stable-worldmodel"
"$PY" -m pip freeze --exclude-editable > "$H/frozen_${ENV_NAME}_${TORCH_CUDA}.txt"
echo "[$ENV_NAME] installed; package list in scripts/hpc/frozen_${ENV_NAME}_${TORCH_CUDA}.txt"
[[ "${SKIP_GPU_TEST:-0}" == 1 ]] || check
echo "Done. Next: bash scripts/hpc/run_job.sh smoke"
