#!/usr/bin/env bash
# Conda environment "gwm" for the graph-anchor planning experiments (scripts/hpc/run_job.sh).
#
#   bash scripts/hpc/env_setup.sh          # create (or update) the environment, then run the GPU / rendering check
#   SKIP_GPU_TEST=1 bash scripts/hpc/env_setup.sh   # on a login node without a GPU; run `bash scripts/hpc/env_setup.sh check`
#                                                   # later on a GPU node
#   bash scripts/hpc/env_setup.sh check    # only the check
#   bash scripts/hpc/env_setup.sh python   # print the path of the environment's python
#
# Options (environment variables):
#   GWM_ENV=gwm            environment name
#   GWM_FRESH=1            install the newest allowed package versions, not those of the last install on this machine
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

# The environment's python, found by its path.  `conda run` is not used: its wrapper stops without a message when any
# activation script of the calling shell returns an error, and the environment then looks missing although it is there.
env_prefix() {
    local d
    for d in ${CONDA_ENVS_PATH:+${CONDA_ENVS_PATH//:/ }} "$CONDA_BASE/envs" "$HOME/.conda/envs"; do
        [[ -d "$d/$ENV_NAME" ]] && { printf '%s\n' "$d/$ENV_NAME"; return 0; }
    done
    d="$(conda env list 2>/dev/null | awk -v n="$ENV_NAME" '$1 == n {print $NF}' | head -1)"
    [[ -n "$d" && -d "$d" ]] && { printf '%s\n' "$d"; return 0; }
    return 1
}
env_python() {
    local p; p="$(env_prefix)" || { echo "[$ENV_NAME] no such Conda environment" >&2; return 1; }
    "$p/bin/python" -c 'import sys' 2>/dev/null || { echo "[$ENV_NAME] $p/bin/python does not run" >&2; return 1; }
    printf '%s\n' "$p/bin/python"
}

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
if [[ "${1:-}" == python ]]; then env_python; exit $?; fi   # print the path of the environment's python

if P="$(env_prefix)"; then
    # An existing environment is never created again: `conda create -y` deletes the one that is there without asking.
    "$P/bin/python" -c 'import sys' || { echo "[$ENV_NAME] $P exists but its python does not run. Nothing was changed."
        echo "    To build it again from nothing: conda env remove -n $ENV_NAME, then run this script again."; exit 1; }
    echo "[$ENV_NAME] exists at $P, installing its packages"
else
    # conda-forge only (no Anaconda 'defaults' terms-of-service prompt); zstd unpacks the PointMaze-Medium data
    conda create -n "$ENV_NAME" python=3.12 zstd -c conda-forge --override-channels -y
fi
PY="$(env_python)"
FROZEN="$H/frozen_${ENV_NAME}_${TORCH_CUDA}.txt"   # written by the last complete install on this machine
WHL="https://download.pytorch.org/whl/$TORCH_CUDA"
"$PY" -m pip install -U pip
if [[ -s "$FROZEN" && "${GWM_FRESH:-0}" != 1 ]] && grep -q '^torch==' "$FROZEN"; then
    echo "[$ENV_NAME] installing the same package versions as the last install here ($FROZEN)"
    grep -vE ' @ |^-e ' "$FROZEN" > "$FROZEN.use"   # without the packages Conda itself installed (listed by a build path)
    "$PY" -m pip install --no-deps -r "$FROZEN.use" --extra-index-url "$WHL" \
        || { echo "[$ENV_NAME] that failed; installing from the requirements instead"; FROZEN_BAD=1; }
fi
if [[ ! -s "$FROZEN" || "${GWM_FRESH:-0}" == 1 || "${FROZEN_BAD:-0}" == 1 ]] || ! grep -q '^torch==' "$FROZEN"; then
    "$PY" -m pip install torch torchvision --index-url "$WHL"
    if [[ -f "$H/requirements-lock.txt" ]]; then REQ="$H/requirements-lock.txt"; else REQ="$H/requirements.txt"; fi
    echo "[$ENV_NAME] installing $REQ"
    "$PY" -m pip install -r "$REQ"
    # dm_control / OGBench pin older MuJoCo versions than the rest of the stack; install them without their pins (as in our runs)
    "$PY" -m pip install --no-deps dm_control ogbench
fi
"$PY" -m pip install --no-deps -e "$ROOT/code/third_party/stable-worldmodel"
"$PY" -c "import torch, stable_worldmodel, ogbench, gymnasium_robotics"   # stop here, before the list is rewritten, if incomplete
"$PY" -m pip freeze --exclude-editable > "$FROZEN.new" && mv "$FROZEN.new" "$FROZEN"
echo "[$ENV_NAME] installed; package list in scripts/hpc/frozen_${ENV_NAME}_${TORCH_CUDA}.txt"
[[ "${SKIP_GPU_TEST:-0}" == 1 ]] || check
echo "Done. Next: bash scripts/hpc/run_job.sh smoke"
