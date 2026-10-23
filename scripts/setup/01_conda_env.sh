#!/usr/bin/env bash
# Step 1 of 3: create the two Conda environments used by the experiments.
#
#   jepa-wms  DINO-WM / JEPA-WM planning on Metaworld, Push-T, Wall (code/jepa-wms, all evals/ suites).
#             Exact package set of the original runs (env/requirements-5090.txt) + PyTorch 2.7.0.
#   swm       LeWM study and GoalShift / TwoRoom / cube experiments (code/third_party/stable-worldmodel,
#             code/lewm_explorations). Built from stable-worldmodel's own dependency list (not pinned in the
#             original runs); the installed versions are written to env/frozen/ for the record.
#
# Usage (from anywhere; run on a node with internet access):
#   bash scripts/setup/01_conda_env.sh            # both environments
#   bash scripts/setup/01_conda_env.sh jepa-wms   # one of them
#
# Options (environment variables):
#   TORCH_CUDA=cu126|cu128  PyTorch CUDA build. Default: cu128 on Blackwell GPUs (RTX 50xx, B200), cu126 otherwise
#                           (A100 / H100 / L40 / RTX 30-40xx; needs NVIDIA driver >= 525).
#   CONDA_ROOT=/path        Conda installation, if `conda` is not on PATH (e.g. after `module load anaconda3`
#                           it usually is).
#   CONDA_ENVS_PATH=/path   Where Conda puts new environments (useful when $HOME has a small quota).
#   SKIP_GPU_TEST=1         Skip the CUDA/rendering smoke test (e.g. on a login node without a GPU).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/env/paths.sh"
source "$ROOT/env/conda.sh"
CONDA_BASE="$(project_conda_root)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ $# -gt 0 ]]; then ENVS=("$@"); else ENVS=(jepa-wms swm); fi

if [[ -z "${TORCH_CUDA:-}" ]]; then
    TORCH_CUDA=cu126
    if command -v nvidia-smi >/dev/null 2>&1; then
        cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)"
        [[ "${cc:-0}" -ge 10 ]] && TORCH_CUDA=cu128
    fi
fi
TORCH_INDEX="https://download.pytorch.org/whl/$TORCH_CUDA"
echo "Conda: $CONDA_BASE | PyTorch build: $TORCH_CUDA | environments: ${ENVS[*]}"
mkdir -p "$ROOT/env/frozen"

env_python() {  # path of the environment's python (conda activate is unreliable inside scripts)
    conda run -n "$1" python -c 'import sys; print(sys.executable)'
}

create_env() {  # name
    if conda run -n "$1" python --version >/dev/null 2>&1; then
        echo "[$1] exists, reusing it"
    else
        # conda-forge only: avoids the Anaconda 'defaults' channel terms-of-service prompt on new Conda versions
        conda create -n "$1" python=3.10 "ffmpeg=7" -c conda-forge --override-channels -y
    fi
    local py
    py="$(env_python "$1")"
    "$py" -m pip install -U pip
    "$py" -m pip install "torch==2.7.0" "torchvision==0.22.0" --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple
}

setup_jepa_wms() {
    create_env jepa-wms
    local py
    py="$(env_python jepa-wms)"
    # --no-deps: the recorded set has a known dm_control/mujoco pin conflict; install it exactly as it was run.
    "$py" -m pip install --no-deps -r "$ROOT/env/requirements-5090.txt"
    "$py" -m pip install --no-deps -e "$ROOT/code/jepa-wms"
    "$py" -m pip freeze > "$ROOT/env/frozen/jepa-wms_${TORCH_CUDA}.txt"
    "$py" -c "import torch, mujoco, metaworld, tensordict, decord; print('[jepa-wms] imports ok, torch', torch.__version__)"
}

setup_swm() {
    create_env swm
    local py
    py="$(env_python swm)"
    "$py" -m pip install -e "$ROOT/code/third_party/stable-worldmodel[train,env,format,data]" \
        --extra-index-url "$TORCH_INDEX"
    "$py" -m pip freeze > "$ROOT/env/frozen/swm_${TORCH_CUDA}.txt"
    "$py" -c "import torch, stable_worldmodel, pymunk, ogbench; print('[swm] imports ok, torch', torch.__version__)"
}

gpu_test() {  # name: CUDA matmul + headless MuJoCo render (MUJOCO_GL=egl, set in env/paths.sh)
    local py
    py="$(env_python "$1")"
    "$py" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available: check the driver and TORCH_CUDA"
x = torch.randn(256, 256, device="cuda")
print("cuda ok:", torch.cuda.get_device_name(0), "| arch list", torch.cuda.get_arch_list(), "| matmul", float((x @ x).sum()))
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64)
r.update_scene(mujoco.MjData(m))
print("headless MuJoCo render ok:", r.render().shape)
EOF
}

for e in "${ENVS[@]}"; do
    case "$e" in
        jepa-wms) setup_jepa_wms ;;
        swm) setup_swm ;;
        *) echo "unknown environment: $e (expected jepa-wms or swm)" >&2; exit 1 ;;
    esac
    [[ "${SKIP_GPU_TEST:-0}" == 1 ]] || gpu_test "$e"
done

echo "Done. Activate with:  source env/env.sh   (jepa-wms)   or   source env/lewm.sh   (swm)"
