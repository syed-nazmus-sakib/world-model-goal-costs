#!/usr/bin/env bash
# Recreate the jepa-wms env on an RTX 5090 (Blackwell, sm_120): same pins as
# requirements-frozen.txt except torch/torchvision/triton/nvidia-* come from the CUDA 12.8 wheels.
# Uses the env's python explicitly (conda activate is unreliable inside scripts).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
source "$(dirname "${BASH_SOURCE[0]}")/conda.sh"
PROJECT_CONDA="$(project_conda_root)"
source "$PROJECT_CONDA/etc/profile.d/conda.sh"
if ! conda run -n jepa-wms python --version >/dev/null 2>&1; then
    conda create -n jepa-wms python=3.10 ffmpeg=7 -c conda-forge --override-channels -y
fi
conda activate jepa-wms
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade
python -m pip install -U pip
python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
# The original environment has a dm_control/mujoco pin conflict; preserve its exact package set.
python -m pip install --no-deps -r "$JEPAWM_HOME/env/requirements-5090.txt"
python -m pip install -e "$JEPAWM_HOME/code/jepa-wms" --no-deps
python -c "import torch; print('torch', torch.__version__, 'cuda ok', torch.cuda.is_available(), torch.cuda.get_arch_list()); x=torch.randn(64,64,device='cuda'); print('matmul ok', float((x@x).sum()))"
echo ENV_DONE
