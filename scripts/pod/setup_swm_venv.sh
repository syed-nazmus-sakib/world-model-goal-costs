#!/bin/bash
# LeWM / stable-worldmodel environment on a pod (uv venv at /root/venv_swm, torch 2.6.0 cu124; pod D, 2026-10-22).
# Installs the bundled code/third_party/stable-worldmodel (with our planning scripts) in editable mode.
# Datasets go under ~/.stable_worldmodel/datasets (names as in the configs: lewm-pusht, tworoom.h5, ogb_cube_single).
# Cube (galilai-group/ogb_cube_single, Lance): drop the legacy string column once, or loading fails:
#   python -c "import lance; lance.dataset('<path>.lance').drop_columns(['privileged_target_task'])"
# Cube rendering also needs: uv pip install --python /root/venv_swm/bin/python mujoco dm_control ogbench
# --index-strategy unsafe-best-match: with two indexes uv otherwise takes `requests` from the PyTorch index (too old) and fails (pod K, 2026-10-24).
# usage: bash scripts/pod/setup_swm_venv.sh
set -eo pipefail
R=${JEPAWM_HOME:-/root/jepa-wms-project}
apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglib2.0-0 > /dev/null || true
pip install -q uv
export UV_LINK_MODE=copy
[ -x /root/venv_swm/bin/python ] || uv venv -q -p 3.11 /root/venv_swm
P=/root/venv_swm/bin/python
uv pip install -q -p $P --index-strategy unsafe-best-match torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install -q -p $P --index-strategy unsafe-best-match "torch==2.6.0" "transformers<5" stable-pretraining pymunk shapely pygame \
  gymnasium h5py hdf5plugin einops hydra-core omegaconf tabulate tqdm typer rich pillow opencv-python-headless \
  scikit-learn "huggingface_hub>=0.34,<1.0" "datasets>=3,<4.1" "imageio[ffmpeg]" lancedb pylance pyarrow scipy \
  --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install -q -p $P --no-deps -e $R/code/third_party/stable-worldmodel
$P -c "import torch, stable_worldmodel; print('SWM', torch.__version__, torch.cuda.is_available())"
echo SETUP_DONE
