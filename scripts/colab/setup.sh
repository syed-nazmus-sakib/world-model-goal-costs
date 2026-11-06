#!/bin/bash
# Colab VM: unpack the code bundle and install the stable-worldmodel stack (+ zstd, MuJoCo stack for Cube / Reacher).
# Prints GA_SETUP_DONE on success.
cd /content && mkdir -p jepa-wms-project out && tar xzf /content/ga_code.tgz -C /content/jepa-wms-project 2>/dev/null
apt-get install -y -qq zstd libegl1 libgl1 > /dev/null 2>&1 || echo "apt warn"
pip install -q "transformers<5" stable-pretraining pymunk shapely pygame gymnasium h5py hdf5plugin einops hydra-core omegaconf \
  tabulate tqdm typer rich pillow opencv-python-headless scikit-learn "huggingface_hub>=0.34,<1.0" "datasets>=3,<4.1" \
  "imageio[ffmpeg]" lancedb pylance pyarrow scipy statsmodels "h5py>=3.16" > /content/pip.log 2>&1 \
  || { echo "PIP FAILED"; grep -iE "error|conflict" /content/pip.log | tail -5; exit 1; }
pip install -q --no-deps -e /content/jepa-wms-project/code/third_party/stable-worldmodel >> /content/pip.log 2>&1 \
  || { echo "SWM INSTALL FAILED"; tail -5 /content/pip.log; exit 1; }
pip install -q mujoco dm-env dm-tree glfw lxml pyopengl absl-py > /content/pip_cube.log 2>&1 \
  && pip install -q --no-deps dm_control ogbench >> /content/pip_cube.log 2>&1 || echo "mujoco extras failed"
pip install -q gymnasium-robotics > /content/pip_gr.log 2>&1 || echo "gymnasium-robotics failed"
python3 -c "import torch, stable_worldmodel; print('SWM', torch.__version__, torch.cuda.is_available())" || exit 1
nproc; free -g | head -2
echo GA_SETUP_DONE
