#!/bin/bash
# JEPA-WM environment on a RunPod pod (uv venv at /root/venv_jwm, torch 2.7.0 cu126; used on RTX 3090 / A4500 pods, 2026-10-22).
# Expects the project at JEPAWM_HOME with code/, configs/, env/, artifacts/checkpoints/mw_jepa-wm.pth.tar and the Metaworld
# parquet files in artifacts/datasets/Metaworld/data (the model build reads them for the action normalisation).
# Run scripts/pod/check_egl.sh first: hosts without the NVIDIA EGL driver render on the CPU, about 15x slower.
# usage: bash scripts/pod/setup_jwm_venv.sh            log: stdout; ends with SETUP_DONE
set -x
R=${JEPAWM_HOME:-/root/jepa-wms-project}
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libegl1 libgl1 libglib2.0-0 libpython3.10 > /dev/null) || echo APT_FAILED
pip install -q uv
export UV_LINK_MODE=copy
[ -x /root/venv_jwm/bin/python ] || uv venv -q /root/venv_jwm --python 3.10
P=/root/venv_jwm/bin/python
uv pip install -q --python $P torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126 || echo TORCH_FAILED
grep -v -E '^(torch|torchvision)==' $R/env/requirements-5090.txt > /root/req_jwm.txt
uv pip install -q --python $P --no-deps -r /root/req_jwm.txt || echo REQ_FAILED
uv pip install -q --python $P av || echo AV_FAILED  # PyAV: decoding the dataset videos (mw_head_data.py)
uv pip install -q --python $P --no-deps -e $R/code/jepa-wms || echo PKG_FAILED
$P -c "import torch; x = torch.randn(64, 64, device='cuda'); print(torch.__version__, float((x @ x).sum()))"
df -h / | tail -1
echo SETUP_DONE
