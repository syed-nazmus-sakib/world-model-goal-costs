set -eo pipefail
C=/root/miniconda3/bin/conda
P=/root/miniconda3/envs/jepa-wms/bin/python
[ -x $P ] || $C create -n jepa-wms python=3.10 ffmpeg=7 -c conda-forge --override-channels -y
$P -m pip install -U pip
$P -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
echo TORCH_DONE
until [ -f /root/jepa-wms-project/env/requirements-5090.txt ] && grep -q UPLOAD_OK /root/upload.flag 2>/dev/null; do sleep 5; done
$P -m pip install --no-deps -r /root/jepa-wms-project/env/requirements-5090.txt
cd /root/jepa-wms-project/code/jepa-wms && $P -m pip install -e . --no-deps
$P -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list()); x=torch.randn(64,64,device=\"cuda\"); print(float((x@x).sum()))"
echo ENV_DONE
