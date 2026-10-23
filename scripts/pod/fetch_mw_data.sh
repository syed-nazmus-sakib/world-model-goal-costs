#!/bin/bash
# Fetch the Metaworld expert dataset (127 parquet files, about 0.7 GB) straight from Hugging Face onto the pod, into the
# layout the configs expect (artifacts/datasets/Metaworld/{data,hf_data}). The repo facebook/jepa-wms is GATED: listing
# works without login but downloads give 401 (2026-10-23). Accept the terms on the website once, then run with
# HF_TOKEN=<your token> in the environment (never written to disk). About a minute on a pod, vs 20-40 min of laptop upload.
# The model build reads these files for the action normalisation, so they are needed even for evaluation only.
# usage: HF_TOKEN=... bash scripts/pod/fetch_mw_data.sh [JEPAWM_HOME=/root/jepa-wms-project]
set -e
[ -n "$HF_TOKEN" ] || { echo "HF_TOKEN is not set: the dataset is gated"; exit 1; }
R=${1:-${JEPAWM_HOME:-/root/jepa-wms-project}}
D=$R/artifacts/datasets/Metaworld
if [ "$(ls $D/data/*.parquet 2>/dev/null | wc -l)" -ge 126 ]; then echo "Metaworld data present in $D"; exit 0; fi
pip install -q huggingface_hub 2>/dev/null || true
python3 - <<'P'
from huggingface_hub import snapshot_download
import os
print(snapshot_download("facebook/jepa-wms", repo_type="dataset", allow_patterns=["metaworld/*"], local_dir="/root/hf_mw", token=os.environ["HF_TOKEN"]))
P
mkdir -p $R/artifacts/datasets && rm -rf $D && mv /root/hf_mw/metaworld $D
echo "Metaworld: $(ls $D/data/*.parquet | wc -l) parquet files in $D"
