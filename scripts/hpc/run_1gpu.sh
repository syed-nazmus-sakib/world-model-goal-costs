#!/usr/bin/env bash
# All experiments on ONE large GPU (e.g. H100 80 GB, H200 141 GB): several jobs share the GPU at the same time.
#
#   bash scripts/hpc/run_1gpu.sh
#
# Same behaviour as run_all.sh: environment check, 15-min smoke test, jobs J1-J5, one file to send back
# (out/hpc/ALL_results.tgz), and re-running the same command continues after a time limit or crash.
#
# How many jobs run at once: each job needs at most ~24 GB of GPU memory and ~60 GB of host RAM, so
#   parallel jobs = min(GPU memory / 24 GB, available host RAM / 60 GB, 5)
# On a 141 GB GPU with >= 300 GB of free RAM all five jobs run together (~10-12 h); with less RAM, fewer.
#
# Options: GPU=0 (which GPU), PAR=3 (force the number of parallel jobs), VRAM_GB / RAM_GB (override the detected
# sizes), plus everything run_all.sh accepts (PLAN_ONLY=1, JOBS, QUICK=1, STABLEWM_HOME, ...).
set -uo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
VRAM_GB="${VRAM_GB:-$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%d", $1 / 1024}')}"
RAM_GB="${RAM_GB:-$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}')}"
VRAM_GB="${VRAM_GB:-24}"; RAM_GB="${RAM_GB:-64}"
if [[ -n "${PAR:-}" ]]; then n="$PAR"
else
    n=$(( VRAM_GB / 24 )); r=$(( RAM_GB / 60 )); (( r < n )) && n=$r; (( n > 5 )) && n=5
fi
(( n < 1 )) && n=1
echo "run_1gpu: GPU $GPU with ${VRAM_GB} GB, ${RAM_GB} GB free host RAM -> $n job(s) at the same time on this GPU"
slots=""; for ((i = 0; i < n; i++)); do slots="$slots $GPU"; done
GPUS="${slots# }" exec bash "$H/run_all.sh"
