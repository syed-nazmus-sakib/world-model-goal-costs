#!/usr/bin/env bash
# Step 2 of 3: download the datasets and model checkpoints into artifacts/ (the paths the configs expect).
#
#   metaworld   Metaworld expert dataset (42 tasks, about 0.7 GB of parquet) -> artifacts/datasets/Metaworld/
#               (the model build reads it for the action normalisation, so it is needed even for evaluation)
#   ckpt        released JEPA-WM / DINO-WM checkpoints (Metaworld, Push-T, Wall; about 1.5 GB) -> artifacts/checkpoints/
#   lewm        LeWM checkpoints and datasets for PushT and Cube, fetched into ~/.stable_worldmodel/ (about 25 GB)
#
# The Hugging Face repo facebook/jepa-wms may ask for access approval: accept it on the website once, then run
# `huggingface-cli login` (or set HF_TOKEN) before this script.
#
# Usage (after 01_conda_env.sh, on a node with internet access):
#   bash scripts/setup/02_download.sh                  # metaworld + ckpt (enough for the Metaworld experiments)
#   bash scripts/setup/02_download.sh metaworld ckpt lewm
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/env/conda.sh"
CONDA_BASE="$(project_conda_root)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
export JEPAWM_HOME="$ROOT" JEPAWM_DSET="$ROOT/artifacts/datasets" JEPAWM_CKPT="$ROOT/artifacts/checkpoints"
mkdir -p "$JEPAWM_DSET" "$JEPAWM_CKPT"
if [[ $# -gt 0 ]]; then WHAT=("$@"); else WHAT=(metaworld ckpt); fi

for w in "${WHAT[@]}"; do
  case $w in
    metaworld)
      conda activate jepa-wms
      (cd "$ROOT/code/jepa-wms" && python src/scripts/download_data.py --dataset metaworld)
      ls "$JEPAWM_DSET/Metaworld/data" | head -2 && echo "[metaworld] $(ls "$JEPAWM_DSET/Metaworld/data" | wc -l) parquet files"
      ;;
    ckpt)
      for f in mw_jepa-wm mw_dino-wm pt_jepa-wm pt_dino-wm wall_jepa-wm wall_dino-wm; do
        [[ -s "$JEPAWM_CKPT/$f.pth.tar" ]] || curl -L --fail -o "$JEPAWM_CKPT/$f.pth.tar" "https://dl.fbaipublicfiles.com/jepa-wms/$f.pth.tar"
      done
      ls -la "$JEPAWM_CKPT"
      ;;
    lewm)
      conda activate swm
      python - <<'P'
import stable_worldmodel as swm
for m in ("quentinll/lewm-pusht", "quentinll/lewm-cube"):
    swm.wm.utils.load_pretrained(m)
    print("checkpoint ok:", m)
for d in ("galilai-group/lewm-pusht", "galilai-group/ogb_cube_single"):
    try:
        swm.data.load_dataset(d, num_steps=2, frameskip=5, keys_to_load=["action"])
        print("dataset ok:", d)
    except Exception as e:  # the Cube Lance dataset has one legacy column to drop, see scripts/pod/setup_swm_venv.sh
        print("dataset NOT ready:", d, repr(e)[:200])
P
      ;;
    *) echo "unknown item $w (metaworld | ckpt | lewm)"; exit 1 ;;
  esac
done
echo "Done. Next: bash scripts/setup/03_run.sh smoke"
