# Setup after moving or cloning

The checkout contains source code, configs, documents, and curated results.
Datasets, checkpoints, and raw runs are stored separately in `artifacts/`.
See `artifacts/README.md` and `PROVENANCE.md` for their contents and origins.

## 1. Activate the environment

From the project root:

```bash
source env/env.sh
```

Conda is discovered from the command on PATH or a standard Miniconda, Anaconda,
or Miniforge installation. For another installation, set `CONDA_ROOT` before
activation. If the environment does not exist, run `bash env/setup_env_5090.sh`.
That script installs the recorded Python 3.10 / PyTorch CUDA 12.8 environment;
its filename records the GPU used for the original experiments.

## 2. Restore artifacts

Copy the preserved datasets and weights to these defaults:

```text
artifacts/datasets/pusht_noise/
artifacts/datasets/wall_single/
artifacts/datasets/Metaworld/
artifacts/checkpoints/{pt,wall,mw}_{jepa,dino}-wm.pth.tar
```

For a fresh clone, download them using the dataset and checkpoint instructions
in `code/jepa-wms/README.md`. Push-T and Wall archives must be extracted.
The project uses the original DINO-WM public datasets and released JEPA-WMs
weights; the repository contains neither data nor weights.

To keep artifacts elsewhere, export `JEPAWM_DSET`, `JEPAWM_CKPT`, or
`JEPAWM_LOGS` before sourcing `env/env.sh`. YAML configs use these variables
and the loader expands them, so moving the checkout requires no path rewrite.
`bash scripts/relocate_configs.sh` converts any older absolute-path configs.

## 3. Verify

```bash
python3 scripts/check_repo.py --check-assets
python3 -m unittest discover -s tests -v
cd code/jepa-wms
python -m tests.test_fast_rollout_mw --tasks mw-push --trials 2
```

The first two checks run without a GPU or simulator dependencies. The final
check requires the activated simulation environment and must print zero state
and pixel differences. An interrupted experiment resumes from completed
episodes; new episode outputs are written atomically. Simulator worker waits
are bounded by `JEPAWM_ROLLOUT_TIMEOUT` (300 seconds per batch by default).

## Earlier LeWM study

Install `code/third_party/stable-worldmodel` in editable mode in the `swm`
environment, then source `env/lewm.sh`. Set `PUSHT_EXPERT_H5` for an expert HDF5
dataset outside `~/.stable_worldmodel/datasets/pusht_expert_train.h5`.
Activation creates the study's `results` link to
`artifacts/lewm_explorations_raw` when it is absent. The LeWM scripts use the
activated interpreter and the bundled stable-worldmodel source.
