# Graph-anchor planning experiments: setup and jobs

## Quick start: one command

```bash
git clone <repo> && cd world-model-goal-costs
bash scripts/hpc/run_all.sh
```

This one command:

1. Creates the Conda environment, or checks it if it already exists.
2. Runs a 15-minute smoke test. If the smoke test fails, it stops here.
3. Runs all experiment jobs (J1-J5), one per GPU at a time, spread over the node's GPUs.
4. Packs everything into **one file to send back: `out/hpc/ALL_results.tgz`** (a few MB).

How long it takes, roughly: 30 A100 hours of work in total. That is about 30 h on 1 GPU, 16 h on 2 GPUs, 11 h on 3 GPUs, or 10 h on 4 GPUs. `PLAN_ONLY=1 bash scripts/hpc/run_all.sh` prints which GPU will run what.

**If it stops** (cluster time limit, crash, reboot), run the same command again. Finished jobs are skipped, and unfinished ones continue where they stopped.

Everything below is optional detail.

## Requirements

- One NVIDIA GPU per job. An A100 40 GB is plenty; the jobs were developed on 24 GB L4s. Driver 525 or newer.
- About 64 GB of host RAM per job (the baseline trainers cache the dataset in memory).
- 12 or more CPU cores per job.
- About 45 GB of disk for the data cache, set by `STABLEWM_HOME` (default `~/.stable_worldmodel`). PointMaze-Large alone is 22.5 GB.
- About 5 GB for the Conda environment.
- Internet access on the node that runs `env_setup.sh` and on the first run of each job (for downloads).

## Running the parts by hand (optional)

### Environment (once, about 5-10 minutes)

```bash
git clone <repo> && cd world-model-goal-costs
bash scripts/hpc/env_setup.sh            # creates the conda env "gwm", then prints "[check] PASSED"
```

- If the login node has no GPU, run `SKIP_GPU_TEST=1 bash scripts/hpc/env_setup.sh` there. Then run `bash scripts/hpc/env_setup.sh check` on a GPU node.
- If `conda` is not on `PATH`, set `CONDA_ROOT=/path/to/miniconda3` (or run `module load anaconda3` first).
- If `$HOME` has a small quota, set `CONDA_ENVS_PATH` and `STABLEWM_HOME` to a scratch directory. Use the same `STABLEWM_HOME` for every job.

### Smoke test (about 15-20 minutes)

```bash
bash scripts/hpc/run_job.sh smoke
```

This downloads the Two-Room and PointMaze-Medium data and checkpoints, builds the planning graphs, and runs each planner on 2 episodes. The last lines of `out/hpc/smoke/job.log` say `JOB smoke PASS` or `FAIL`.

### Jobs

Each job uses one GPU. Jobs can run at the same time on different GPUs (`CUDA_VISIBLE_DEVICES`, or one SLURM job each). They are listed in priority order.

| Job | What it runs | Est. A100 hours |
|---|---|---|
| `J1` | PointMaze-Large: trains our LeWM world model (no public one exists), then evaluates 4 planners at 4 distances and on stitched goals, 2 seeds | 8-10 |
| `J2` | GCIVL baseline on PointMaze-Medium (training, then standard and stitched goals) | 4 |
| `J3` | PLDM baseline on PointMaze-Medium, then our graph anchors on top of PLDM | 3-4 |
| `J4` | Third seed for all main-table rows (Two-Room and PointMaze-Medium) | 4-5 |
| `J5` | Ablations (graph edges, anchor span, graph size, k-NN, compute-matched baseline) | 6-8 |

```bash
bash scripts/hpc/run_job.sh J1      # then J2 ... J5, or in parallel on separate GPUs
```

When a job finishes, `out/hpc/<job>/job.log` ends with a results table and `JOB <job> PASS` or `FAIL`. Its own archive is `out/hpc/<job>_results.tgz`. `run_all.sh` collects them all into `ALL_results.tgz`.

**Time limits and crashes:** run the same command again. Finished evaluations are skipped, and finished trainings are not repeated.

Example SLURM script for the whole run (adapt partition, account and time limit; resubmit the same script if it hits the limit):

```bash
#!/bin/bash
#SBATCH --job-name=gwm-all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
cd /path/to/world-model-goal-costs
bash scripts/hpc/run_all.sh
```

## Options

| Variable | Default | Meaning |
|---|---|---|
| `GPUS` | all GPUs | GPUs `run_all.sh` uses, e.g. `"0 1"` |
| `MAXPAR` | one per GPU | At most this many jobs at once (each needs about 64 GB of host RAM) |
| `JOBS` | `J1 J2 J3 J4 J5` | Subset of jobs for `run_all.sh` |
| `GWM_ENV` | `gwm` | Conda environment name |
| `STABLEWM_HOME` | `~/.stable_worldmodel` | Data and checkpoint cache (shared by all jobs) |
| `OUT_ROOT` | `out/hpc` | Where results go |
| `QUICK=1` | off | Plumbing test: a few training steps and 2 episodes per evaluation, written to `out/hpc_quick/` |
| `TORCH_CUDA` | `cu126` | PyTorch CUDA build used by `env_setup.sh` |
| `MUJOCO_GL` | `egl` | Headless rendering backend |
| `GWM_EGL_FIX` | `1` | Register NVIDIA's EGL library for MuJoCo when it is present (`0` to skip) |

## Troubleshooting

- **`[check]` fails at the MuJoCo render step.** The node has no working EGL. Try `GWM_EGL_FIX=0`, or ask for nodes with the NVIDIA EGL libraries (`libEGL_nvidia.so`). As a last resort, use `MUJOCO_GL=osmesa` after `conda install -n gwm -c conda-forge mesalib`, which is slower.
- **Download errors.** The data comes from Hugging Face (`quentinll/*`, `yjang43/lp2-pointmaze-medium`, `LeWAM/lewam-pointmaze-large`). Check the node's internet access, or set `HF_HUB_OFFLINE=1` after the first successful download.
- **A run shows `FAILED` in `<job>/*/queue.log`.** Its own log, `<job>/<env dir>/<run name>.log`, has the error. `ALL_results.tgz` includes it, so sending that one file is enough.
- **Jobs get killed (`rc=137`).** The node ran out of host RAM. Set `MAXPAR` lower so fewer jobs run at once.
