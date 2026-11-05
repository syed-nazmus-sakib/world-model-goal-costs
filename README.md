# Keep the Goal in Range: Graph Anchors for Long-Horizon Latent World-Model Planning

Research code and results for a CVPR 2027 submission (deadline 2026-11-16).

Latent world-model planners (LeWM, DINO-WM, PLDM) search for actions whose predicted embedding comes close to the
embedding of a goal image. We measure where this latent distance is a usable planning cost and find that it is
reliable only near the goal: on the graph of the model's own training data, it ranks one-step moves correctly next to
the goal but at chance beyond about 50 steps. **Graph anchors** keep the model, the cost and the search unchanged.
They link the encoded training frames by temporal successors and latent nearest neighbours, find the shortest path
to the goal, and give the planner the node six hops along it as its target: a goal it can actually reach.

## Results at a glance

Success (%) at the longest goal distance in each environment, LeWM checkpoints, same pairs and budgets for every
method (mean of two seeds unless noted).

| | Push-T d150 | Two-Room d75 | PointMaze-M d200 | PointMaze-L d400 | Reacher d150 |
|---|---|---|---|---|---|
| LeWM planner | 5 | 31 | 13 | 8 | 60 |
| LeWM + min-over-plan | 9 | 62 | 12 | 26 | 74 |
| Best offline GCRL baseline (GCBC) | – | **100** | 51 | 18 | – |
| **Graph anchors** | **51**¹ | 93 | **85** | **87** | 67 |
| **+ gated goal handoff** | – | 88 | 83 | – | **81** |

¹ with episode-switch cost λ = 3. Graph anchors also reach goals taken from other episodes (PointMaze-M: 95 vs 39),
work with graphs built from random-play data or 1% of the episodes, and need no training. They do not help on
OGBench-Cube, where failures come from contact, not from distance. Full tables, seeds and baselines are in the paper.

## Repository layout

```
.
├── code/
│   ├── lewm_explorations/paa/      graph anchors: graph construction (paa_nodes.py), anchor cost (paa_cost.py),
│   │                               baselines and helpers
│   ├── third_party/stable-worldmodel/   planning stack (CEM, eval_wm.py, environments) with our additions
│   └── jepa-wms/                   fork of facebookresearch/jepa-wms (earlier Meta-World diagnosis)
├── scripts/
│   ├── hpc/                        cluster jobs J1-J18 (run_job.sh, run_all.sh, run_1gpu.sh, env_setup.md)
│   ├── pod/                        single-GPU runners: run_ga_env.sh (all methods per env), run_baselines.sh,
│   │                               RunPod helpers
│   └── ...                         launch scripts of the earlier diagnosis phase (scripts/README.md)
├── configs/                        planner / eval configs (paths via JEPAWM_LOGS, JEPAWM_CKPT)
├── docs/                           reports and decision logs, read in numerical order (docs/README.md)
├── results/                        small curated result files of the diagnosis phase (results/README.md)
├── env/                            conda environment files and env.sh
├── tests/                          regression tests that need no GPU
└── artifacts/                      large files, not in git: datasets, checkpoints, run outputs, HPC results
```

Local only (git-ignored): `paper/` (LaTeX manuscript, `paper/v3` is current), `notes/` (meeting notes, write-ups,
research notes, reviews), `others/` (third-party papers and reference repos, scratch files) and hand-off bundles
in the root such as `dinowm_files.tgz`.

## Running

**One environment on one GPU** (data, checkpoint, graph and every method at the given distances):

```bash
bash scripts/pod/run_ga_env.sh tworoom "0 42" out/ga_tworoom          # LeWM, min-over-plan, graph anchors
OFFS="100 150" TAG=_hg12 EVX="+paa_handoff=true +paa_handoff_hops=12" \
  bash scripts/pod/run_ga_env.sh reacher 0 out/ga_reacher anchor      # with the gated goal handoff
```

**Cluster jobs** (environment setup in [`scripts/hpc/env_setup.md`](scripts/hpc/env_setup.md)):

```bash
QUICK=1 bash scripts/hpc/run_job.sh J18                                 # plumbing test, a few minutes
GPU=0 OUT_ROOT=out/hpc JOBS="J18" bash scripts/hpc/run_1gpu.sh            # full job; results in out/hpc/ALL_results.tgz
```

Each job is described at the top of its block in `scripts/hpc/run_job.sh`. Finished runs are skipped, so an
interrupted job continues where it stopped.

## Checks before pushing

```bash
python3 scripts/check_repo.py
python3 -m unittest discover -s tests -v
```

Never commit `artifacts/`, `paper/`, `notes/`, `others/` or `.env`. GitHub workflow: [`docs/setup/GITHUB.md`](docs/setup/GITHUB.md).

## Earlier phase (October 2026)

The project began as a diagnosis of latent goal costs on manipulation (Meta-World, Push-T): with exact simulator
dynamics, success barely changes, while scoring the same futures by the true state succeeds far more often, so the
goal cost, not the world model, is the bottleneck there. These results and the closed directions are documented in
[`docs/01_ASSESSMENT.md`](docs/01_ASSESSMENT.md) to [`docs/10_METHOD_C_D_RESULTS.md`](docs/10_METHOD_C_D_RESULTS.md).

## Provenance

- `code/third_party/stable-worldmodel`: upstream `galilai-group/stable-worldmodel` at commit `8b2e8c68`, with our
  additions under `scripts/plan/`, `scripts/data/` and `stable_worldmodel/envs/two_room/trap_env.py`.
- `code/jepa-wms`: upstream `facebookresearch/jepa-wms` at commit `13cf1d9c`; our additions are under `evals/`.
- Checkpoints: LeWM (`quentinll/lewm-*`), PointMaze-Medium LeWM (`yjang43/lp2-pointmaze-medium`), released DINO-WM
  Push-T. Licenses and revisions: [`PROVENANCE.md`](PROVENANCE.md).
