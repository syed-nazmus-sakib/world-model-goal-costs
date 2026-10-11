# When the goal image says more than the task: failures of latent goal costs in visual world-model planning

Research code and results, October 2026. Status: **analysis complete, method not yet started.**

Visual world-model planners (DINO-WM, JEPA-WMs, LeWM) receive the goal as an **image** and score candidate
action sequences by a distance between predicted and goal **latents**. This project asks where these planners
actually fail, using exact simulator oracles to separate failures of the world model, the goal cost and the
search.

**Start here:** [`docs/01_ASSESSMENT.md`](docs/01_ASSESSMENT.md), a two-page verdict on every failure case found.

## Main findings

1. **The goal cost, not the world model, is the main bottleneck on manipulation.** With exact simulator
   dynamics in place of the learned model, success barely changes (Push-T 64.6% → 64.6%), while scoring the
   same futures by the true state reaches 90.6%. On Metaworld push it is 8% vs 83%.
2. **Task-equivalence violation (main candidate for a method).** Two goal images that specify the same task
   outcome but show the robot arm in a different pose should be interchangeable. They are not:

   | | Metaworld push | Metaworld drawer-close |
   |---|---|---|
   | candidate rankings that flip (true criterion: 0%) | 43% | 36% |
   | same best plan selected | 6% | 8% |
   | closed-loop success, original goal → arm-moved goals | floored | 22/22 → 19/22, **16/22** (p = 0.031), 22/22 |

3. **Which parts of the goal matter depends on the task**: the pusher matters in Push-T, the arm is irrelevant
   for drawer-close, and on a spring-loaded button the arm *must* stay. No fixed rule (e.g. "mask the robot")
   is correct everywhere, which motivates a learned, task-aware goal representation.
4. **Goal-appearance shift** (earlier LeWM study): re-rendering the goal with a different background colour
   drops PushT success from 90% to 7%.
5. Closed or negative directions, reported plainly: area-invariant (top-k) patch costs, controllability-geometry
   terminal values, same-state branching data, post-hoc latent projections.

## Folder layout

```
.
├── README.md                 this file
├── docs/                     reports, read in numerical order (index: docs/README.md)
├── results/                  small, curated result files behind every number in docs/ (results/README.md)
├── code/
│   ├── jepa-wms/             fork of facebookresearch/jepa-wms + our evaluation suites under evals/
│   ├── lewm_explorations/    earlier LeWM / stable-worldmodel experiments (ctrlgeo/, scripts/, ogbench_shift/)
│   └── third_party/          stable-worldmodel snapshot used by lewm_explorations (with our additions)
├── configs/                  planner/eval configs using JEPAWM_LOGS and JEPAWM_CKPT
├── scripts/                  launch scripts for every experiment (scripts/README.md)
├── env/                      conda environment files and env.sh
├── artifacts/                large files, not for git: datasets, checkpoints, raw logs (artifacts/README.md)
└── tests/                    maintenance regressions requiring no GPU
```

## Reproducing

```bash
source env/env.sh                       # activates conda env `jepa-wms`, sets dataset/checkpoint/log paths
cd code/jepa-wms
python -m tests.test_fast_rollout_mw --tasks mw-push --trials 2      # exact simulator snapshot/restore test
bash ../../scripts/run_drawer_variants.sh                            # e.g. the drawer-close equivalence runs
```

Recorded experiment environment: Linux, one RTX 5090 (32 GB), CUDA 12.8, torch 2.7.0+cu128, MuJoCo 3, Metaworld. To rebuild the
environment: `bash env/setup_env_5090.sh` (exact package list in `env/requirements-5090.txt`). Each
experiment's exact command and decision rule is in `docs/04_NEXT_STEPS.md` and `scripts/README.md`.

## Moving the folder

`env/env.sh` and the launch scripts find the checkout on their own. YAML configs use `${JEPAWM_LOGS}` and
`${JEPAWM_CKPT}`, expanded by the config loader. Moving or cloning needs no path rewrite. Set `CONDA_ROOT`
if Conda is installed outside the usual locations; override `JEPAWM_DSET`, `JEPAWM_CKPT` or `JEPAWM_LOGS`
before activation to store artifacts elsewhere. See [setup instructions](docs/setup/MOVING.md).

Before uploading, run the checks below. They require NumPy and PyYAML, without GPU or simulation packages:

```bash
python3 scripts/check_repo.py
python3 -m unittest discover -s tests -v
```

GitHub preparation and push commands: [docs/setup/GITHUB.md](docs/setup/GITHUB.md).

## Provenance

- `code/jepa-wms`: upstream `facebookresearch/jepa-wms` at commit `13cf1d9c` (2026-04-11). Our work is purely
  additive: `evals/{failure_attribution, pusht_failure_attribution, pusht_goal_decomp, mw_cycles,
  mw_hidden_physics, mw_metric_conflict}/` and `tests/test_*fast_rollout*.py`. The upstream ignore rules are
  adjusted so our attribution entry points are included in a clone.
- `code/third_party/stable-worldmodel`: upstream `galilai-group/stable-worldmodel` at commit `8b2e8c68`, with
  our additions under `scripts/data/`, `scripts/plan/` and `stable_worldmodel/envs/two_room/trap_env.py`.
- Released checkpoints: JEPA-WM / DINO-WM (facebookresearch/jepa-wms), LeWM (`quentinll/lewm-pusht`).
- Upstream sources are bundled as ordinary folders, with their licenses and revisions recorded in
  [PROVENANCE.md](PROVENANCE.md). The original nested Git metadata is preserved locally under
  `artifacts/local_git/` and excluded from the GitHub repository.
