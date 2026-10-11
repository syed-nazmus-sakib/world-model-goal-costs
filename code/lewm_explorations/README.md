# LeWM explorations (earlier phase, 2026-10-03 to 10-05)

Frozen and retrained LeWM world models (stable-worldmodel) on PushT and OGBench cube-single. All directions here
were closed by pre-registered kill tests; outcomes are summarised in `../../docs/01_ASSESSMENT.md` (F6, F7, F9)
and `../../results/README.md`.

| Path | Content |
|---|---|
| `ctrlgeo/` | library: frozen LeWM wrapper (`wm.py`), batched CEM with pluggable terminal values (`planner.py`, `scores.py`), exact cube simulator (`cube_sim.py`), LeWM trainer (`train.py`) |
| `scripts/exp1_ranking.py`, `exp2_planning.py` | controllability-geometry terminal value: ranking, closed-loop planning, confirmation |
| `scripts/exp3_subspace.py` | linear projection against goal-appearance shift |
| `scripts/exp4_*.py`, `exp5_*`, `exp6_*` | counterfactual data studies (PushT, cube): datasets, training pipeline, evaluation |
| `scripts/exp7_cube_cost.py` | cube: patch-cost aggregations with exact dynamics |
| `ogbench_shift/` (+ `.tar.gz`) | goal-appearance shift benchmark kit for OGBench |
| `run_logs/` | stdout logs of the runs |

Recorded environment: conda env `swm` (torch with CUDA 13, RTX 5090), `../third_party/stable-worldmodel` installed with
`pip install -e`. Activate from the project root with `source env/lewm.sh`. Checkpoints/datasets are expected in `~/.stable_worldmodel` (copies in
`../../artifacts/lewm_checkpoints/`; the PushT expert h5 dataset is not included and is downloaded by
stable-worldmodel). Set `PUSHT_EXPERT_H5` to use a different expert dataset location. Scripts are run from
this folder and read/write `results/`; `env/lewm.sh` creates the link to
`../../artifacts/lewm_explorations_raw` when it is absent. Both environments use `CONDA_ROOT` when supplied.
