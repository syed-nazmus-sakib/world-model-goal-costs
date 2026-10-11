# Code

## `jepa-wms/`: main study

Upstream `facebookresearch/jepa-wms` (commit `13cf1d9c`), plus our evaluation suites and ignore-rule exceptions:

| Path | Purpose |
|---|---|
| `evals/failure_attribution/` | **Main code.** Oracle attribution (`run.py`, `run_closed.py`), exact MuJoCo snapshot/restore and fast parallel rollouts (`mw_sim.py`, `sim_worker.py`), task-equivalent goal generation (`goal_variants.py`, `check_goal_variants.py`), equivalence levels L1-L3 (`goal_equiv_mw.py`), patch-aggregated costs (`costs.py`), per-task adapters (`adapters.py`), summaries. |
| `evals/pusht_failure_attribution/` | Push-T version of the attribution, α sweep and cost sweep. |
| `evals/pusht_goal_decomp/` | Push-T cost analysis: block/pusher decomposition, area bias, trade-off pairs, equivalence control. |
| `evals/mw_metric_conflict/`, `evals/mw_hidden_physics/`, `evals/mw_cycles/` | earlier Metaworld diagnostics (closed). |
| `tests/test_fast_rollout*.py` | exactness tests: parallel simulator rollouts must match the env step for step. |

Run everything as modules from `code/jepa-wms/` after `source ../../env/env.sh`,
e.g. `python -m evals.failure_attribution.run_closed --help`.

## `lewm_explorations/`: earlier LeWM study

Frozen and retrained LeWM (stable-worldmodel) on PushT and OGBench cube. `ctrlgeo/` (library: model wrapper,
CEM planner, terminal values, cube simulator, trainer), `scripts/exp1-7` (one script per experiment, each with
a docstring giving its command), `ogbench_shift/` (goal-appearance shift benchmark kit), `run_logs/`. Uses the
conda env `swm` and `third_party/stable-worldmodel` installed in editable mode. Activate with
`source env/lewm.sh` from the project root; set `PUSHT_EXPERT_H5` for a dataset outside the default cache.

## `third_party/stable-worldmodel/`

Snapshot of `galilai-group/stable-worldmodel` (commit `8b2e8c68`). Our additions: `scripts/plan/eval_goalshift.py`
(goal-appearance shift eval), `scripts/plan/eval_meantrap*.py` and `stable_worldmodel/envs/two_room/trap_env.py`,
`scripts/data/*` (colour re-rendering, TwoRoom trap data).
