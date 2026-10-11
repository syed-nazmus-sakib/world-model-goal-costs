# Launch scripts

Each script sources `env/env.sh`, runs from `code/jepa-wms/` and writes to `artifacts/raw_logs/`. Finished
episodes are skipped on rerun. Run one world-model job at a time (about 18 GB of GPU memory each).
Paths are quoted for checkouts containing spaces. Set `JEPAWM_LOGS` before activation to change the output
root. Launchers create their output folders and stop when a command fails.

| Script | Experiment | Result folder |
|---|---|---|
| `run_repro.sh` | reproduce the released Push-T eval (both models, 24 episodes) | `pt_repro/` |
| `run_attribution.sh` | Push-T oracle attribution, 96 episodes, JEPA-WM then DINO-WM | `pusht_failure_attribution/` |
| `run_attribution_wall.sh` | Wall oracle attribution | `failure_attribution/wall_*` |
| `run_cost_sweep.sh`, `run_cost_sweep_alpha.sh` | Push-T, simulator dynamics: patch aggregations (top-k, power mean) and α | `pusht_failure_attribution/jepa_wm/cost_sweep/` |
| `run_alpha_wm.sh` | real world-model planner at α = 10 on Push-T | `pusht_failure_attribution/` |
| `run_mw_push_costs.sh` | Metaworld push: latent vs true cost, top-k, α | `failure_attribution/mw_push_costs/` |
| `run_alpha0_mw.sh` | Metaworld push without the proprio term | `failure_attribution/goal_equiv_closed/` |
| `run_goal_equiv_mw.sh` | push: task-equivalent goals, offline L1-L3 | `failure_attribution/goal_equiv_mw_push*/` |
| `run_goal_equiv_closed.sh` | push: equivalent / robot-free goals, closed loop | `failure_attribution/goal_equiv_closed/` |
| `run_drawer_close.sh` | drawer-close: L1-L3 and released / robot-free closed loop | `failure_attribution/drawer_close/` |
| `run_drawer_variants.sh` | drawer-close: closed loop with arm-moved goals 2-4 (L4) | `failure_attribution/drawer_close/variant*` |
| `run_mw_collect.sh`, `run_mw_fit.sh`, `run_mw_highcap.sh` | earlier Metaworld metric-conflict pilot | `mw_metric_conflict/` |
| `resume_runs.sh` | resumed two runs interrupted by the machine move | — |
| `run_queue_next.sh` | prepared queue for the next experiments (never launched) | — |
| `relocate_configs.sh` | convert legacy configs to environment placeholders; current configs need no rewrite | — |
| `check_repo.py` | source/config/result checks; add `--check-assets` to verify local data and weights | — |
| `legacy/make_archives.sh` | export the files selected by Git into a source zip | `artifacts/exports/` |

The LeWM study has its own entry point: `code/lewm_explorations/run_all.sh` (see its README).
