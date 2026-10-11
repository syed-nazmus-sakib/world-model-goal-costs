# Results

Small, curated files (summaries, JSON, plots) behind the numbers in `docs/`. Per-episode raw files
(`ep_XXX.npz`, candidate dumps, datasets) are in `artifacts/raw_logs/` and `artifacts/lewm_explorations_raw/`
with the same folder names.

## `jepa_wms/`: released JEPA-WM / DINO-WM planners (main study)

| Folder | Experiment | Finding (see docs/01_ASSESSMENT.md) |
|---|---|---|
| `pt_repro/` | reproduction of the released Push-T eval (24 episodes, both models) | sanity |
| `pusht_failure_attribution/{jepa_wm,dino_wm}/` | `wm` / `sim_latent` / `sim_true` on 96 Push-T episodes; α sweep and cost sweep | F1, F5 |
| `failure_attribution/wall_{jepa,dino}_wm*` | same attribution on Wall | F1 |
| `failure_attribution/mw_push_costs/` | Metaworld push, simulator dynamics, latent vs true cost, top-k, α | F1, F4, F8 |
| `failure_attribution/goal_equiv_mw_push*/` | task-equivalent goals on push, levels L1-L3 (with and without robot) | F2, F4 |
| `failure_attribution/goal_equiv_closed/` | push, closed loop with robot-free and arm-moved goals (L4) | F2, F4 |
| `failure_attribution/drawer_close/` | drawer-close: L1-L3 (`goal_equiv/`) and L4 (`released`, `variant2-4`, `robot_free_a0`) | **F2** |
| `failure_attribution/goal_variants_check_*.png` | montages of the generated equivalent goals per task | F2, F3 |
| `failure_attribution/puck_sensitivity*.json` | latent cost vs puck displacement (monotonicity) | F4 |
| `pusht_goal_decomp/` | Push-T: block/pusher decomposition, area bias, trade-off pairs, Push-T equivalence control | F3, F5, F8 |
| `mw_metric_conflict/` | earlier: one global learned metric vs per-task metrics (94.5% vs 96.9% ranking accuracy) | closed |
| `mw_hidden_physics/` | earlier: identical pixels, different hidden physics: futures barely change | closed |
| `mw_cycles/` | earlier: reversibility of action cycles | closed |
| `failure_attribution/{smoke_*,regress_*}` | smoke tests and the regression check after moving to the RTX 5090 | sanity |

## `lewm_explorations/`: earlier LeWM study (stable-worldmodel)

| File / folder | Experiment | Outcome |
|---|---|---|
| `exp1/`, `exp2/`, `PREREGISTRATION_confirm.md`, `confirm_stats.txt`, `mechanism.txt` | controllability-geometry terminal value for CEM | pre-registered confirmation failed (+2.6 pt, CI includes 0): killed |
| `exp3_subspace.txt`, `exp3/` | post-hoc projection against goal-appearance shift | offset isotropic: killed |
| `goalshift/goalshift_results.txt` | goal-appearance shift on PushT (4 seeds x 50 goals) | 90% → 7% (F6) |
| `exp4/` | counterfactual error of off-support actions (frozen LeWM) | 1.89x error |
| `exp5/` | PushT: expert vs decorrelated vs branched training data (fixed budget) | decorrelation 41.3% → 72.3%; branching no gain (F7) |
| `exp6/` | OGBench cube: same data study | CF error 20x lower, planning unchanged (F7) |
| `exp7_cube_cost/` | cube: latent cost vs top-k vs true cost with exact dynamics | 40% / 38-44% / 54% |

Each `PREREGISTRATION.md` states the decision rule fixed before the run.

## Maintenance corrections (2026-10-12)

Wall JEPA-WM attribution now counts each failed episode once: 11 dynamics,
10 cost and 3 search (45.8%, 41.7%, 12.5%). The prior summary counted episode 12
in both dynamics and search. Success rates and saved episodes are unchanged.
The empty `goal_equiv_closed/robot_free_a0` report was rebuilt from 23 completed
episodes (0 successes); its recovered metadata records the intended 24-episode
run and explains how the options were recovered. Original derived reports are
preserved locally under `artifacts/maintenance_backup_2026-10-12/`.
