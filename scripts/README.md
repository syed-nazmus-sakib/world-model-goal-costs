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

## Running on a cluster with Conda (`setup/`, three steps)

| Step | Script | What it does |
|---|---|---|
| 1 | `setup/01_conda_env.sh` | creates the Conda envs `jepa-wms` (Metaworld / Push-T / Wall planning) and `swm` (LeWM), with a GPU and rendering smoke test |
| 2 | `setup/02_download.sh [metaworld ckpt lewm]` | Metaworld dataset and released checkpoints into `artifacts/`; LeWM checkpoints and datasets into `~/.stable_worldmodel/` |
| 3 | `setup/03_run.sh <smoke\|mw_state\|lewm_cube\|lewm_pusht>` | `smoke` first (about 5 min), then the experiments: Metaworld state head on 9 tasks, LeWM Cube / PushT heads over 3 seeds |

Results land in `artifacts/raw_logs/<experiment>/`; the success lines are in each folder's `queue.log` / `summary.txt`.

## Pod scripts for the learned goal costs (`pod/`, 2026-10-22)

These run on a RunPod pod with the project at `/root/jepa-wms-project` (override with `JEPAWM_HOME`) and write
to `/root/out/...`. Order on a new pod: `check_egl.sh`, then the setup script, then one experiment script.
Results and findings: `docs/10_METHOD_C_D_RESULTS.md`.

| Script | What it does |
|---|---|
| `pod/check_egl.sh` | first check on a new pod: NVIDIA EGL present and render time per frame (CPU fallback is about 15x slower) |
| `pod/fetch_mw_data.sh` | Metaworld dataset from Hugging Face onto the pod (about a minute; the model build needs it for the action normalisation) |
| `pod/setup_jwm_venv.sh`, `pod/jwm_env.sh` | JEPA-WM uv venv (`/root/venv_jwm`); source `jwm_env.sh` before the Metaworld scripts |
| `pod/setup_swm_venv.sh` | LeWM / stable-worldmodel uv venv (`/root/venv_swm`), dataset notes for TwoRoom and Cube |
| `pod/run_lewm_methodC.sh <pusht\|tworoom\|cube> [seed]` | Method C on LeWM: latents, state and time heads, released / min-over-plan / head runs at offsets 50 and 75 |
| `pod/run_mw_time_head.sh` | Method C on Metaworld: time head on the expert dataset, closed loop head vs released per task |
| `pod/run_mw_methodD_pilot.sh` | Method D pilot: labelled encodings, four heads, offline test, gated closed loop |
| `pod/run_mw_state_head.sh` | state head on any list of Metaworld tasks, closed loop head vs released (pre-stated rule in the header) |
| `pod/run_mw_reach_oracle.sh` | reach-then-change Stage 1: oracle reach term with the released, true and robot-free change terms, 4 tasks + reach-wall |
| `pod/run_mw_reach_image.sh` | reach-then-change Stage 2: robot-mask frames, robot map, smoke, 12-episode closed loop from images only; door-open oracles with the fixed true cost |
| `pod/backup_and_stop.sh <pod_id> <host> <port> <remote_out> <name>` | laptop side: wait for `ALL_DONE`, copy results home and into `artifacts/`, then stop the pod (key from `.env`, never printed) |
