# NEXT_STEPS: runbook for continuing the project

Written 2026-10-08 for whoever continues (another agent or the user). Read in this order:
`02_REPORT_progress.md` (what was found), this file (what to do), `setup/HANDOFF_from_previous_machine.md` / `setup/MOVING.md` (setup history),
`03_REPORT_area_cost.md` (the closed top-k direction).

Deadline: CVPR 2027 main track. Registration **2026-11-10**, paper **2026-11-16**, supplementary 2026-11-23 (AoE).

---

## 0. Working principles

- **CVPR main track is the only target.** Do not suggest other venues, workshops or tracks.
- Recent papers in the same area do not invalidate the work: take the strongest ones as baselines and show we
  beat them on a clearly demonstrated failure. Old papers (e.g. RAC 2022, LEXA 2021) are context, not blockers.
- Report negative results plainly, with numbers. Kill directions quickly when a pre-stated test fails.
- Smoke-test code on 1-2 episodes before any long run. Give long runs a `timeout` (see pitfalls).

---

## 1. Setup after a reboot (do this first, every session)

1. **Open the checkout** (mount its disk first if it is stored on removable media):
   ```bash
   cd /path/to/jepa-wms-project
   ```
2. **Environment** (conda env `jepa-wms`, torch 2.7.0+cu128 for the RTX 5090):
   ```bash
   source env/env.sh
   R="$JEPAWM_HOME"
   cd "$R/code/jepa-wms"
   python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
   python -m tests.test_fast_rollout_mw --tasks mw-push --trials 2     # must print OK (0.0 / 0 diffs)
   ```
   If the env is broken: `bash "$R/env/setup_env_5090.sh"` (rebuilds it; uses `--no-deps` on purpose).
3. Check nothing is running: `nvidia-smi`, `pgrep -af "run_closed|goal_equiv"`.

---

## 2. Pitfalls already hit (avoid them)

| Pitfall | Fix |
|---|---|
| `pkill -f <pattern>` kills your own shell when the pattern appears in your command line (exit 144) | get PIDs with `pgrep -f`, then `kill <pid>`; or run kills from a script file |
| A one-liner `bash -c 'while pgrep -f "X"…'` waits forever (matches itself) | put waiting loops in a script file |
| Rendering dies after ~9 episodes (EGL contexts pile up) | `goal_equiv_mw.py` / `run_closed.py` close `evaluator.expert_env` each episode; keep doing that in new code |
| Older SimPool runs could hang when a worker died | Worker batches now time out after `JEPAWM_ROLLOUT_TIMEOUT` (default 300 s); also bound full runs with `timeout <sec>` |
| Old interrupted runs can contain a 0-byte `ep_XXX.npz` | New writes are atomic; remove a legacy incomplete file before resuming |
| Two world-model jobs (≈18 GB each) do not fit in 32 GB | run world-model jobs one at a time; simulator-dynamics runs are light (≈2-6 GB) |
| `sim_worker.py` needs spawned workers | run scripts with `python -m …` or from a file, never from stdin |
| Metaworld goal states are not at rest (expert still pushing) | `goal_variants.make_arm_variants` handles it (free objects: rest flat at goal x-y; articulated: start from the goal state, hold/ghost the object while moving the arm) |

---

## 3. Code map (all under `code/jepa-wms/evals/failure_attribution/` unless noted)

| File | Purpose | Key options |
|---|---|---|
| `run_closed.py` | closed-loop Metaworld episodes with planners `wm`, `sim_latent`, `sim_true` | `--fname --out_dir --episodes --planners --cost {l2,topk:K,pmean:P} --alpha --goal_variant k --robot_free` |
| `summarize_closed.py` | success ladder, attribution, first successful replan | `--run_dir` |
| `goal_equiv_mw.py` | offline levels L1-L3 on task-equivalent goals | `--fname --out_dir --episodes --variants --candidates --robot_free` |
| `goal_variants.py` | builds equivalent goals; `robot_free_frame`; `hook_expert_capture` | |
| `check_goal_variants.py` | sanity check + montage for a task: `python -m … <task-suffix>` (e.g. `drawer-close`) | prints rejection reasons and object drift |
| `mw_sim.py` | exact snapshot/restore, robot geoms/bodies, object joints, `SUCCESS_THRESHOLD` per task | add a threshold when adding a task |
| `adapters.py` | per-task proprio / true cost (`MetaworldAdapter` reads `obj_to_target`) | |
| `costs.py` | `PatchAggMPCObjective` (patch aggregations) | add new costs here |
| `run.py`, `summarize.py` | single-plan attribution (Push-T, Wall) | |
| `../pusht_goal_decomp/goal_equiv.py` | Push-T equivalence control (offline) | |
| `../../tests/test_fast_rollout_mw.py` | exactness test | |

Configs: `configs/mw_jepa-wm_{push,drawer-close,door-open,button-press-topdown}.yaml`
(JEPA-WM encoder). `mw_dino-wm.yaml` exists (task line is `  task: mw-reach`).
Results: `artifacts/raw_logs/failure_attribution/{mw_push_costs,goal_equiv_mw_push,goal_equiv_mw_push_robotfree,goal_equiv_closed,drawer_close}/`.

Episode time (simulator dynamics, one planner): ≈ 65-90 s per Metaworld episode. With `--planners wm` (the
real world model): roughly 5-10 min per episode (7 replans × 15 CEM iterations × 300 samples) — measure on 1
episode first.

---

## 4. Experiments to run next (in order)

### E1. Replicate on the second model family (DINO-WM) — ~4 h, light GPU

Why: one model family is not enough; DINO-WM is the other released checkpoint.

```bash
cd $R/configs
for t in push drawer-close; do
  sed -e "s/^  task: mw-reach$/  task: mw-$t/" mw_dino-wm.yaml > mw_dino-wm_$t.yaml
  grep -n "task: mw-$t" mw_dino-wm_$t.yaml   # must print one line
done
cd $R/code/jepa-wms
# smoke (1 episode)
timeout 1800 python -m evals.failure_attribution.goal_equiv_mw --fname $R/configs/mw_dino-wm_drawer-close.yaml \
  --out_dir $R/artifacts/raw_logs/failure_attribution/dino/smoke --episodes 1
```
Then the full set (write it as a script file, run with `nohup … &`):
```bash
D=$R/artifacts/raw_logs/failure_attribution/dino
timeout 3600 python -m evals.failure_attribution.goal_equiv_mw --fname $R/configs/mw_dino-wm_drawer-close.yaml --out_dir $D/drawer_goal_equiv --episodes 24
timeout 3600 python -m evals.failure_attribution.goal_equiv_mw --fname $R/configs/mw_dino-wm_push.yaml        --out_dir $D/push_goal_equiv   --episodes 24
for spec in "released|--planners sim_latent sim_true" "variant2|--planners sim_latent --goal_variant 2" "variant3|--planners sim_latent --goal_variant 3" "robot_free_a0|--planners sim_latent --robot_free --alpha 0"; do
  name=${spec%%|*}; args=${spec#*|}
  timeout 5400 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_dino-wm_drawer-close.yaml --out_dir $D/drawer_$name --episodes 24 $args >> $D/drawer_$name.log 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir $D/drawer_$name > $D/drawer_${name}_summary.txt
done
```
Read: `$D/*_goal_equiv/summary.json` (L1-L3) and `$D/drawer_*_summary.txt` (L4). Compare paired success per
episode with the eval goal exactly as in REPORT_progress §2.4 (exact McNemar = binomial test on lost vs gained).

Decision: if DINO-WM shows the same pattern (flips ≥ 25%, top-1 agreement ≤ 20%, L4 drop on ≥ 1 variant),
the collapse generalises across models. If not, report it and investigate why (encoder normalisation differs).

### E2. Strengthen L4 on drawer-close — ~4 h

Why: the current L4 effect is one significant variant on 22 episodes.

- More variants: run `--goal_variant 5`, `6`, `7`, `8` (sim_latent, 24 episodes each; the generator builds
  variants 1..k deterministically per episode, so variant k is the same arm pose every time).
- More episodes: `--episodes 48` for the eval goal and variants 2, 3 (runs resume, so re-run with the larger
  number into the same out_dir).
- Report per variant: success, lost/gained vs eval goal on the same episodes, final `obj_to_target`, and
  correlate the drop with how far the goal arm is from the eval goal arm (`goal_hand` is saved per episode).

### E3. Real world-model planner on drawer-close (L4 without simulator dynamics) — ~6-10 h

Why: reviewers will ask whether the effect holds with the actual planner, not only with oracle dynamics.

```bash
# measure time per episode first
timeout 3600 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_drawer-close.yaml \
  --out_dir $R/artifacts/raw_logs/failure_attribution/drawer_close/wm_smoke --episodes 1 --planners wm
```
Then `--planners wm` with `--goal_variant 0` (no flag) and `--goal_variant 3`, 24 episodes each (one job at a
time). Note: `--robot_free` cannot be used with `wm`.

### E4. A third object task without floor/ceiling effects — ~1 day

Why: push is floored (0-8%), drawer-close is near ceiling (100%). We need a task where the released cost
works moderately (30-80%) so L4 effects are measurable.

Candidates (Metaworld v3 goal-observable, success depends only on the object): `window-close`,
`window-open`, `drawer-open`, `handle-press-side`, `faucet-open`, `coffee-push`, `plate-slide`,
`sweep-into`. For each candidate:
1. make a config: `sed -e "s/^  task: mw-reach-wall$/  task: mw-<task>/" mw_jepa-wm.yaml > mw_jepa-wm_<task>.yaml`
2. read its success rule in `metaworld/envs/sawyer_<task>_v3.py` (`evaluate_state`: `success` and
   `obj_to_target`) and add the threshold to `mw_sim.SUCCESS_THRESHOLD`
3. `python -m evals.failure_attribution.check_goal_variants <task>`: needs ≥ 3 arm variants per episode and
   small object drift; look at `logs/failure_attribution/goal_variants_check_<task>.png`
4. `run_closed --planners sim_latent sim_true --episodes 12`: keep the task if sim_latent is 30-80% and sim_true
   is clearly higher
5. then run L1-L3 (`goal_equiv_mw`) and L4 (variants) as for drawer-close.
Exclude tasks whose goal needs the arm (button presses, anything spring-loaded) from the "irrelevant" set, but
keep one (button-press-topdown) as the counter-example: there, moving the arm out of the goal SHOULD change
success. Measure it with `run_closed --goal_variant k` (the generator fails to keep the button pressed; document
that this is because the arm is task-relevant, which is the point).

### E5. Baselines (on Push-T, drawer-close, button, and the E4 task)

All must be evaluated with the same protocol (L1-L4) and the same planner budget.

| Baseline | What it does | How to implement here | Expected failure |
|---|---|---|---|
| Released latent cost | mean patch MSE + α·proprio | `--cost l2` | equivalence violation |
| Proprio weight α | DINO-WM / JEPA-WM's own fix | `--alpha 10`, `--alpha 0` | cannot remove the arm's image term |
| Robot-free oracle (upper bound of fixed masking, RAC-style) | goal and candidates without the robot | `--robot_free` (sim dynamics only) | breaks tasks where the robot matters (Push-T, button) |
| Fixed robot-mask cost (non-oracle RAC-style) | drop robot patches from the cost | add `--cost robotmask` to `costs.py`: from the goal snapshot render a robot-only mask (`render` vs `render_hidden` pixel difference → 16×16 patch mask), average the patch MSE only over non-robot patches of the goal **and** each candidate (candidate masks from the simulator render in `sim_latent`; for `wm`, use the goal mask only) | same as above |
| Temporal / reachability distance (LEXA, TD-JEPA style) | learned time-to-goal on frozen features | train a 2-layer MLP on frozen encoder features of offline Metaworld trajectories (`artifacts/datasets/Metaworld/data/*.parquet`, inspect the columns first) to regress the step gap between frames of the same trajectory; use it as the CEM cost | likely inherits arm dominance |
| Controllability split (CF-JEPA style) | ignore controllable (robot) features | per state, patches whose features change under small action perturbations of the world model → "controllable" → masked | masks the pusher in Push-T and the arm in button, both of which matter |

Kill test for the paper's argument: show a 2×N table where every fixed rule (mask robot / keep robot) is
right on one side of {drawer, push} vs {Push-T, button} and wrong on the other.

### E6. Method development (start by ~Oct 15 at the latest)

Requirement: infer, per task, which goal factors matter; fix drawer (and the E4 task) without breaking Push-T
or button; beat all E5 baselines. Keep 2-3 candidates alive and kill them by pre-stated tests.

**M1. Goal-equivalence contrastive projection (most concrete).** Train a small head g(·) on frozen patch
features (e.g. patch-wise linear map + attention pooling) so that task-equivalent goals map close and
non-equivalent goals map apart; plan with ‖g(z_T) − g(z_goal)‖². Supervision comes free from the simulator:
`goal_variants.py` gives equivalent pairs (same object state, different arm) and non-equivalent pairs (object
state changed; for Push-T, pusher changed). Train on a set of tasks, evaluate on held-out episodes and,
critically, held-out tasks. Kill test: drawer L4 back to ≥ 95% on all variants AND Push-T control unchanged
(flip rate within 10% of the official one) AND button arm-relevance preserved.

**M2. Relevance from the task's own evidence (training-free).** Estimate per-patch relevance from what
changes between the start image and the goal image *beyond what the robot alone explains*: render or
imagine (world model) the start scene with only the robot moved to its goal pose; patches that still differ
from the goal are object-relevant; weight the cost by that map. Push-T: the pusher is the robot but its goal
pose matters → needs proprio/goal semantics, so expect M2 to need α > 0 for agent-is-object tasks; test.

**M3. Success-supervised patch weighting.** Learn per-task patch weights from a few success/failure labels
collected in the simulator (privileged at training time only). Simplest, strongest-supervision baseline for
M1/M2.

Report for each candidate: L1-L4 on drawer, push (offline), Push-T control, button, E4 task; compute cost.

---

## 5. Timeline (suggested)

| Dates | Work |
|---|---|
| Oct 8-10 | E1 (DINO-WM), E2 (more variants/episodes), start E4 task search |
| Oct 10-13 | E3 (real wm planner), E4 finished, E5 baselines implemented |
| Oct 13-27 | E6 method candidates, kill tests, pick one |
| Oct 27-Nov 7 | full experiments with the chosen method (2 models × ≥4 tasks, baselines, ablations, seeds) |
| Nov 7-10 | register; figures (equivalence montage, L1-L4 chain, task-dependence table) |
| Nov 10-16 | writing, final runs only for claims; submit |

---

## 6. What to keep updating

- `02_REPORT_progress.md`: add each finished result (numbers, n, p-values) to the relevant table.
- Keep every run's `*_summary.txt` and `summary.json` next to its episodes.
- When a direction is killed, write one paragraph on why (as in `03_REPORT_area_cost.md`).
