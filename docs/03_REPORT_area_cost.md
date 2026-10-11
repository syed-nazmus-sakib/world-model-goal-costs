# The latent goal cost of world-model planners: what fails, what we tried, and why top-k failed

Written 2026-10-06. Covers the work done after the project moved to the RTX 5090 machine (see `setup/HANDOFF_from_previous_machine.md`
for the earlier history and `setup/MOVING.md` for the setup). Target: CVPR 2027 main track.

---

## 1. Setting

The planners studied here are the released goal-conditioned visual world-model planners of
`facebookresearch/jepa-wms` (JEPA-WM and DINO-WM checkpoints for Push-T, Wall and Metaworld). One planning
step works like this:

1. The goal image and the current image are encoded by a **frozen DINOv2 ViT-S/14**, giving a 16 x 16 grid
   of **256 patch features** (384 numbers each), plus a small proprioceptive embedding (robot position).
2. A **world model** (predictor) predicts the patch features of the scene after a candidate action sequence.
3. **CEM** samples 300 candidate action sequences, scores each predicted final scene against the goal, keeps
   the best ones, refits its sampling distribution, and repeats (Push-T: 30 iterations; Metaworld: 15).
4. The best plan is executed (Push-T, Wall: one open-loop plan; Metaworld: 15 steps, then replan).

The **score** ("latent goal cost") used by all of these models is

```
cost = mean over patches and features of (pred_patch - goal_patch)^2   +   alpha * mean (pred_proprio - goal_proprio)^2
```

with `alpha = 0.1` in the released configs.

---

## 2. The method we used to find failures: oracle attribution

To find out *which part* of the planner causes a failure, every episode of a released eval is played again
with parts of the system replaced by the simulator (an oracle), keeping the start state, goal, CEM budget and
replanning schedule fixed:

| Planner | Dynamics | Cost |
|---|---|---|
| `wm` (the released eval) | world model | latent cost |
| `sim_latent` | **exact simulator** | latent cost |
| `sim_true` | exact simulator | **true-state cost** (distance of the true state to the goal state) |

- If `sim_latent` succeeds where `wm` failed, the failure is due to the **dynamics** (world-model error).
- If `sim_latent` also fails but `sim_true` succeeds, the failure is due to the **cost** (the latent score
  ranks futures wrongly even when the futures are perfect).
- If `sim_true` fails too, it is a **search** failure (CEM cannot find a good plan in its budget).

Inside CEM, the simulator replaces the world model through exact fast rollouts in worker processes
(`evals/failure_attribution/sim_worker.py`, `mw_sim.py`). These were verified to reproduce the env stack of
the eval **bit-exactly** (0.0 state and 0 pixel difference; `tests/test_fast_rollout.py`,
`tests/test_fast_rollout_mw.py`), and re-planned episodes on the 5090 are bit-identical to the ones saved on
the old 3090 Ti.

Code: `evals/failure_attribution/run.py` (single-plan evals: Push-T, Wall), `run_closed.py` (replanning
evals: Metaworld), `summarize.py`, `summarize_closed.py`.

---

## 3. The failures we found

### 3.1 Success ladders (96 released episodes each, one checkpoint, one seed)

| Task, model | `wm` | `sim_latent` (perfect dynamics, latent cost) | `sim_true` (perfect dynamics, true cost) |
|---|---|---|---|
| Push-T, JEPA-WM | 64.6% | 64.6% | 90.6% |
| Push-T, DINO-WM | 65.6% | 61.5% | 90.6% |
| Wall, JEPA-WM | 75.0% | 83.3% | 92.7% |
| Wall, DINO-WM | 63.5% | 75.0% | 92.7% |
| Metaworld push, JEPA-WM encoder (24 ep.) | — | **8.3%** (2/24) | **83.3%** (20/24) |
| Cube (OGBench single, frozen DINOv2, 50 ep.) | — | 40.0% | 54.0% |

Attribution of the world-model failures:

| Task, model | dynamics | cost | search |
|---|---|---|---|
| Push-T, JEPA-WM (34 failures) | 15% | **59%** | 26% |
| Push-T, DINO-WM (33 failures) | 18% | **55%** | 27% |
| Wall, JEPA-WM (24 failures) | **45.8%** | 41.7% | 12.5% |
| Wall, DINO-WM (35 failures) | **46%** | 37% | 17% |

**Failure 1, the latent cost is the main bottleneck when objects must be manipulated.** Replacing the world
model with a perfect simulator does not raise success on Push-T, and the gap to the true-state cost is large
everywhere: Push-T 65% vs 91%, Metaworld push 8% vs 83%, Cube 40% vs 54%. On Wall (navigation), the
dynamics matter about as much as the cost.

### 3.2 What the cost gets wrong on Push-T: the pusher is underweighted

- In failed Push-T episodes the **block** usually reaches the goal (89% within tolerance) but the
  **pusher** does not (68%); failed episodes end with the pusher a median 87-96 px from its goal.
- The proprio term contributes under 1% of the final cost at `alpha = 0.1`.

**How we measured it (offline, no planning).** `pusht_goal_decomp` built 64 goals, each with a grid of
23 block variants x 19 pusher variants placed exactly in the simulator, rendered and encoded with both
released encoders (patch features saved). From these grids:

- `evals/pusht_goal_decomp/tradeoff_check.py`: regress the cost on block and pusher position errors
  (translation-only cells), and count how often the cost disagrees with the official success distance on
  **trade-off pairs** (one state has the better block, the other the better pusher).

| Cost | pusher weight per px relative to block (task needs 1.00) | wrong on trade-off pairs (2.16 M pairs) |
|---|---|---|
| released mean, JEPA-WM / DINO-WM | **0.33 / 0.37** | **22.1% / 21.0%** |

So the cost values one pixel of pusher error at about a third of one pixel of block error, while Push-T
success weighs them equally. CEM therefore trades a large pusher error for a small block improvement.

- `evals/pusht_goal_decomp/repro_area_bias.py`: compares "block perfect, pusher displaced 40-160 px
  (official failure)" with "pusher perfect, block displaced 5-10 px (official success)". Here the mean cost is
  usually right (wrong in 0-8% of pairs): a pusher moved by 40+ px changes about 18 of 256 patches, more than a
  10 px block shift (about 2). **Correction to our first explanation:** the pusher is not invisible to the cost;
  what is wrong is the *exchange rate* between pusher and block errors (above), which matters in the
  intermediate regime where CEM actually operates. Figure: `artifacts/raw_logs/pusht_goal_decomp/area_bias_*.png`.

### 3.3 What the cost gets wrong on Metaworld push: the robot dominates and the object is never moved

With perfect dynamics, the latent cost **never pushes the puck**:

| Cost (simulator dynamics, 24 episodes) | success | episodes where the puck moved > 2 cm | median puck displacement |
|---|---|---|---|
| released latent cost | 2/24 | 3/24 | **0.000 m** |
| true-state cost | 20/24 | 21/24 | 0.206 m |

The goal image is the last frame of an expert episode and also shows **the robot arm in its final pose**. The
arm occupies far more of the image than the puck. To move the puck the arm must first leave that pose, go to
the puck and push it; every such candidate looks *worse* to an image-goal cost until the push is complete, so
CEM converges on moving the arm toward its goal pose and never discovers pushing. The true-state cost ignores
the arm and rewards puck motion directly. This is the feature-space version of the failure Visual Foresight
(Ebert et al., 2018) described for pixel costs: *"the planner matches the arm position with its position in
the goal image, disregarding smaller objects."*

### 3.4 Other observations

- **Goal appearance.** On LeWM / PushT (our earlier work in `code/lewm_explorations/`), changing only the background
  colour of the goal image drops success from 88% to about 6%; the offset is spread over the whole latent.
- **Cube.** With a perfect world model, LeWM's latent distance ranks candidate actions by true cube progress
  at a Spearman correlation of only about 0.24 (`results/lewm_explorations/exp6/`).

---

## 4. Our initial idea: "area dominance" and the top-k cost

### 4.1 The hypothesis

The released cost averages the mismatch over all 256 patches, so each scene element contributes in
proportion to the image **area** it covers. Small task-critical objects (the Push-T pusher, a puck, a small
cube) would then be underweighted relative to large regions (the block, the arm, the background).

### 4.2 The fix: area-invariant aggregation (top-k)

Keep the encoder, world model, planner and all settings; change only how the patch mismatches are combined
(`evals/failure_attribution/costs.py`):

```python
d = (goal_feats - pred_feats).pow(2).mean(-1).flatten(2)        # (time, candidates, 256) per-patch mismatch
cost = d.topk(K, dim=-1).values.mean(-1) + alpha * proprio_term   # ours:   mean of the K largest mismatches
# released:  cost = d.mean(-1) + alpha * proprio_term              #         mean of all 256 mismatches
```

- **top-k** (K = 4 or 8): the score is the average of the K worst-matching patches of each candidate, so a
  badly placed small object counts as much as a badly placed large one. Which patches are selected is
  decided separately for every candidate.
- **power mean** (p = 4): `(mean_p d_p^4)^(1/4)`, a smooth version of the same idea (generalised-mean pooling).
- **Properties:** no retraining (the released weights are used as downloaded), no extra labels, masks or
  simulator state (only the goal image, the model's own predicted features and the released proprio term),
  and a one-expression change in any planner that scores patch features. It does not apply to models that
  use a single pooled embedding (e.g. LeWM's CLS token).

### 4.3 Baselines we identified (verified on arXiv)

| Paper | What it does | Needs |
|---|---|---|
| Temporal-Distance JEPA (2607.25337, MIT code) | learned temporal cost | retrains the world model; its own cost hurts Push-T (69% vs 86%) |
| SCALE (2608.16287) | aligns latent distances to simulator state | privileged state, retraining |
| LEAP (2609.03294) | state-decoder cost, gradient planner | privileged goal state |
| D-JEPA (2609.24749, Apache code) | learned reranker of candidates | simulator-labelled outcomes |
| CGS (2609.35603) | action/latent geometry straightening | retraining |
| "Keeping JEPA WMs plannable when little of the frame moves" (2610.03137) | inverse-dynamics loss for small moving entities | retraining |
| Visual Foresight (2018), designated-pixel / keypoint costs | pixel or point costs for small objects | user-selected points |

None of them changes how patch mismatches are aggregated at planning time; that was the opening for top-k.

---

## 5. Experiments with top-k, and results

### 5.1 Offline ranking (no planning)

`evals/pusht_goal_decomp/cost_bench.py` and `tradeoff_check.py`, on the 64 Push-T goal grids, both released
encoders. "Rank vs official" = fraction of state pairs ordered like the official success distance.

| Cost | rank vs official (JEPA-WM / DINO-WM) | AUC success vs failure | pusher/block weight per px | wrong on trade-off pairs |
|---|---|---|---|---|
| released mean | 0.735 / 0.747 | 0.916 / 0.923 | 0.33 / 0.37 | 22.1% / 21.0% |
| mean + alpha 10 (proprio) | 0.809 / 0.817 | 0.935 / 0.940 | — | — |
| **top-8 patches** | **0.818 / 0.831** | **0.964 / 0.968** | 0.65 / 0.77 | **13.7% / 13.5%** |
| top-4 patches | 0.820 / 0.830 | 0.964 / 0.968 | 0.72 / 0.84 | 13.6% / 14.1% |
| power mean p=4 | 0.814 / 0.829 | 0.962 / 0.965 | — | — |

Offline, top-k was clearly better: it doubled the relative weight of the pusher and cut wrong trade-off
decisions from about 22% to about 14%, without proprio and without training.

### 5.2 Closed loop, Push-T (simulator dynamics, so only the cost changes; 96 episodes)

`evals/pusht_failure_attribution/alpha_sweep.py` and `cost_sweep.py` (same episodes, start, goal, CEM samples;
`l2` reproduces the `sim_latent` planner bit-exactly). Paired exact McNemar tests.

| Cost | success | pusher < 20 px | block < 20 px | vs released | vs alpha 10 |
|---|---|---|---|---|---|
| released mean, alpha 0.1 | 64.6% | 67.7% | 89.6% | — | 3 gained / 17 lost |
| mean, alpha 1 | 70.8% | 74.0% | 90.6% | +7 / -1 (p 0.07) | |
| **mean, alpha 10** | **79.2%** | 86.5% | 89.6% | +17 / -3 (p 0.003) | — |
| mean, alpha 100 | 68.8% | 100% | 70.8% | n.s. | worse (p 0.03) |
| top-8, alpha 0.1 | 72.9% | 76.0% | 90.6% | +12 / -4 (p 0.08) | 6 / 12 |
| top-4, alpha 0.1 | 71.9% | 74.0% | 88.5% | +11 / -4 (p 0.12) | 7 / 14 |
| power mean 4, alpha 0.1 | 72.9% | 75.0% | 88.5% | +14 / -6 (p 0.12) | 7 / 13 |
| top-8, alpha 1 | 74.0% | 77.1% | 89.6% | +13 / -4 (**p 0.049**) | 6 / 11 |
| top-8, alpha 10 | 77.1% | 81.2% | 90.6% | | +4 / -6 (p 0.75) |
| true-state cost | 90.6% | | | | |

### 5.3 Closed loop, Metaworld push (simulator dynamics, replanning, 24 episodes)

`run_closed.py --planners sim_latent [sim_true] --cost ... --alpha ...` on `configs/mw_jepa-wm_push.yaml`
(the released Metaworld JEPA-WM encoder; task switched from reach-wall to push).

| Cost | success | puck moved > 2 cm | median puck move |
|---|---|---|---|
| released mean, alpha 0.1 | 2/24 | 3/24 | 0.000 m |
| top-8, alpha 0.1 | 2/24 | 3/24 | 0.000 m |
| mean, alpha 10 | 0/24 | 1/24 | 0.000 m |
| top-8, alpha 10 | 1/24 | 3/24 | 0.000 m |
| true-state cost | **20/24** | 21/24 | 0.206 m |

### 5.4 Closed loop, Cube (OGBench cube-single, exact simulator, frozen DINOv2 patches, 50 episodes)

`code/lewm_explorations/scripts/exp7_cube_cost.py`. Tasks: held-out expert start, goal = expert state 25 steps later with
the cube moved at least 5 cm; success = cube within 4 cm of its goal at any step of a 50-step budget. CEM:
300 samples, 30 iterations, horizon 25 steps, replan every 25 steps.

| Cost | success |
|---|---|
| patch mean (DINO-WM style) | 40.0% |
| top-8 | 38.0% |
| top-4 | 40.0% |
| mean + alpha 10 (effector position) | 40.0% |
| top-8 + alpha 10 | 44.0% |
| true-state cost (cube distance) | 54.0% |

---

## 6. Why top-k failed

1. **Better offline ranking did not become better plans.** Top-k ranks pairs of fixed states better, but CEM
   needs a cost whose landscape leads from the start to the goal. The offline bench only scores states near
   the goal; it says nothing about the path there. (ARC-Bench, 2609.05461, reports the same disconnect for
   ranking metrics.)
2. **On Push-T the small object is the robot.** The pusher's position is in the proprioceptive input, so
   raising alpha fixes its underweighting directly and more precisely (79.2%). Top-k only partly corrects the
   exchange rate (0.33 to 0.65-0.77, not 1.0), and adds nothing on top of alpha 10 (77.1% vs 79.2%).
3. **On Metaworld push and Cube the failure is not about area.** The cost is dominated by the **robot's own
   goal pose**, which conflicts with manipulating the object. To push the puck, the arm must temporarily
   *increase* its mismatch with the goal image. Top-k still contains the arm's mismatch among the largest
   patches, so the planner still prefers to move the arm toward its goal pose; the puck is never touched
   (median displacement 0.000 m for every image cost). Reweighting patches cannot create an incentive to
   accept a temporarily worse score. Raising alpha makes it worse (0/24), because it pulls the arm toward its
   goal pose even harder.
4. **Area invariance has known weaknesses that never got to matter:** it amplifies mismatches from moving
   distractors and from world-model prediction errors (which concentrate at contacts), and it does not
   cancel uniform shifts such as a background colour change.

**Verdict:** the top-k cost is closed as a method. The failure it targeted (pusher underweighting on Push-T)
is real but is already addressed by the proprio weight; the larger failure (Metaworld push, Cube) has a
different cause.

---

## 7. What still stands, and the next test

Robust findings:

- With perfect dynamics, image-goal costs lose badly to the true-state cost exactly when an object other
  than the robot must be moved: Metaworld push 8% vs 83%, Cube 40% vs 54%, Push-T 65% vs 91%.
- On manipulation the cost is the main failure source (55-59% of Push-T failures); on navigation (Wall) the
  dynamics are (46%).
- On Metaworld push the cause is visible: the planner matches the arm's goal pose and never moves the object.

Next kill test (proposed, not started): **a cost that ignores the robot.**

1. Upper bound: mask the robot's patches using the simulator's exact segmentation, and compare only object
   and scene patches. If even a perfect robot mask does not bring Metaworld push and Cube close to the
   true-state cost, this direction is closed.
2. Only if (1) works: replace the perfect mask by a label-free one (for example, patches whose features are
   predictable from the robot's proprio, or that change when only the robot moves), so the method stays
   training-free.

---

## 8. Files and commands

All paths are relative to the project root unless noted; `source env/env.sh`
and run from `code/jepa-wms/`.

| What | Where |
|---|---|
| Area-invariant costs | `code/jepa-wms/evals/failure_attribution/costs.py` |
| Push-T offline bench / reproduction / exchange rate | `code/jepa-wms/evals/pusht_goal_decomp/{cost_bench,repro_area_bias,tradeoff_check}.py` |
| Push-T closed-loop cost sweeps | `code/jepa-wms/evals/pusht_failure_attribution/{alpha_sweep,cost_sweep}.py`; results `artifacts/raw_logs/pusht_failure_attribution/jepa_wm/{alpha_sweep,cost_sweep}/` |
| Attribution runners | `code/jepa-wms/evals/failure_attribution/{run,run_closed,summarize,summarize_closed,mw_sim}.py` |
| Wall attribution | `artifacts/raw_logs/failure_attribution/wall_{jepa,dino}_wm_summary.txt` |
| Metaworld push costs | `artifacts/raw_logs/failure_attribution/mw_push_costs/` (script `scripts/run_mw_push_costs.sh`) |
| Cube costs | `code/lewm_explorations/scripts/exp7_cube_cost.py`, results `results/lewm_explorations/exp7_cube_cost/` |
| Exactness tests | `code/jepa-wms/tests/test_fast_rollout.py`, `code/jepa-wms/tests/test_fast_rollout_mw.py` |

```bash
python -m evals.pusht_goal_decomp.tradeoff_check jepa_wm dino_wm
python -m evals.pusht_goal_decomp.repro_area_bias --model jepa_wm --figure area_bias.png
python -m evals.pusht_failure_attribution.cost_sweep --fname $R/configs/pt_jepa-wm_repro.yaml \
    --run_dir $R/artifacts/raw_logs/pusht_failure_attribution/jepa_wm --costs topk:8 --alpha 0.1
python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_push.yaml --out_dir <dir> \
    --episodes 24 --planners sim_latent sim_true --cost topk:8
```

Caveats: one released checkpoint per model and one seed; 24 episodes on Metaworld push (confidence intervals
about +/-15 points), 50 on Cube, 96 on Push-T and Wall. The Metaworld and Cube cost tests use simulator
dynamics only; the Cube test uses raw DINOv2 patch features with no released world model for that task.
