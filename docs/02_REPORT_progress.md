# Progress report: failures of the latent goal cost in visual world-model planning

Written 2026-10-08. Covers all work on this machine (RTX 5090) since the project moved here on 2026-10-05.
Target: **CVPR 2027 main track** (registration 2026-11-10, paper 2026-11-16, supplementary 2026-11-23, AoE).
Earlier history: `setup/HANDOFF_from_previous_machine.md`. Top-k cost study in detail: `03_REPORT_area_cost.md`. What to run next:
`04_NEXT_STEPS.md`.

---

## 0. One-paragraph summary

Goal-conditioned visual world-model planners (the released JEPA-WM and DINO-WM checkpoints of
`facebookresearch/jepa-wms`) score candidate futures by a latent distance to a **goal image**. With exact
simulator dynamics (no world-model error) we showed that this score, not the dynamics, is the main source of
failure on manipulation. We first tried a training-free area-invariant aggregation of patch mismatches
(top-k); it improved offline rankings but **failed in closed loop** and was closed. We then established a
sharper failure, a **task-equivalence violation**: two goal images that specify the *same task outcome* but
differ in something irrelevant (the robot arm's pose) produce different costs, rankings, chosen plans and,
on Metaworld drawer-close, different **closed-loop success** (100% → 73%, p = 0.031). Which goal factors are
irrelevant depends on the task (Push-T: pusher pose matters; drawer: arm pose does not; button-press: the
arm must stay on the spring-loaded button, so it matters), so no fixed mask can be right. No method yet.

---

## 1. Infrastructure built (all verified)

| Component | File | Verified |
|---|---|---|
| Env rebuilt for RTX 5090 (torch 2.7.0+cu128) | `env/setup_env_5090.sh`, `requirements-5090.txt` | GPU matmul OK |
| Push-T / Wall exact fast rollouts | `code/jepa-wms/evals/failure_attribution/sim_worker.py` | `tests/test_fast_rollout.py`: 0.0 state, 0 pixel |
| Metaworld exact snapshot / restore + fast rollouts | `evals/failure_attribution/mw_sim.py` | `tests/test_fast_rollout_mw.py`: 0.0 state, 0 pixel (reach-wall, push) |
| Re-planned episodes on 5090 vs saved 3090 Ti | — | bit-identical |
| Oracle attribution, single-plan evals | `evals/failure_attribution/run.py`, `summarize.py` | |
| Oracle attribution, replanning evals (Metaworld) | `evals/failure_attribution/run_closed.py`, `summarize_closed.py` | |
| Area-invariant costs (top-k, power mean) | `evals/failure_attribution/costs.py` | `l2` reproduces released cost bit-exactly |
| Task-equivalent goal variants (same object state, arm moved) | `evals/failure_attribution/goal_variants.py` | variant 0 == eval goal (0 pixel); object drift ≤ 4 mm |
| Robot-free rendering (robot = body subtree of `base` + `mocap`) | `mw_sim.robot_geoms`, `render_hidden` | 0-pixel change when hidden arm moves (no shadows) |
| Offline equivalence levels 1-3 | `evals/failure_attribution/goal_equiv_mw.py` | |
| Push-T equivalence control | `evals/pusht_goal_decomp/goal_equiv.py` | |
| Encoder determinism | — | 300 batched encodings of one frame identical (fp32) |

Oracle attribution: every episode is played three times from the same start and goal with the same CEM
budget: `wm` (world model + latent cost, the released eval), `sim_latent` (exact simulator + latent cost),
`sim_true` (exact simulator + true-state cost). Failure → dynamics if `sim_latent` succeeds, cost if only
`sim_true` succeeds, search if neither.

---

## 2. Results

### 2.1 Where planners fail (96 released episodes per row; Metaworld/Cube: simulator dynamics only)

| Task, model | `wm` | `sim_latent` | `sim_true` | Failures: dynamics / cost / search |
|---|---|---|---|---|
| Push-T, JEPA-WM | 64.6% | 64.6% | 90.6% | 15% / 59% / 26% |
| Push-T, DINO-WM | 65.6% | 61.5% | 90.6% | 18% / 55% / 27% |
| Wall, JEPA-WM | 75.0% | 83.3% | 92.7% | 45.8% / 41.7% / 12.5% |
| Wall, DINO-WM | 63.5% | 75.0% | 92.7% | 46% / 37% / 17% |
| Metaworld push, JEPA-WM enc. (24) | — | 2/24 | 20/24 | — |
| Metaworld drawer-close, JEPA-WM enc. (22-24) | — | 22/22 | 22/22 | — |
| Cube (OGBench, frozen DINOv2, 50) | — | 40% | 54% | — |

On manipulation the cost is the main failure source; on navigation (Wall) dynamics matter as much.

Maintenance correction (2026-10-12): attribution categories now form a disjoint partition.
Wall JEPA-WM episode 12 succeeds with `sim_latent` but fails with `sim_true`; it is counted only
under dynamics. Counts are 11 dynamics, 10 cost and 3 search, totaling 24 failures.
The earlier summary counted that episode twice. All success rates are unchanged.

### 2.2 Push-T: the cost under-weights the pusher (offline, 64 goal grids, 2.16 M trade-off pairs)

| Cost | pusher weight per px vs block (task: 1.00) | wrong on trade-off pairs |
|---|---|---|
| released, JEPA-WM / DINO-WM | 0.33 / 0.37 | 22.1% / 21.0% |

Closed loop (sim dynamics, 96 ep., Push-T JEPA-WM): released α=0.1 64.6%; α=1 70.8%; **α=10 79.2%**
(+17/−3, p=0.003); α=100 68.8% (pusher perfect, block 70.8%).

### 2.3 The top-k idea and why it was closed (details: `03_REPORT_area_cost.md`)

Cost = mean of the K largest per-patch mismatches instead of the mean of all 256 (training-free).

| | Offline Push-T ranking vs official | Closed loop |
|---|---|---|
| released | 0.735 | Push-T 64.6%, MW push 2/24, Cube 40% |
| top-8 | 0.818 (wrong trade-offs 22% → 14%) | Push-T 72.9% (n.s. vs released, below α=10), MW push 2/24, Cube 38% |
| top-8 + α=10 | — | Push-T 77.1% (vs α=10 79.2%, n.s.), MW push 1/24, Cube 44% |

Closed because: better offline ranking did not become better plans; on Push-T the small entity is the robot
(proprio α fixes it); on push/Cube the failure is not about area.

### 2.4 Task-equivalence violation (the current finding)

**Protocol.** For each Metaworld episode, the eval's goal (variant 0) plus goals with *identical object state*
and the arm moved elsewhere (variants 1-4; `goal_variants.py`). All costs are the released latent cost with
the released encoder; dynamics = exact simulator.
Levels: L1 cost change of the same candidate futures; L2 pairwise ranking flips (true criterion: 0 by
construction); L3 which candidate is selected (top-1 / top-10); L4 closed-loop success.

**Push-T control (pusher pose IS part of success):** when the pusher's goal pose changes by 10/40/80 px,
the latent ranking flips 0.031/0.101/0.179 (JEPA-WM; DINO-WM 0.036/0.109/0.192) vs the official criterion's
0.051/0.143/0.255. The latent cost responds in the right direction at ~70% of the needed strength. ✔

**Metaworld push (small puck), 24 episodes:**

| L1 rel. cost change | L2 flips | L2 flips vs eval goal | L3 top-1 | L3 top-10 | Spearman latent vs true progress (variants 0-4) |
|---|---|---|---|---|---|
| 82% | **43%** | 48% | **6%** | 14% | 0.11, −0.12, −0.05, 0.00, 0.02 |

First planned action differs between equivalent goals by 7.97 vs a CEM noise floor of 5.38.

Closed loop on push (all 24 ep., sim dynamics): released 2/24; top-8 2/24; α=10 0/24; **robot-free oracle
(goal and candidates rendered without the robot) 0/24 at α=0.1 and 0/23 at α=0**; true-state 20/24.
Puck moved > 2 cm in 3/24 episodes with any image cost vs 21/24 with the true cost.
Diagnostics: the latent cost IS monotonic in puck distance from the goal up to 30 cm (Spearman 0.91
robot-free), no shadow leak, deterministic encoding; but robot-free latent cost ranks actual pushes only weakly
(Spearman 0.04 for <0.5 mm moves, 0.33 for 0.5-5 mm, 0.48 for 5 mm-2 cm), and only 4% of CEM's random
candidates move the puck > 0.5 mm. **Conclusion:** push is floored by a weak small-object progress signal, a
second failure; L4 cannot be shown on push.

**Metaworld drawer-close (large articulated object), 24 episodes:**

| L1 | L2 flips | L2 vs eval | L3 top-1 | L3 top-10 | Spearman vs true progress (variants 0-4) |
|---|---|---|---|---|---|
| 51% | **36%** | 23% | **8%** | 23% | **0.65, 0.64, 0.36, 0.27, 0.34** |

Moving only the arm in the goal halves the cost's correlation with real drawer progress.

Closed loop drawer-close (sim dynamics, paired on the same 22 episodes):

| Goal | Success | vs eval goal |
|---|---|---|
| eval goal (variant 0) | **22/22** | — |
| variant 2 (arm moved 0.35 m) | 19/22 | 3 lost / 0 gained, p = 0.25 |
| **variant 3** (0.35 m) | **16/22** | **6 lost / 0 gained, p = 0.031** |
| variant 4 (0.37 m) | 22/22 | 0 / 0 |
| variants 2+3 pooled | 35/44 | 9 / 0, p ≈ 0.004 |
| true-state cost | 22/22 | |
| robot-free oracle, α=0 | 24/24, success reached at replan 1.0 (vs 2.5) | |

Caveat: final drawer errors sit near the threshold (median 5.2-5.4 cm vs 5.5 cm), so part of the effect is
near-miss failures.

**Other tasks.** door-open: equivalent goals could not be built (the door keeps moving after release).
**button-press-topdown: the button is spring-loaded and pops back when the arm leaves, so the arm pose is a
genuine part of that goal**, a natural counter-example to "always ignore the robot".

---

## 3. Literature position (all verified on arXiv / venue pages)

- Prior observation that the arm dominates image/latent goal costs: Know Thyself/RAC (ICLR 2022), LEXA
  (NeurIPS 2021), ROLL (CoRL 2020), Visual Foresight (2018). These are older and use pixel or task-trained
  models; none measures task equivalence, uses modern frozen-encoder world models, or shows the task-dependence
  quantitatively.
- Recent, most relevant to position against: "Supervise What Decides Success" (2610.01224), "When Low
  Prediction Error Misleads Planning" (2610.05550, render-matched goals change rankings on Reacher), CF-JEPA
  (2610.00727, controllable/uncontrollable split), DWM (2607.18715), The Planning Limits of Latent World Models
  (2609.39235), JEPA-Bisim (2602.18639), "Keeping JEPA WMs plannable when little of the frame moves"
  (2610.03137), GS-HER goal sets (2606.09476), TD-JEPA (2607.25337), SCALE (2608.16287), LEAP (2609.03294),
  D-JEPA (2609.24749), CGS (2609.35603).
- What appears new: task equivalence as a required invariance of goal costs; task-equivalent goal families as
  an evaluation protocol for frozen-encoder latent world models; exact-dynamics measurement through all four
  levels; the task-dependence (Push-T / drawer / button) showing no fixed mask works.

---

## 4. Honest status

- Established: the L1-L4 chain on drawer-close; L1-L3 on push; correct behaviour on the Push-T control.
- Weak spots: L4 on one task with one significant variant; push floored by a different failure; one model
  family (JEPA-WM encoder); L4 only with simulator dynamics; **no method yet**.
- To reach the CVPR bar: a method that infers per-task which goal factors matter and beats strong baselines
  (fixed robot mask, temporal distance, controllability split) on drawer AND keeps Push-T/button correct;
  replication on DINO-WM; the real world-model planner on at least one task.
