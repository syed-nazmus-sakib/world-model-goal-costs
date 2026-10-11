# Assessment: which failure cases are real, and which can carry a method

Written 2026-10-11, at the end of the work on this machine. Every number below was re-derived from the saved
per-episode files (`results/`), not copied from earlier notes. Full chronological record: `docs/02_REPORT_progress.md`.

## How to read this

A failure case is useful for a method paper only if it is:

1. **real**: reproducible, not an artefact of our code (exact simulators, deterministic encoding, paired
   comparisons on identical episodes);
2. **causally isolated**: we know which component fails (dynamics, cost or search), and the explanation was
   tested, not assumed;
3. **large and general enough**: a clear effect on more than one task or model;
4. **open**: a strong method does not already fix it trivially (e.g. a hyperparameter), and recent work does
   not cover it.

Verdict scale: **Strong** (all four), **Promising** (real and isolated, but narrow or small so far),
**Real but not a method target**, **Not established**.

---

## Summary table

| # | Failure case | Real | Isolated | Size / breadth | Open | Verdict |
|---|---|---|---|---|---|---|
| F1 | The goal **cost**, not the world model, is the main bottleneck on manipulation | yes | yes (oracle attribution) | large; 4 tasks, 2 models | partly (same message as recent analysis papers) | **Strong as a finding**, motivation rather than contribution |
| F2 | **Task-equivalence violation**: equivalent goal images (same object state, different arm pose) change cost, rankings, plans and success | yes | yes (L1-L4 chain, exact dynamics) | rankings: large on 2 tasks; success: moderate, 1 task | yes | **Promising**: best foundation for a method |
| F3 | Task-dependence of goal relevance (Push-T pusher matters, drawer arm does not, button arm must stay) | yes | yes | 3 tasks, qualitative contrast | yes | **Promising**: the argument that no fixed rule works |
| F4 | Weak **small-object progress signal** (Metaworld push, OGBench Cube) | yes | partly (cause narrowed, not pinned) | very large on push (8% vs 83%) | yes | **Real, method-worthy but hard**, cause not yet isolated |
| F5 | Push-T cost **under-weights the pusher** (0.33-0.37 px/px vs 1.0) | yes | yes | Push-T only | **no**: raising α fixes it (64.6% → 79.2%) | **Real but not a method target** |
| F6 | **Goal-appearance shift** collapse (LeWM / PushT, background colour 90% → 7%) | yes | yes (latent offset analysis) | 1 task, 1 model so far | partly (robustness literature) | **Promising**, separate direction |
| F7 | Expert data → poor **counterfactual** prediction (LeWM; decorrelated data 3.9-20× lower error) | yes | yes | 2 tasks | yes | **Real**; planning gain only on PushT (+31 pt), none on Cube |
| F8 | Area dominance of patch-averaged costs (top-k fix) | offline only | — | — | — | **Not established** in closed loop; direction closed |
| F9 | Far-goal degradation of LeWM (goal 25 → 50 steps ahead: 88% → 41.5%) | yes | — | — | **no**: well covered by 2026 literature | not pursued |

---

## F1. The cost is the bottleneck (strong, but not the contribution)

With exact simulator dynamics replacing the world model (`sim_latent`), success does not improve on Push-T,
while the true-state cost (`sim_true`) succeeds far more often:

| Task, model | world model | exact dynamics + latent cost | exact dynamics + true cost |
|---|---|---|---|
| Push-T, JEPA-WM | 64.6% | 64.6% | 90.6% |
| Push-T, DINO-WM | 65.6% | 61.5% | 90.6% |
| Wall, JEPA-WM | 75.0% | 83.3% | 92.7% |
| Wall, DINO-WM | 63.5% | 75.0% | 92.7% |
| Metaworld push (24) | — | 8.3% | 83.3% |
| Cube (50, frozen DINOv2) | — | 40% | 54% |

On Push-T, 55-59% of world-model failures are attributed to the cost. On Wall, dynamics account for 46%,
so the bottleneck depends on the task. **Assessment:** solid and well measured, but recent analysis papers
make the same high-level point (e.g. The Planning Limits of Latent World Models, 2609.39235). Use it as
motivation and as the evaluation protocol, not as the headline.

## F2. Task-equivalence violation (the most promising foundation)

**What.** Goals built to be task-equivalent: same object state (drawer joint within 4 mm, puck pose
identical), only the robot arm moved elsewhere. Verified: variant 0 is pixel-identical to the eval goal;
object drift ≤ 4 mm (success margins 5-5.5 cm).

| Level | Metaworld push (24 ep.) | Metaworld drawer-close (24 ep.) |
|---|---|---|
| L1 cost change for the same future | 82% | 51% |
| L2 ranking flips (true criterion: 0%) | **43%** | **36%** |
| L3 same best candidate | **6%** | **8%** |
| Correlation of cost with true object progress, eval goal vs arm-moved goals | ~0 for all | **0.65 vs 0.27-0.36** |
| L4 closed-loop success, eval goal → arm-moved goals | floored (0-8% for all) | **22/22 → 19/22, 16/22 (p = 0.031), 22/22** (paired) |

**Why it is real.** Exact simulator dynamics (no model error), the same candidate futures scored against
each goal (no search noise), deterministic encoding (verified), paired episodes for L4. With the robot
removed from all images the drawer is solved faster (success at replan 1.0 vs 2.5).

**Why it is not yet strong.** L4 is shown on one task, with one significant variant (p = 0.031) and partly
near-miss failures (final error ≈ threshold); one model family (JEPA-WM encoder); simulator dynamics only;
on push L4 is masked by F4.

**What would make it strong.** DINO-WM replication; more arm variants and episodes on drawer; the real
world-model planner; one more object task with success between 30% and 80% (exact commands:
`docs/04_NEXT_STEPS.md`, E1-E4).

## F3. Relevance depends on the task (the argument a method needs)

- **Push-T (control):** the pusher is part of success; the latent cost responds to pusher-goal changes in
  the right direction at ~70% of the needed strength (ranking flips 0.10 vs 0.14 official at 40 px).
- **Drawer-close / push:** the arm is irrelevant to success, yet it changes rankings and (drawer) success.
- **Button-press-topdown:** the button is spring-loaded and pops up when the arm leaves; the arm pose is a
  genuine part of the goal.

**Assessment:** this is what makes F2 more than "mask the robot". Any fixed rule (always ignore the robot,
always keep it, ignore controllable parts) is wrong on one side. It is currently qualitative; the next step is
the baseline table in `docs/04_NEXT_STEPS.md` E5.

## F4. Weak small-object progress signal (real, large, unresolved)

On Metaworld push with exact dynamics: latent cost 2/24 vs true cost 20/24; the puck moves > 2 cm in 3/24
episodes vs 21/24. Ruled out as causes: the arm (robot-free rendering: 0/24, 0/23 without proprio), area
weighting (top-k: 2/24), resolution or saturation (the cost is monotonic in puck distance up to 30 cm),
rendering artefacts (no shadow leak), numerical noise (deterministic encoding). What remains: the cost ranks
the small, partly random pushes that CEM's samples produce only weakly (Spearman 0.03-0.48 depending on push
size), and only ~4% of candidates move the puck at all, so CEM has no consistent gradient to climb. The same
symptom appears on Cube (perfect-model latent ranks cube progress at ~0.24).
**Assessment:** large and method-worthy (a planner/cost that can follow small object progress), but the
mechanism is not pinned down, and it overlaps with long-horizon/shaping work. Riskier as a foundation than F2.

## F5. Pusher under-weighting on Push-T (real, not a method target)

The released cost values 1 px of pusher error at 0.33 (JEPA-WM) / 0.37 (DINO-WM) of block error; it is wrong
on 21-22% of trade-off pairs. But raising the proprio weight α from 0.1 to 10 already lifts success from 64.6%
to 79.2% (17 gained / 3 lost, p = 0.003), and our top-k cost did not beat that (72.9%; 77.1% combined with
α = 10). Report as a finding about released configs; do not build a method on it.

## F6. Goal-appearance shift (promising, separate direction)

From the earlier LeWM work (`code/lewm_explorations/`, raw numbers in `results/lewm_explorations/goalshift/`,
4 seeds x 50 goals): the goal image shows the correct target state but is rendered differently.

| Goal rendering | Success (full PushT criterion) | Block-only success |
|---|---|---|
| identical (control) | 90.0% (84-92) | 96.5% |
| background colour changed | **7.0%** (2-10) | 58.0% |
| block colour changed | 11.5% | 62.0% |
| agent colour changed | 27.0% | 85.0% |
| agent removed from goal | 25.5% | 80.5% |

The latent offset is spread isotropically, so no post-hoc linear projection removes it (killed, idea 2).
**Assessment:** clean, large and visual, but robustness to observation distractors is an active area and it
needs its own (trained) method. It is complementary to F2: in both, the goal image carries things that should
not matter. Note that block-only success degrades much less, so part of the drop is the pusher placement.
A ready benchmark kit exists (`code/lewm_explorations/ogbench_shift/`).

## F7. Data for world models (real, partial)

With matched action marginals and a fixed budget, decorrelating actions from states reduces counterfactual
error 3.9× (PushT) to 20× (Cube) and lifts PushT planning from 41.3% to 72.3%, but gives no planning gain on
Cube (planning there is limited by search). Same-state branching does not help. Solid analysis; not pursued.

## F8, F9. Closed or not pursued

Top-k / power-mean aggregation: better offline rankings (0.735 → 0.818), no closed-loop gain where it would
matter (`docs/03_REPORT_area_cost.md`). LeWM horizon collapse: real but covered by recent work.

---

## Bottom line

- **Yes, there are genuine failure cases.** The strongest foundation for a method is **F2 + F3**: visual
  goal costs are not invariant to task-irrelevant parts of the goal image, and which parts are irrelevant
  depends on the task. It is causally isolated with an exact-dynamics, four-level protocol and has a
  significant closed-loop effect on drawer-close.
- **It is not yet paper-strength on its own.** The closed-loop effect is moderate and shown on one task and
  one model. Before building the method: replicate on DINO-WM, add arm variants and episodes, find one more
  object task without floor/ceiling effects, and run the fixed-rule baselines (`docs/04_NEXT_STEPS.md`, E1-E5).
- **Method direction:** a goal representation that infers per task which factors matter, e.g. a
  goal-equivalence contrastive projection trained on simulator-generated equivalent / non-equivalent goal pairs
  (M1 in `docs/04_NEXT_STEPS.md`). Success criterion: fix drawer-close variants without breaking the Push-T control or
  the button case, and beat the fixed robot mask, temporal-distance and controllability baselines.
- **Second option:** F6 (goal-appearance shift) has the largest single effect (90% → 7%) and a ready benchmark
  kit, but needs a separate method.
- **F4** is the largest gap overall (8% vs 83%) but the least understood.
