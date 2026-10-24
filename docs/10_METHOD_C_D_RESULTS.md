# Learned goal costs: results of Methods A, C, D and the state head (2026-10-21 to 10-23)

After the diagnosis (docs 01-07: the released latent goal cost is dominated by the robot arm), we tested whether a
small learned cost on the frozen world-model latents fixes planning. The world model, encoder and CEM search never
change; only the cost that ranks candidate futures does.

Summary files: `results/learned_goal_cost_2026-10-22/`. Run scripts: `scripts/pod/` (see `scripts/README.md`).

## Terms

| Term | Meaning |
|---|---|
| released cost | the planner's own cost: mean squared latent (patch) difference to the goal image |
| head | a small network f on frozen latents; the cost is ‖f(z) − f(g)‖² for candidate latent z and goal latent g |
| state head | trained so that ‖f(z) − f(g)‖ ≈ log(1 + task success distance); the distance comes from simulator states **at training time only** (test: images only) |
| time head | trained so that ‖f(z) − f(g)‖ ≈ log(1 + steps from z to g / 5) in the same episode; no labels (self-supervised) |
| evid head | trained on 64 success frames (pulled onto the goal) and 64 near misses (pushed away): weak task evidence |
| min (agg=min) | score a plan by its best step instead of its last step (with horizon 10 instead of 5) |
| fair control | the released cost with the same min aggregation and horizon (`released + min h10`) |
| sim_latent | closed loop with exact simulator dynamics and the given cost (Metaworld; removes world-model error) |

## Method A: action guidance (LeWM PushT): killed

The world model's next latent is extrapolated away from a "null action" prediction, z_null + w (z_a − z_null).
Every w ≠ 1 lowered success (offset 75: w = 1 → 20%, w = 1.5 → 10%, w = 2 → 8%, w = 0.5 → 2%), with the
training-free null and with a learned null embedding (`candA_finetune.py`). Not pursued.

## Method C on LeWM (stable-worldmodel), 50 goals per run, budget = 2 × goal offset

**TwoRoom** (navigation through a door):

| Cost | Δ50 (seed 0) | Δ75 (seeds 0 / 42) | Δ75 pooled |
|---|---|---|---|
| released, h5 | 52 | 30 / 32 | 31/100 |
| released + min, h10 (fair control) | 84 | 52 / 64 | 58/100 |
| state head, h5 | 96 | 90 / – | – |
| time head, h5 | 98 | 94 / 98 | 96/100 |
| **time head + min, h10** | – | **98 / 100** | **99/100** (p = 5e-14 vs control) |

**Cube** (OGBench, success = block within 4 cm):

| Cost | Δ50 (s0 / s42) | Δ75 (s0 / s42) | Δ50+75 pooled |
|---|---|---|---|
| released, h5 | 58 / 56 | 64 / 60 | 119/200 |
| released + min, h10 | 54 / 54 | 70 / 74 | 126/200 |
| **state head + min, h10** | **66 / 64** | **78 / 90** | **149/200** (+11.5 points, p = 0.017) |
| time head + min, h10 | 60 / – | 66 / 70 | – |

**PushT**:

| Cost | Δ50 (s0 / s42) | Δ75 (s0 / s42) | Δ50+75 pooled |
|---|---|---|---|
| released, h5 | 36 / 50 | 20 / 20 | 63/200 |
| released + min, h10 | 48 / 62 | 40 / 34 | 92/200 |
| state head + min, h10 | 52 / 68 | 36 / 34 | 95/200 (p = 0.84) |

Last-step scoring, h10, Δ75, seeds 0/42/1: state head 43/150 vs released 23/150 (p = 0.008). With min aggregation
the gap disappears: on PushT the head and min aggregation are substitutes.

**Offline ranking does not predict planning.** The heads rank held-out pairs far better than the released cost
(Spearman, state: PushT 0.995 vs 0.85, Cube 0.987 vs 0.834; time: PushT 0.966 vs 0.887), but the PushT time head plans
worse than the released cost.

## Method C on Metaworld (JEPA-WM, frozen DINOv2 ViT-S, exact dynamics, 12 episodes)

One time head on the released 8-task expert dataset (Spearman with the step gap 0.94, released cost 0.68).
drawer-open: **0/12**, same as the released cost. Both bring the hand to the goal-hand position (0.153 → 0.056 m)
and leave the drawer where it was (0.190 m from target). The other 8 tasks did not run (pod disk full).

## Method D pilot (Metaworld, 2026-10-22)

Question: can a few task examples (success + near-miss images) replace privileged labels? All heads use the same
architecture and data. Training episodes 100-123, test = the closed-loop episodes 0-11. Candidate sets start from the
start pose and from 12 and 4 steps before the expert first moves the object.

**Offline** (ordering accuracy on clearly different candidate pairs; eval goal / arm-moved goal):

| task | released | Method C time | state | time (per task) | evid | evid_time |
|---|---|---|---|---|---|---|
| drawer-open | 0.958 / – | 0.924 / – | 0.957 / – | 0.930 / – | 0.959 / – | 0.929 / – |
| plate-slide | 0.800 / 0.669 | 0.774 / **0.436** | **0.999 / 0.999** | 0.891 / 0.888 | 0.658 / 0.644 | 0.839 / 0.816 |
| drawer-close | 0.850 / 0.657 | 0.838 / **0.404** | 0.956 / 0.955 | 0.834 / 0.824 | 0.746 / 0.746 | 0.776 / 0.773 |
| reach-wall | 0.853 / 0.459 | 0.569 / 0.398 | 0.990 / 0.977 | 0.772 / 0.710 | **0.024** / 0.104 | 0.819 / 0.798 |

Decoy test on drawer-close (prefers a real success over an arm-at-goal failure): released 0.62, time 0.44,
state / evid / evid_time 1.00.

**Closed loop** (12 episodes):

| task | released | true-state cost | state head | evid | evid_time |
|---|---|---|---|---|---|
| drawer-open | 0/12 | 10/12 | 0/12 | 0/12 | 0/12 |
| plate-slide | 0/12 | 10/12 | **8/12** | 0/12 | 0/12 |
| drawer-close (no-harm control) | 22/22 | – | – | – | 11/12 |
| reach-wall (arm-is-task control) | 23/24 | – | – | – | **0/12** |

## State head on more Metaworld tasks (2026-10-23)

Offline, the state head ranks better than the released cost on every task where candidates change the true cost:

| task | released (eval / arm-moved) | state head | top-1 success: released / state / best available |
|---|---|---|---|
| push | 0.889 / 0.553 | 0.930 / 0.918 | 0.22 / **0.38** / 0.38 |
| button-press-topdown | 0.992 / 0.854 | 1.000 / 0.932 | 1.00 / 1.00 / 1.00 |
| window-open | 0.992 / 0.476 | 1.000 / 1.000 | 0.83 / 0.83 / 1.00 |
| faucet-open | 0.911 / 0.603 | 0.984 / 0.942 | 0.61 / 0.65 / 0.65 |
| door-open | no candidate changes the true cost: not measurable | | |

**Closed loop** (12 episodes, exact dynamics, same episodes for both costs; pods E and G):

| task | released cost | state head | true-state cost (diagnosis) |
|---|---|---|---|
| push | 0/12 | 0/12 | – |
| button-press-topdown | 0/12 | 0/12 | – |
| door-open | 0/12 | 0/12 | – |
| window-open | 0/12 | 0/12 | – |
| faucet-open | 0/12 | 0/12 | 4/12 |
| drawer-open | 0/12 | 0/12 | 10/12 |
| plate-slide | 0/12 | **8/12** | 10/12 |
| drawer-close (control) | 22/22 | 12/12 | 22/22 |
| reach-wall (arm-is-task control) | 23/24 | **7/12** | – |

**Verdict against the pre-stated rule** (beat the released cost by ≥ 3/12 on ≥ 4 of 8 object tasks, both
controls ≥ 10/12): **FAIL.** The state head wins on 1 of 8 object tasks (plate-slide) and lowers reach-wall
from 23/24 to 7/12.

- The failure is not ranking: offline the head ranks near-contact candidates almost perfectly. On push it picks the best available candidate in every set (top-1 success 0.38 of 0.38).
- From the start pose, no plan within the horizon touches the object. A dense learned cost pulls the arm to a definite place, and that place is not the object.
- The true-state cost is flat until the object moves, so CEM explores at random. That plausibly explains why it succeeds on drawer-open (10/12) where every learned cost fails. This is a hypothesis to test.
- On reach-wall the head (trained on 24 episodes) is less precise about the hand than the released patch cost, which already solves the task.

## Reach-then-change (2026-10-23): Stage 1 oracle and the start of Stage 2

A hand-written Metaworld reward says "bring the hand to the object, then the object to its target". The reach-then-change
cost imposes that order on a goal cost: candidates that have moved the object are ranked by a change term; the others
rank below them, by the hand-to-object distance (the reach term, saturating within 2 cm); if start and goal differ only
in the robot, the released cost is used (fallback). Code: `costs.ReachThenChangeOracle` (Stage 1, simulator hand and
object positions) and `costs.ReachThenChangeImage` (Stage 2, images only). Pre-stated rule: ≥ 6/12 on ≥ 2 of 4 object
tasks with the fallback ≥ 10/12.

**Stage 1** (12 episodes, exact dynamics, same episodes; reach term from simulator positions, change term varies):

| task | released | released + reach | **robot-free latent cost + reach** (no labels) | true object cost alone | true + reach |
|---|---|---|---|---|---|
| button-press-topdown | 0/12 | 0/12 | **12/12** | 0/12 | 12/12 |
| push | 0/12 | 3/12 | **7/12** | 11/12 | 12/12 |
| drawer-open | 0/12 | 1/12 | **6/12** | 10/12 | 6/12 |
| door-open | 0/12 | 0/12 | 0/12 | invalid: the env reports obj_to_target = 0 (fixed in `adapters.py`, not rerun) | invalid |
| reach-wall (fallback) | 23/24 | 12/12 | – | – | – |

- The bar is met with the robot-free change term (3 of 4 tasks). The released cost is 0/12 on every one of these tasks.
- Button: the flat true cost never finds the button (0/12); with the reach term it is 12/12, success by replan 3 in every episode. This is the approach-phase failure, fixed.
- Released + reach fails after contact: once the object is nudged the released cost drags the arm toward the goal arm pose. The change term must be object-focused.
- Drawer-open: the reach term costs 4 episodes against the pure object cost (the hand parks 3-5 cm from the handle); a cost of the structure, to fix by releasing the reach term sooner.
- Per-episode strings and the smoke history: `results/learned_goal_cost_2026-10-22/mw_reach_oracle/README.md`.

**Stage 1 on LeWM PushT (structure test, 50 goals, offset 75, exact dynamics; `oracle_cost.py` modes, pod K 2026-10-24).**
Pre-stated rule: the reach variant must beat its base by 10 points, or released + min (40%). Result: a **fail**, and a
scoping result. Released + reach 36% (block 36%); true block cost 48% (block 54%); true + reach 50% (block 58%). The
pusher starts next to the T-block in PushT and the released cost has no approach-phase failure there, so a reach term has
nothing to fix; the structure helps where the arm must first travel to the object (Metaworld), which is where the
diagnosis located the failure. PushT stays as the "where it does not apply" control.

**Stage 2 (images only), design arrived at on pods J and K (2026-10-23/24) through one-episode smokes.** Each step below
was a failure of the previous design, found in a smoke episode, and fixed before any 12-episode run:

| # | design | what the smoke showed | change |
|---|---|---|---|
| 1 | robot map (per-patch logistic head on DINOv2 features, render-mask labels: held-out AUC 0.998, IoU 0.94); change map = non-robot patches differing start vs goal; reach = nearest robot patch to the change centroid on the 16x16 grid | hand drifts away: the robot base is near the objects in 2D, so the reach is saturated from the start | reach from the arm's *new* footprint (robot in the candidate, not at the start) |
| 2 | new-footprint reach | hand approaches slowly (0.39 → 0.21 m); "moved" fires on every candidate (10% of a tiny start-goal change < render variation) | batch-relative "moved" (3x the batch median) |
| 3 | 2D reach | still slow; 2D cannot separate forearm from hand nor see depth | calibrated camera (corner2: validated, projected hand on a robot patch in 100% of frames, lift(project(X)) = X); object = change centroid lifted to the goal hand height; reach = 3D hand-object distance |
| 4 | 3D reach | button: the 18 change patches are arm shadows 5 patches from the button; lifted estimate 2.4 m off | keep only change patches whose lifted point lies in the robot's workspace; a detection needs >= 3 of them; otherwise the object is where the goal hand presses (+5 cm) and the change region is the patch disc there |
| 5 | workspace filter | push: only 1 patch near the goal hand (the puck travels 27 cm; the hand must go where it *starts*) | the change cluster far from the goal hand is the object at its start |
| 6 | start cluster | push: hand at the puck (estimate 5 cm short, lifted at hand height), nudges it; the change term cannot say "puck to target": its goal region is under the hand in the goal image and masked as robot | include the goal's contact disc in the change term → the arm copies the goal hand instead (push: hand 0.17 → 0.28 m away): reverted |
| 7 | progress gate (10% closer to the goal on the change region) | fires for 30-80% of candidates before contact: arm shadows move with the arm (the oracle's hidden robot cast none) | dropped |
| 8 | **robot-free-mask change term** (`reachrf`): latent distance over patches that are robot in none of start, goal, candidate; **geometric gate**: hand within 8 cm of the object estimate | button: reached in 3 replans, pressed 4.6 cm (success needs about 7); push: reached in 2 replans, pushed 7.6 cm toward the target; drawer-open: estimate 6.6 cm beside the handle (the drawer's side panel is part of the change), hand blocked by the cabinet | 12 episodes (8 cm stand-off): button 0/12, push 0/12 |
| 9 | stand-off 4 cm (the hand must end within one planning step of the object) | **button 3/12**: the 3 successes are exactly the episodes where the change detector found < 3 workspace patches and the goal-hand fallback was used (estimate 1 cm off); the 9 failures all had 3-7 "change" patches (arm shadows) and a target 0.24-0.53 m off. **push 0/12**: estimate 5-18 cm off (lifted at hand height, centroid midway along the puck's path); the puck moved 3-13 cm toward the target in every episode | first image-only successes on a Metaworld object task; the detector, not the cost, is the failure |
| 10 | v2 (`reachrf2`): a detected cluster is trusted only within 0.35 m of the goal hand (6 of the 9 false button targets were farther); closed-loop verification: hand at the target for two replans with no change toward the goal → fall back to the goal hand | smoke 0/3 on button, including the two episodes v1 solved: trajectories identical until the last replans, then the latent term on the contact disc fails to separate "pressed" from "touching" | geometric second phase for the fallback case (v3) |
| 11 | **v3**: in the fallback (nothing but the robot's contact region changes), the second phase is the distance to the goal hand itself; 4 cm stand-off | smoke 1/3 on button: ep 0 pressed (0.008); ep 1 stopped 6-7 cm from the button, never inside the 4 cm stand-off of a point 5 cm above the goal hand, so the second phase never engaged (to do: 6 cm for the fallback case); ep 2 false detection, verification did not fire | 12 episodes on 8 tasks (03:50 UTC) |
| 12b | v3, 12 episodes on push | **0/12**. The far cluster fires in 11 episodes, but the estimate is 5-12 cm off in 8 of them and 14-29 cm in 3, always displaced toward the start arm (shadow patches in the centroid) and toward the camera (the puck sits 4-7 cm below the goal hand, and lifting at hand height slides the point along the ray). The hand comes within 4-7 cm of the puck in 7 episodes and nudges it 1-5 cm; one episode pushed it 13 cm; the verification fallback fired 4 times, each time to the wrong place for a moving object | v3b: start-arm exclusion and strength filter clean the centroid; the far cluster is lifted at goal-hand height minus a 3 cm hover offset |
| 12c | v3, 12 episodes on drawer-open and plate-slide | **drawer-open 1/12**: the far cluster is the whole drawer front (8-18 patches), so its centroid is 6-14 cm from the handle in every episode; the hand parks there and the drawer moves in 3 episodes only; the one success and the 3.7 cm near-miss came *after* the verification fallback sent the hand to the goal hand (the handle's open position), which dragged the drawer. **plate-slide 0/12**: every estimate 9-14 cm toward the camera: the puck sits at 1.5 cm, the goal hand at 8 cm, and 1 cm of lifting height is 1.5 cm of horizontal error with this camera; the hand nudged the puck 4-8 cm in 5 episodes. The object at its goal is never visible as a separate cluster (0 near-goal-hand change patches in all push and plate-slide episodes: it is under the hand), so no start-to-goal displacement can be measured | v3b lifts the far cluster at table height when the goal hand is below 10 cm (table objects are pushed from above) and at goal-hand height otherwise (raised handles are grasped) |
| 12d | v3, 12 episodes on faucet-open and window-open | **faucet-open 0/12**: 8 episodes detect a far cluster at (0.1-0.28, 0.60-0.68, 0.19), 20 cm from the handle, and the hand parks there; the other 4 take the goal-hand fallback and reach the goal hand to 1.1-1.4 cm without turning the handle (the final pose alone does not make the sweeping motion; the handle never moves). **window-open 0/12**: every estimate lands at z = 0.08 (lifted at goal-hand height on a ray that meets the table first), 5-13 cm below and short of the handle; the hand ends 2-16 cm from the goal hand, the window moved in 1 episode (15 cm, 5 cm short) | localisation again; v3b's rules do not address faucet's far cluster (it passes the workspace and radius gates) |
| 12e | v3, 12 episodes on the controls | **drawer-close 10/12, reach-wall 8/12** (released controls on the same episodes still running; the drawer-close loss is 2 episodes, reach-wall 4) | the image cost must not cost more than 2/12 on the controls |
| 12 | v3, 12 episodes on button | **3/12** (same count as v1, different mechanism). 6 episodes used the goal-hand fallback and all 6 brought the hand within 0.8-1.6 cm of the goal hand; 3 of them pressed the button, the other 3 hit the button housing (1-2 cm lateral error: the second phase stops at its 1 cm convergence, nothing pushes further). The other 6 episodes are all the same false detection: 5-7 "change" patches at the robot's **start** position (lifted to (0.07-0.15, 0.55-0.60, 0.24) in every one of them) pass the 0.35 m radius gate and the far-from-goal-hand rule, so the cost treats the start arm's shadow / vacated region as the object; the hand parks there; the verification fired once, at replan 5 | v3b (to run after the v3 table completes, as `reachrf3`): exclude change patches inside the dilated start-robot mask; keep a far cluster only if its change strength is at least 0.7 x the robot's own appearance change (measured: start-arm shadows 0.2-0.6 x, real objects up to 1.0-1.8 x, table below); in the fallback case the second phase is lexicographic again (within 6 cm of the goal hand, rank by the robot-free latent term) |

Change-patch strength relative to the robot's own appearance change (patch feature difference start vs goal, divided by the mean
difference over the goal's robot patches; non-robot patches in the workspace; 12 test episodes per task, `podk_change_mag.py`):

| task | object travel | workspace change patches / episode | median | p10 | p90 | max |
|---|---|---|---|---|---|---|
| button-press-topdown | 0.09 m | 3.8 | 0.24 | 0.16 | 0.58 | 1.03 |
| push | 0.24 m | 5.8 | 0.56 | 0.23 | 1.04 | 1.17 |
| drawer-open | 0.19 m | 13.2 | 0.35 | 0.19 | 0.96 | 1.28 |
| plate-slide | 0.28 m | 7.8 | 0.88 | 0.33 | 1.55 | 1.81 |
| faucet-open | 0.25 m | 3.6 | 0.47 | 0.29 | 0.89 | 1.03 |
| window-open | 0.20 m | 11.8 | 0.37 | 0.22 | 0.73 | 1.40 |
| drawer-close | 0.15 m | 15.6 | 0.20 | 0.10 | 0.86 | 1.18 |
| reach-wall (no object) | 0.00 m | 4.8 | 0.36 | 0.25 | 0.59 | 0.79 |

reach-wall has no object, so its 4.8 patches per episode (median 0.36, max 0.79) are the shadow baseline; the object tasks' upper
tails (0.9-1.8) are the objects. A relative threshold near 0.7 keeps the large objects (push, plate-slide, drawer, window) and
drops the small ones (button, faucet) into the goal-hand fallback, which is the right place for them.

**Bookkeeping error, 2026-10-24 (recorded for honesty):** the 4 cm stand-off was patched on the pod but not in the repo copy; two
later uploads of the repo copy (adding the TEMPO objective and v2) silently restored 8 cm on the pod. Image-cost runs that
started after 02:05 UTC (drawer-open, faucet-open in the S2 queue; the v2 smoke) therefore used 8 cm and are discarded; the
button 3/12 and push 0/12 above started before and are valid. v3 reruns every task.

The full Stage 2 cost (`costs.ReachThenChangeImage`, `--cost reachrf:<robot head>`): robot map → change patches → lifted to the goal
hand height → workspace filter → object-at-start cluster (or goal-hand fallback) → 3D reach to it until the hand is within 8 cm →
then the robot-free-mask latent distance to the goal. No task labels; the robot map is trained once from render masks (no task
labels either); the camera is the calibrated render camera.

## Approach-aware learned head (rtc), 2026-10-24: one head for 8 tasks

Target: hand-to-object distance until the object moves, then the task's success distance (simulator labels at training time
only); the state head (plain success distance) trained on the same data is the ablation; 24 training episodes per task,
test on the 12 closed-loop episodes. Offline (ordering accuracy on candidate sets where the true cost varies; eval goal /
arm-moved goal; top-1 = the selected candidate is a success, "best" = any candidate is):

| task | released | rtc | state | top-1 released / rtc / best |
|---|---|---|---|---|
| button-press-topdown | 0.992 / 0.857 | 1.000 / 0.995 | 1.000 / 1.000 | 1.00 / 1.00 / 1.00 |
| push | 0.888 / 0.548 | 0.930 / 0.924 | 0.917 / 0.908 | 0.22 / 0.38 / 0.38 |
| drawer-open | 0.958 / – | 0.924 / – | 0.961 / – | 0.92 / 0.92 / 0.92 |
| plate-slide | 0.795 / 0.661 | 0.997 / 0.997 | 0.999 / 0.999 | 0.06 / 0.08 / 0.14 |
| faucet-open | 0.918 / 0.609 | 0.998 / 0.996 | 0.918 / 0.884 | 0.61 / 0.65 / 0.65 |
| window-open | 0.992 / 0.465 | 1.000 / 0.996 | 1.000 / 1.000 | 0.83 / 0.83 / 1.00 |
| drawer-close | 0.851 / 0.657 | 0.824 / 0.832 | 0.947 / 0.942 | 0.21 / 0.75 / 0.83 |
| reach-wall | 0.853 / 0.459 | **0.119 / 0.047** (inverted) | 1.000 / 0.999 | 0 / 0 / 0 |

Both heads rank near-perfectly on the object tasks and are invariant to the goal's arm pose. The rtc head is inverted on
reach-wall: with no object, its "object" is a dummy that never moves, so the approach term pointed at the dummy. Fixed
(object-free tasks keep the plain target; `rtc2`, retraining).

**Closed loop, 12 episodes, exact dynamics (pod K, 2026-10-24): button 0/12, push 3/12.** The pre-stated rule
(≥ 6/12 on ≥ 4 of 7 object tasks) is already out of reach for these two tasks; the remaining six are running for the record.
The failure is not ranking but **exploitation**: the planner finds states the head never saw and prices them low.

- Button: in all 12 episodes the hand ends 20-45 cm from the button while the head's cost falls (e.g. 1.21 → 0.10); the hand
  drifts to the right of the button (x ≈ +0.27 instead of 0.09) or stays low (z ≈ 0.05). Training candidates start on the
  expert path and last one planning horizon (30 steps), so arm poses more than ~0.3 m off the path have no labels.
- Push: in 8 of 12 episodes the hand reaches the goal hand position to 1-2 cm, but the puck lags 6-10 cm behind in 5 of
  them, at cost 0.004-0.03: "hand at goal, puck short" never occurs in expert data, so the head prices it as success.
- Offline tables cannot see this: they rank candidates drawn from the same distribution as the training data.

Remaining first-version tasks (for the record): drawer-open 0/12, plate-slide 1/12, faucet-open 0/12, window-open 2/12,
drawer-close 12/12 (control intact).

**Fix 1 (H3 queue, `mw_pilot_data --walks`, `mw_pilot_heads --n_ens`): (1) arm-coverage walks**, kind 3 frames from random
action sequences of the planner's own prior, continued for a whole episode (70 steps) from the start and from 4 steps
before contact, 16 per start, a frame every 10 steps (40,320 walk frames, 136,752 in all); (2) an **ensemble of 3 heads**,
planner cost mean + β·std (β = 1; β = 0 as the ablation). Label-light protocol unchanged. Offline ranking unchanged
(0.83-1.00, reach-wall fixed).

**Closed loop (ensemble, β = 1), 12 episodes: button 1/12, push 3/12, drawer-open 1/12, plate-slide 5/12. Exploitation
is gone; the final precision is not.**

| | rtc v1 (no walks, 1 head) | ensemble + walks (β = 1) |
|---|---|---|
| button: episodes whose hand ends within 5 cm of the goal hand | 0 / 12 (20-45 cm) | **12 / 12** (0.9-5.4 cm) |
| button: episodes that move the button | 0 / 12 | 9 / 12 (0.4-8.6 cm of travel; 2 fully) |
| push: episodes whose hand ends within 5 cm of the goal hand | 8 / 12 | **12 / 12** (0.3-3.1 cm) |
| push: puck moved | – | 12 / 12 (15-34 cm); success at some replan 9 / 12, at the end 3 / 12 |
| drawer-open: hand within 5 cm of the goal hand | success 0 / 12 | **10 / 12** (1.1-4.8 cm); 2 park 21 cm away without touching the handle |
| drawer-open: drawer pulled | – | 10 / 12 (13-16 cm); it stops 3.3-6.7 cm short of the 3 cm threshold in 9 of them: success 1 / 12 |
| plate-slide: hand within 5 cm of the goal hand | success 1 / 12 | 9 / 12 (0.3-2.7 cm), 3 more at 7-8 cm |
| plate-slide: puck moved | – | 12 / 12 (10-29 cm); success at some replan 8 / 12, at the end 5 / 12 |

The arm now goes where it should in every episode, so the coverage + pessimism fix removed the exploitation. What is left is
the released cost's own near-success failure, now inside the learned head: with the hand at the goal hand, the head cannot
tell the object's state. Button ep 006 ends 1.1 cm from the goal hand with the button unpressed at cost 0.031, below the
0.047 of ep 002's pressed final state; push ends with the hand at the goal hand and the puck 4-10 cm off at cost
0.005-0.08. The head's proprio input and the arm-weighted patch features make it arm-dominated where the target changes
fastest. "Hand at the goal, object not there" exists in the walks but is rare next to the 95% of frames where the hand
position predicts the target well.

**Fix 2 (HR queue, `costs.HeadChangeObjective`, `--cost headrf:<head>:<robot mask>:β:λ`): add the Stage 1 change term.**
cost = head(z) + λ · rf(z) / rf(z₀), where rf is the squared feature difference to the goal over the patches that show the
robot in neither the start, the goal nor the candidate frame (the image robot mask), divided by the same quantity at the
episode start. The head dominates the approach (0.3-8 against λ ≈ 0.5), the change term decides where the head is flat
(< 0.1 near the goal hand; the term runs from ~1 to ~0.3). No detection, no localisation (the parts of the image pipeline
that failed), no switch. Offline on held-out episodes the premise holds: near the goal the change term is 0.07-0.12 for
pressed states and 0.36-0.56 for unpressed ones (head 0.05 against 1.4-4.8 on the same, in-distribution, candidates).

**Result (12 episodes): button 0/12, push 5/11, drawer-open 3/12, plate-slide 6/12** (ensemble alone: 1, 3, 1, 5; the 12th
push episode hit the 2 h limit with five loops sharing the GPU). Where the object stays visible the change term does what it
should: push ends within the 5 cm threshold in 5 of 11 with the hand at the goal hand in every episode, drawer-open goes
from 1 to 3, plate-slide from 5 to 6 (the first task at the 6/12 bar). On button it changes nothing, for a reason the
Stage 1 oracle could not show. In all 12 episodes the hand reaches the goal pose (0.6-3.9 cm) with the **gripper open
(0.43-0.80, against 0.30 at the goal)**: the fingers straddle the button and the hand descends to the goal height without
pressing it (button top 6 cm above the hand). Hidden under the arm, the button is in no robot-free patch, so the change
term has nothing to see it with: final cost 0.036 for an unpressed state and 0.036 for a nearly pressed one. The oracle's
robot-free renders saw through the arm; the image mask cannot. The head has the gripper opening as an input but never saw
an open gripper at the goal pose over an unpressed button: that configuration is neither on the expert path nor among the
walk frames (walks start 4 steps before contact and are sampled every 10 steps).

**Fix 3 (H4 queue, `mw_pilot_data --jitter`): near-goal coverage.** Kind 4 frames from the expert's own states 10, 6, 4
and 2 steps before contact and along the post-contact path: 10 short action sequences per start, each with a constant
open-or-close gripper command (2 chunks hovering, then 3 chunks of xyz drift, a frame after each), so that gripper, height
and object state vary at the goal pose and the exploited configurations get labels. The 3-member ensemble is retrained on
expert + candidates (1 in 3) + walks + jitter; closed loops head + rf and head alone on button, push, drawer-open,
plate-slide, then head + rf on the other four. Rule unchanged.

Six jitter designs were checked on the first training episode before any training (per-step noise from post-contact states;
one-phase random drift; two-phase hold-then-drift; aimed at the goal hand from pre-contact states, 1 cm lateral spread, longer
descent; aimed from the episode start at full speed). None reproduced the button exploit (0-8 of ~200 near-goal frames with
the button unpressed): a top-down descent presses the plunger even with the fingers fully open, and the straight line from
the episode start is blocked by the button box 14 cm short. **The planner itself arrives at the goal height laterally**, so
the plunger ends between the open fingers. Hand-designed perturbations do not find the planner's blind spots; the planner
does.

**Fix 3b (H5 → H6 queues, `run_closed --ep_start 100 --dump_dir`, `mw_merge_planner_frames.py`): the planner's own frames.**
The deployed cost (ensemble head + rf) is run on the 12 *training* episodes of button and push (exact dynamics); at every
replan the executed frame (kind 5) and 32 of the final CEM population (kind 6, the states the cost actually prices) are saved
with their simulator labels and merged with the expert and goal frames of the same pilot episodes. On the first push
training episode 181 of 181 near-goal candidates are "hand at the goal hand, puck off"; on the first button episode (which
the planner solved) 5 of 133. The next ensemble (H6) trains on expert + candidates (1 in 3) + walks + aimed jitter + planner
frames; closed loops head + rf and head alone on button and push, head + rf on the other six. This is one round of
iterated cost learning on the planner's visited states, with privileged labels at training time only, as before.

**Result (H6, 12 episodes, exact dynamics, 2026-10-24, all 8 tasks done 16:57 pod time). Head + rf: button 3/12, push
7/12, drawer-open 10/12, plate-slide 11/12, faucet-open 2/12, window-open 7/12; controls drawer-close 12/12, reach-wall
11/12. Head alone, same ensemble, no change term: button 2/12, push 10/12.** Previous round (head + rf without planner
frames): 0, 5/11, 3, 6. Every task that supplied planner frames went up; faucet-open and window-open had jitter frames but
no planner frames (dumps were run on four tasks) and went 0 → 2 and 2 → 7. **Four object tasks at or above the 6/12 bar
(push, drawer-open, plate-slide, window-open) with both controls at or above 10/12: the pre-stated rule PASSES**, the
first cost in this project to do so. TEMPO (re-implemented temporal distance, w = 0.5), same episodes: 0/12 on all six
object tasks, drawer-close 12/12, reach-wall 5/12 (it breaks a control the released cost solves 23/24). Pod K was backed up
(`artifacts/lewm_explorations_raw/reach_image_2026-10-24`, head at `H6/heads/head_rtc.pt`, every closed-loop episode
file) and stopped by the watcher at 16:57. Training set 148,518 frames (expert 28.8k, goal 2.4k, candidates 25.2k at 1 in 3, walks 40.3k, jitter 40.7k
over 6 tasks, planner frames 384 executed + 10,752 candidates from 4 tasks × 12 training episodes); 23 min; offline ranking
unchanged (0.86-1.00).

- Button: the three successes all end with the gripper closed (0.32-0.34 against 0.30 at the goal); the nine failures end
  with it open (0.34-0.82) at final cost 0.17-1.0, against 0.006-0.13 for the successes. The cost now prices the exploited
  configuration above the pressed one, which it did not before (0.036 for both); the search still ends there in nine
  episodes, four of them 3.2-3.6 cm from the target. A second planner-frame round on the new cost is the obvious next step.
- Push: head alone 10/12 against head + rf 7/12. With the planner's "hand at goal, puck off" frames in the data the head
  resolves the terminal state by itself, and the change term (λ = 0.5, fixed before the run) now costs episodes; the
  difference is within noise at n = 12 but makes λ the first ablation to run.
- Drawer-open 3 → 10 and plate-slide 6 → 11 with the same change term as before: the planner frames gave the head labels
  for "handle pulled 13-16 cm, 3-7 cm short", which the walks did not contain.

## What the results say

1. **The learned cost helps where the released distance contradicts task progress in a way the data shows.**
   - TwoRoom: the wall makes straight-line latent distance wrong, and time-to-goal learns the path through the door (+41 points over the fair control).
   - Cube: success needs only the block, and the state head learns a block-only distance (+11.5 points).
2. **A frozen encoder carries enough information to plan an object task, when the cost is supervised correctly.**
   On plate-slide the state head goes from 0/12 to 8/12 (the true-state cost gets 10/12).
3. **Time-to-goal inherits the arm bias on manipulation.**
   - The Method C time head ranks arm-moved goals below chance (0.40-0.44).
   - In closed loop it moves the arm, not the drawer.
4. **Few examples define "done" but not "how to get there".**
   - evid fixes the arm decoy (0.62 → 1.00).
   - But as a dense ranking it is weak (plate-slide 0.66) or inverted (reach-wall 0.02).
   - evid_time breaks reach-wall in closed loop (0/12).
5. **A second failure, distinct from the arm bias, decides most Metaworld tasks: reaching contact.** Confirmed on 9 tasks: the correctly supervised state head fails on 7 of 8 object tasks.
   - On drawer-open and button-press-topdown, both costs rank candidates near the object almost perfectly (0.96-0.99, top-1 success 92-100%).
   - Yet every cost fails from the start pose (0/12). From the start pose no candidate within the horizon touches the object.
   - So the cost far from the object, and the search, decide these tasks, not the ranking near success.
6. **Min-over-plan is a cheap general fix** (+27 TwoRoom, +14.5 PushT, +3.5 Cube with the released cost). It is a baseline, not a contribution.
7. **A learned cost that ranks perfectly offline is exploited in closed loop when its data does not cover the planner's search** (rtc head: offline 0.93-1.00, closed loop button 0/12, push 3/12). Arm-coverage walks + an ensemble with pessimism remove the exploitation: the hand reaches the goal hand in 24/24 button and push episodes (from 8/24). Success stays at 1/12 and 3/12 because the head, like the released cost, cannot resolve the object's state once the arm is there: the approach failure is solved by learning, the near-success failure needs the robot-free change term (under test as head + λ·rf).
8. **The reach-then-change structure is specific to the approach failure.** On PushT, where the pusher starts at the block, it changes nothing (36% vs 40%; 50% vs 48% with the true cost). On Metaworld, where the arm must travel, it takes button from 0/12 to 12/12 with the oracle terms.

## Related work checked (verified 2026-10-22)

| Paper | Overlap |
|---|---|
| TEMPO, "How Long, Not How Close" (arXiv 2610.04988, 2026-10-04) | temporal-distance map on frozen LeWM / PLDM / DINO-WM latents; TwoRoom 36% → 99%. Our time head is the same idea |
| NGC, "Latent World Models Should Plan with Distances Along the Latent Manifold" (2610.25013) | graph shortest-path terminal cost, no training; TwoRoom 95%, Reacher 100% |
| "Supervise What Decides Success" (2610.01224) | success-criterion quantities as an auxiliary loss when training the world model; head discarded; PushT +3.5, Cube +3.4 |
| GS-HER, "Goal Sets, Not Goal States" (2606.09476) | query-defined goal sets for offline GCRL |
| TD-JEPA (2607.25337), TRM (2605.22164), "The Objective Is the Bottleneck" (2608.12959), Traj-LeWM (2608.14125) | learned temporal / reachability costs on LeWM, TwoRoom |
| VICE (Fu et al. 2018), RCE (Eysenbach et al. 2021) | success classifier from examples + Bellman progress: the core of Method D |

## Infrastructure lessons

- **Disk:** check free disk after every upload / install step. A full disk made every later run fail instantly and still write its "done" marker (pod D).
- **Rendering:** run `scripts/pod/check_egl.sh` on a new pod. A host without the NVIDIA EGL driver renders on the CPU, at about 30 min per episode instead of 2 (pod F).
- **Backup:** back up to the home folder first. The project drive was unmounted once when a queue finished, so the backup failed and the pod stayed up.
- **Downloads:** fetch shared downloads (DINOv2 torch hub, LeWM checkpoints) once before parallel runs; simultaneous downloads race.
