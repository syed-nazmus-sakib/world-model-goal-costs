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

## State head on more Metaworld tasks (2026-10-23, closed loop in progress)

Offline, the state head ranks better than the released cost on every task where candidates change the true cost:

| task | released (eval / arm-moved) | state head | top-1 success: released / state / best available |
|---|---|---|---|
| push | 0.889 / 0.553 | 0.930 / 0.918 | 0.22 / **0.38** / 0.38 |
| button-press-topdown | 0.992 / 0.854 | 1.000 / 0.932 | 1.00 / 1.00 / 1.00 |
| window-open | 0.992 / 0.476 | 1.000 / 1.000 | 0.83 / 0.83 / 1.00 |
| faucet-open | 0.911 / 0.603 | 0.984 / 0.942 | 0.61 / 0.65 / 0.65 |
| door-open | no candidate changes the true cost: not measurable | | |

Closed loop so far (12 episodes, exact dynamics): button-press-topdown state 0/12 vs released 0/12; door-open
state 0/12; push released 0/12; window-open released 0/12. Remaining runs: push / window-open / faucet-open /
reach-wall / drawer-close with the state head, door-open released. Pre-stated rule: the head "works" if it beats
the released cost by ≥ 3/12 on ≥ 4 of the 8 object tasks, with reach-wall and drawer-close ≥ 10/12.

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
5. **A second failure, distinct from the arm bias: reaching contact.**
   - On drawer-open and button-press-topdown, both costs rank candidates near the object almost perfectly (0.96-0.99, top-1 success 92-100%).
   - Yet every cost fails from the start pose (0/12). From the start pose no candidate within the horizon touches the object.
   - So the cost far from the object, and the search, decide these tasks, not the ranking near success.
6. **Min-over-plan is a cheap general fix** (+27 TwoRoom, +14.5 PushT, +3.5 Cube with the released cost). It is a baseline, not a contribution.

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
