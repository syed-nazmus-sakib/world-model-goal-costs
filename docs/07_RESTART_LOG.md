# Restart log (from 2026-10-15)

Experiments restarted after the literature review (`06_LITERATURE_REVIEW_v2.md`). Target: CVPR 2027. Running on a RunPod pod
(RTX PRO 4000 Blackwell 24 GB, about 10 CPU cores, container disk 50 GB, about $0.57/hr). The workload is CPU-bound (MuJoCo rendering and
simulator workers): about 142 s per drawer-close episode solo, 106 s effective with two jobs in parallel; more jobs only slow each other.

## Pod notes
- SSH: RunPod proxy accepts the Mac key only with a PTY. The key was then installed inside the container and `sshd` started, which enables direct
  `ssh root@<ip> -p <port>` and rsync. sshd and the container disk do not survive a pod reset: rsync results back after each batch.
- Environment: miniconda at `/root/miniconda3`, env `jepa-wms` (torch 2.7.0+cu128, sm_120 works). Project copy at `/root/jepa-wms-project`,
  `macros.py` and configs rewritten to that root. `tests.test_fast_rollout_mw` passes (0.0 state, 0 pixel).
- Launch long jobs with `setsid nohup ... < /dev/null &` from a script file. A job started from an SSH session that then closes can lose its
  simulator workers (KeyboardInterrupt in `scipy`/`cv2` import) and the SimPool hangs. Never `pkill -f <name in the command line>`; kill by PID.

## Findings so far

### 1. The proprio confound in the equivalent-goal variants (code check)
`run_closed.py` builds the variant goal with `goal_td(g_frame, g_obs)` and `goal_td` uses `obs[:4]` as the goal proprio, i.e. the variant's hand
pose. So at the released alpha = 0.1 the planner was also told to reach the variant arm pose through the proprio term. Control: rerun with
`--alpha 0`.

### 2. Drawer-close closed loop, exact dynamics, JEPA-WM encoder, alpha = 0, 24 episodes
| Goal | alpha = 0.1 (earlier, 24 ep.) | alpha = 0 |
|---|---|---|
| eval goal (variant 0) | 22/22 | 24/24 |
| variant 2 | 20/24 | 22/24 (2 lost, 0 gained, p = 0.5) |
| variant 3 | 18/24 | **17/24** (7 lost, 0 gained, exact p = 0.0156; Holm over two variants about 0.031) |

Variant 3's effect survives removing the proprio term. Variant 2 is not significant. Runs extended to 48 episodes for v0 and v3 (in progress).
Note the effect depends on the specific arm pose: variant 4 (similar distance) was 24/24 at alpha = 0.1.

### 3. Offline ranking quality, drawer-close (saved candidates, alpha = 0.1, 24 episodes; script on the Mac, not yet in the repo)
Ordering accuracy against the true cost on clearly different candidate pairs (true-cost gap above its median):

| Goal given | accuracy | per-episode drop vs eval goal | episodes worse than eval goal |
|---|---|---|---|
| eval goal | 0.894 | | |
| variant 1 | 0.890 | -0.003 | 83% |
| variant 2 | 0.712 | -0.182 | 71% |
| variant 3 | 0.662 | -0.231 | 79% |
| variant 4 | 0.709 | -0.184 | 96% |

Push gave NaN (true cost constant on most candidate sets), so this metric does not apply there.

### 4. Oracle goal set, offline (same data)
Goal set = the five equivalent arm-pose goals (each cost z-scored per goal):

| Aggregation | ordering accuracy | Spearman with true cost |
|---|---|---|
| single goal v0 / v3 | 0.894 / 0.662 | 0.646 / 0.265 |
| mean over the set | **0.892** | 0.644 |
| soft-min, T = 0.5 | 0.880 | |
| min over the set | 0.813 | 0.509 |
| mean over variants 1-4 only | 0.861 | 0.598 |

Averaging cancels the arm-pose-specific distortion; taking the min is lenient and loses accuracy. Mean of squared distances over a set equals the
squared distance to the mean latent plus a constant, so the averaged goal costs nothing extra in planning. This is an oracle (simulator-made)
set: it is only an upper bound for a world-model-built set.

## Code changes (additive, in `code/jepa-wms/evals/failure_attribution/`)
- `goal_equiv_mw.py`: `--alpha`, `--no_cem`, `--wm_orbit K --wm_scales --wm_h` (world-model goal set), ordering-accuracy metrics in the summary.
- `run_closed.py`: `--orbit 0,1,2,3,4` (mean encoding of the listed arm-pose goal variants as the goal latent; oracle goal-set closed loop).

## Queue / in progress
- v0 and v3 at 48 episodes, alpha = 0 (JEPA-WM, drawer-close).
- Oracle goal-set closed loop: `--orbit 2,3,4` then `0,1,2,3,4`, alpha = 0, 24 episodes.
- Offline L1-L3 at alpha = 0 on JEPA-WM and DINO-WM; DINO-WM closed loop v0 and v3 at alpha = 0.
- World-model goal-set offline test (`--wm_orbit 16`), then a closed-loop version if it holds.

## Decision rules (stated before the results)
- Oracle goal set must bring variant 3 (and 2) close to the eval-goal success (>= 22/24) at alpha = 0, or the mean-over-a-set mechanism is dropped.
- A world-model-built set must keep ordering accuracy within 0.03 of the eval goal on drawer-close, otherwise it is not a method.
- Any method must also keep Push-T and the spring button correct and beat arm-mask / Know Thyself / CF-JEPA-style baselines; otherwise
  the paper falls back to the diagnosis and protocol and is reported as below the bar.

## Update 2026-10-16 (pod clock Oct 15 19:45 UTC)

### Oracle goal set, closed loop, drawer-close, alpha = 0, 24 episodes, exact dynamics
| Goal given to the planner | success |
|---|---|
| eval goal | 24/24 (48/48 over 48 episodes) |
| variant 3 alone | 17/24 (36/48 over 48 episodes) |
| mean encoding of arm-moved goals {2,3,4} | **22/24** |
| mean encoding of {0,1,2,3,4} | 24/24 (contains the eval goal itself, so partly trivial) |
Averaging three arm-moved goals recovers most of the loss (17 to 22 of 24) but not all of it. The honest oracle upper bound is the {2,3,4} row.

### DINO-WM "replication" is NOT independent evidence
With exact simulator dynamics (`sim_latent`) the predictor is never used, so JEPA-WM and DINO-WM differ only in the encoder input normalisation and the proprio
encoder, on the same frozen DINOv2. Offline ordering accuracy confirms it: eval goal / v1 / v2 / v3 / v4 = 0.893 / 0.890 / 0.712 / 0.664 / 0.711 (JEPA-WM config) vs
0.891 / 0.887 / 0.706 / 0.657 / 0.708 (DINO-WM config). Closed loop at alpha 0: DINO-WM config eval goal 24/24, variant 3 19/24. Treat it as a consistency check of the
cost, not as a second model family. A different encoder (e.g. DINOv3, SigLIP 2, V-JEPA 2, LeWM) or the real world-model planner is needed for model-level generality.

### Offline L2/L3 at alpha = 0 (JEPA-WM config, 24 episodes)
Flip rate between equivalent goals 0.359 (vs eval goal 0.237), top-1 candidate agreement 0.09, top-10 overlap 0.23, Spearman with true cost 0.646 (eval goal) vs 0.268 (variant 3).
Same pattern as at alpha = 0.1, so it is not the proprio term.

## Update 2026-10-16 evening (pod clock about 21:30 UTC)

### Fixed arm mask vs released cost on reach-wall (arm is the goal), closed loop, exact dynamics, 24 episodes
| Cost | success |
|---|---|
| released (alpha 0.1) | **23/24 (95.8%)** |
| oracle robot mask (`--robot_free`, goal and candidates rendered without the arm), alpha 0 | **0/24** |
A mask that hides the arm cannot see the goal when the arm is the goal. Caveat: RAC-style methods keep a separate agent term (here the proprio weight), so the fair
baseline is mask + proprio term. Queued: `reach_robotfree_a01` (mask, alpha 0.1) and `drawer_v3_robotfree_a01`. The point to show is that no single global weight
works on both tasks: alpha 0 breaks reach, alpha 0.1 reintroduces the arm-pose pull on the drawer variants.

### World-model goal set, offline ordering accuracy on drawer-close (24 episodes, alpha 0; `goal_equiv_mw.py --wm_orbit K --wm_mode dir --wm_h H`)
Columns: eval goal, variant 1, 2, 3, 4.
| Goal | v0 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| single goal | 0.893 | 0.890 | 0.712 | 0.664 | 0.711 |
| iid-noise motions, H3 (first attempt) | 0.896 | 0.892 | 0.731 | 0.689 | 0.724 |
| sustained direction, K16 H8, mag 1.0 | 0.899 | 0.898 | 0.817 | 0.735 | 0.817 |
| sustained direction, **K32 H16, mag 0.7** | 0.902 | 0.900 | **0.882** | **0.882** | **0.882** |
| sustained direction, K32 H16, mag 1.0 | 0.901 | 0.899 | 0.894 | 0.859 | 0.900 |
| oracle mean over the five simulator-made goals | 0.902 | | | | |
The training-free, simulator-free goal set (mean of the goal latent and 32 world-model rollouts of 16 planner steps in one random sustained direction each) restores
ordering accuracy on arm-moved goals to within 0.011 of the eval goal and does not hurt the eval goal. Short or iid-noise motions do not work (the arm barely moves).
Pre-stated bar (within 0.03 of the eval goal for variant 3): met at mag 0.7 (0.882 vs 0.893).

### The reversibility gate does not work
Reverse-cycle error as a detector of "the nuisance motion moved the object by more than 1 cm" (labels from the simulator): AUC 0.61 at H8 and **0.47 at H16** (chance), Spearman about 0.
At H16 K32, 23% of the rollouts did move the drawer, yet the ungated set is still accurate. Drop this gate; the open question is whether the ungated set damages tasks where the arm is part of the
goal (reach-wall, button): queued closed-loop `reach_wmset` and `button_wmset` (K32 H16 mag 0.7). If it does, a different mechanism for task-dependence is needed.

### Queue changes
Pod A: after the button mask runs, `queue_A4.sh` runs reach_wmset, reach_robotfree_a01, button_wmset, drawer_v3_robotfree_a01 (two lanes). Pod B: closed loop with the world-model set on
drawer-close variant 3 (H8 K16 mag 1.0 running; H16 K32 mag 1.0 running), then H16 K32 mag 0.7 for variants 3, 0 (no-harm) and 2 via `queue_B3.sh`. A watcher on the Mac
(`artifacts/raw_logs/pod_runs/watch.log`) backs up and stops each pod when its queue flag (`A_QUEUE_DONE`, `B_QUEUE_DONE`) is written and no job runs.

## Update 2026-10-16 about 01:15 UTC: go/no-go checkpoint
All runs: exact simulator dynamics, JEPA-WM encoder, 24 episodes, `sim_latent`. Drawer-close at alpha 0 unless stated.

| Condition | success |
|---|---|
| eval goal / variant 2 / variant 3, single goal | 24/24 (48/48) / 22/24 / 17/24 (36/48) |
| world-model goal set K16 H8 mag 1.0: v3 / v0 | 20/24 / 23/24 |
| goal set K32 H16 mag 1.0: v3 | 22/24 (5 gained, 0 lost vs single v3, p = 0.062) |
| goal set K32 H16 mag 0.7: v3 / v2 / v0 | 22/24 (6 gained, 1 lost, p = 0.125) / 21/24 / **21/24** |
| robot mask + alpha 0.1, drawer variant 3 (`drawer_v3_robotfree_a01`) | **24/24** |
| reach-wall: released / robot mask alpha 0 / mask + alpha 0.1 | 23/24 / 0/24 / 0/15 so far |
| reach-wall with the world-model goal set (K32 H16 mag 0.7) | **0/24** |
| button-press-topdown: released / robot mask (alpha 0) | 0/24 / 0/24 (floored, not usable) |

Arm-explainability ratio (min over arm-only rollouts from the start of the distance to the goal, over the start-goal distance): drawer eval goal 0.42, arm-moved goals 0.73-0.80;
reach eval goal 0.35. It does **not** separate "arm is the goal" from "arm irrelevant". Dead.

Verdict against the pre-stated rules: **NO-GO for the ungated goal set as the headline method.** It fixes the drawer variants offline and mostly in closed loop (+5 episodes on v3) but
(1) breaks reach-wall (0/24), (2) lowers the eval goal (24/24 to 21/24 at mag 0.7), (3) the reversibility gate is at chance (AUC 0.47), (4) a mask + proprio baseline already reaches 24/24 on drawer v3.
Open and decisive: a sweep of the proprio weight with the mask (`reach_robotfree_a1`, `_a10`, `drawer_v3_robotfree_a1`, `_a10`, queue_A5). If one global alpha works on both tasks, the
"no fixed rule works" premise fails on Metaworld and the contribution is the diagnosis/protocol, not a method. If none does, the task-dependent weight is the problem to solve
(candidate: set the agent weight from the object change between the start and goal in robot-free renders, which is about 0 on reach and large on the drawer).
Pod B was stopped by the watcher after backup (results in `artifacts/raw_logs/pod_runs/podB/`); pod A stays up until `A5_DONE` (a `HOLD` file blocks the watcher).

## Final sweep 2026-10-16 (about 04:45 UTC): robot mask + proprio weight, exact dynamics, 24 episodes each
| alpha | reach-wall (arm is the goal) | drawer-close variant 3 (arm irrelevant) |
|---|---|---|
| 0.1 | 0/24 | **24/24** |
| 1 | 0/24 | 18/24 |
| 10 | **22/24** | 19/24 |
Released cost (no mask): reach-wall 23/24, drawer v3 17/24 (alpha 0, single goal).
No single alpha works on both tasks: small alpha solves the drawer and fails reach, large alpha solves reach and loses 5 episodes on the drawer. The task-dependent
weight is confirmed as the real issue, but the fix it implies (per-task alpha) is a tuning knob on Know Thyself (2021), not a new method. Both pods are stopped
(watcher: backed up, EXITED). Balance after the restart: about $13.36 of the original $24. Decision (user, 2026-10-16): this finding is not strong enough to carry a
CVPR 2027 main-track paper; direction closed as a method, kept as a diagnosis.


## 2026-10-17: reduced Experiment A (generality) launched on pod B
Decision (user): the current finding is worth testing for generality before anything else; multi-seed later; cheapest pod; one job at a time.
Plan and pre-stated rules (`scripts/pod/queue_generality.sh`, `scripts/pod/pick_tasks.py`):
1. Task search on window-close, drawer-open, faucet-open, door-close, handle-press-side, plate-slide (12 episodes, exact dynamics, latent and true cost).
   Keep tasks with 3-10/12 latent success, true cost >= latent + 2, >= 3 arm variants per episode, object drift < 1 cm; up to 2 tasks.
2. Encoder generality, offline L1-L3 at alpha 0, 24 episodes: SigLIP 2 ViT-B/16, CLIP ViT-B/16, DINOv2 ViT-B/14, V-JEPA 2 ViT-L/16 (DINOv3 is gated; skipped unless weights are supplied),
   on drawer-close, reach-wall (control: no object, eval goal only) and the selected tasks. Rule: the arm-moved ordering-accuracy drop (eval goal minus variants 2-4) is "general" if it is
   >= 0.10 on at least 3 of 4 encoders on drawer-close, and present on at least one selected task.
3. Task generality, L4 at alpha 0, 24 episodes, eval goal vs variant 3 on the selected tasks. Rule: a paired drop of >= 4 episodes on at least one new task.
4. Real world-model planner on drawer-close (eval goal vs variant 3, 24 episodes) if one episode fits in 20 GB and takes < 10 min.
Estimated cost about $8-10 of the $13 balance; everything one lane, resumable.

### Results, 2026-10-17 (queue started 23:44 UTC on 10-16; Stage 1 took 2 h 34 min, not the 6-8 h estimated; Stage 2 runs take about 2 min each)
**Stage 1, task search** (12 episodes, exact simulator dynamics, same CEM budget and success rule for both costs; released cost alpha 0.1):
| task | latent cost | true-state cost | arm variants / episode | latent final object-to-target (threshold) | rule |
|---|---|---|---|---|---|
| window-close | 0/12 | 0/12 | 4 | 0.210 (0.05) | drop: even the true cost fails (search/horizon, not cost) |
| drawer-open | 0/12 | **10/12** | 1 | 0.190 (0.03) | drop: arm touches the handle at the goal, 1 variant only |
| faucet-open | 0/12 | 4/12 | 4 | 0.290 (0.07) | drop (latent at floor) |
| door-close | 0/12 | 0/12 | 4 | 0.486 (0.08) | drop: true cost fails too |
| handle-press-side | 0/12 | **12/12** | 1 | 0.060 (0.02) | drop: 1 variant only |
| plate-slide | 0/12 | **10/12** | 4 | 0.278 (0.07) | **fallback selection** (largest gap with 4 variants) |
On all six tasks the latent-cost planner left the object where it started (final object distance equals the start distance in every episode), while the
true-state cost with the same dynamics and search solved four of them. No task fell in the 3-10/12 band, so `pick_tasks.py` used its fallback: plate-slide.
This is a floor effect on every new task: the released cost is not "degraded" there, it does not work at all. The L4 paired test (Stage 3) therefore cannot show a
further drop on plate-slide and was left to run only because it was pre-stated (2 x 24 episodes, about 50 min each).

**Stage 2, encoder generality** (offline, exact dynamics, alpha 0, 200 candidates per episode, 24 episodes; ordering accuracy of the latent cost against the true
cost on clear pairs; v0 = eval goal, v1 = smallest arm move, v2-v4 = arm moved; drop = v0 minus mean(v2..v4)):
| encoder | drawer-close v0 / v1 / v2 / v3 / v4 | drop | reach-wall v0 (control) | plate-slide v0 / v1 / v2 / v3 / v4 |
|---|---|---|---|---|
| DINOv2 ViT-S/14 (released) | 0.893 / 0.890 / 0.712 / 0.664 / 0.711 | 0.20 | 0.838 | 0.447 / 0.548 / 0.405 / 0.447 / 0.329 |
| SigLIP 2 ViT-B/16 | 0.900 / 0.894 / 0.676 / 0.624 / 0.709 | 0.23 | 0.912 | **0.178** / 0.377 / 0.408 / 0.469 / 0.427 |
| CLIP ViT-B/16 | 0.896 / 0.893 / 0.721 / 0.697 / 0.758 | 0.17 | 0.662 | 0.603 / 0.698 / 0.547 / 0.629 / 0.596 |
| DINOv2 ViT-B/14 | 0.885 / 0.883 / 0.686 / 0.653 / 0.688 | 0.21 | 0.831 | 0.307 / 0.445 / 0.369 / 0.410 / 0.296 |
| V-JEPA 2 ViT-L/16 | 0.876 / 0.872 / 0.683 / 0.631 / 0.719 | 0.20 | 0.857 | 0.261 / 0.326 / 0.323 / 0.369 / 0.267 |
Sanity check that the swap is real: on episode 0 the true costs are bit-identical across runs (same candidates), the latent costs differ in scale and value, and the
per-candidate Spearman correlation between encoders is 0.90-0.98 while each encoder's correlation with the true cost is 0.62-0.67 (eval goal). The encoders agree
with each other far more than with the task: they all read the same arm pixels.
- Rule part 1 (drawer-close drop >= 0.10 on >= 3 of 4 encoders): **met, 4 of 4** (0.17-0.23; the released DINOv2-S gives 0.20).
- Rule part 2 (drop present on a selected task): **not met on plate-slide.** There is no arm-moved drop; instead the eval-goal accuracy is at or below chance for
  4 of 5 encoders (0.18-0.45; CLIP 0.60) and the arm-moved goals are no worse (SigLIP 2 rises from 0.18 to 0.38-0.47). Mechanism: the expert's final arm pose
  sits at the far end of the slide, so candidates that move the arm toward it without touching the puck get a low latent cost, and candidates that start pushing
  the puck (arm still near its start) get a high one. Moving the arm in the goal removes the misleading cue. Same cause (arm pixels dominate the patch cost),
  different symptom: a wrong cost at the eval goal itself, consistent with 0/12 closed loop against 10/12 for the true cost. Report it as that, not as the drop rule.
Tooling fixed on the way (details in `09_PODS_ACCESS.md`): the reach-wall config was missing on the pod (copy the pod's base config, not the Mac's: Mac configs hold
the lab checkpoint path); the V-JEPA 2 torch-hub `main` is broken (checkpoint URL points at localhost, blocks reject `H_patches`), fixed by pre-downloading `vitl.pt`
(5.1 GB) into the hub cache and a kwarg adapter in `alt_encoders.py`; `pgrep -f name` matched the ssh command that contained the name, so the reruns were launched
from a script on the pod. Spend so far on this queue about $1.

**Stage 3, task generality L4 on plate-slide (DINOv2, alpha 0, 24 episodes):** eval goal 0/24, arm-moved variant 3 0/24. Uninformative, as predicted from the
Stage 1 floor; the paired-drop rule cannot be evaluated on any of the six new tasks because the released cost never succeeds on them.

**Stage 4, real world-model planner (released JEPA-WM predictor + CEM, drawer-close, alpha 0):** smoke test 1/1 success, 884 s per episode, 17.6 of 20.5 GB VRAM,
no OOM. The pre-stated budget rule said < 10 min per episode; it is 14.7 min. Decision: run anyway (2 x 24 episodes = about 12 h, about $3.4 of the remaining
$12) because this is the only test that connects the diagnosis to the actual released planner. The queue's 16000 s timeout would have cut each run at ~18
episodes, so the queue was stopped 7 min into the eval-goal run and stage 4 relaunched as `/root/stage4.sh` with a 24000 s timeout (same commands, writes QUEUE_DONE).
Result, eval goal (released predictor, alpha 0, 24 episodes, finished 10:14 UTC): **18/24 = 75%** [55, 88]. The exact-dynamics planner gets 24/24 on the same goals, so
the released predictor alone costs 6 episodes. Arm-moved variant 3 running next (expected about 16:30 UTC); exact-dynamics reference for v3 is 17/24.
**Result, arm-moved variant 3 (finished 16:09 UTC): 6/24 = 25% [12, 44].** Paired against the eval goal on the same episodes: 14 lost, 2 gained,
exact McNemar p = 0.004. The released planner loses 12 episodes to the goal-arm change, against 7 with exact dynamics: the predictor's error compounds
the cost's error. Stage 4 total: 2 x 24 x 884 s = about 11.8 h, about $3.3.

## 2026-10-17 13:00 UTC: method candidate 1 queued (stage 5, `/root/stage5.sh`, runs after stage 4; writes `generality/ALL_DONE`)
Decision (user): design a method that mitigates the failure; start with the cheapest candidate. Three candidates were laid out, cheapest first:
1. **Change-conditioned nuisance projection** (test time, no training; implemented as `--proj K --proj_h --proj_mag` in `goal_equiv_mw.py`,
   math in `goalset.nuisance_projection`). K = 32 free-motion rollouts from the goal under the released predictor give U = z_k - g, the directions
   in which the arm can move at the goal. d = g - z_start is the start-to-goal change (what the task asks for). Three costs are scored on the same
   candidates: `dperp` projects out span(U - <U, d_hat> d_hat) (nuisance with the task direction removed; never discards progress along d),
   `full` projects out span(U) (the plain goal-set subspace; expected to break reach), `gated` interpolates plain and full by rho = ||proj_U d||^2 / ||d||^2.
2. Self-supervised per-patch agent-ness head (predict proprio from each patch; downweight robot patches unless the start-goal change lives in them). Needs training.
3. Controllability-factored JEPA (agent factor predictable from proprio and actions; environment factor constant across free-motion segments; per-factor cost gated
   by the start-goal change). Needs retraining on the A100s; the architecture-level version if 1 or 2 show the principle works.
Prior art to cite and distinguish: Denoised MDPs (Wang et al. 2022), Iso-Dream (Pan et al. 2022), Know Thyself (Hu et al. 2022); none targets goal-image planning
with a frozen encoder, none uses the start-goal change to pick the relevant factor, none is evaluated on task-equivalent goal variants.
Pre-stated pass criteria for candidate 1 (offline, exact dynamics, alpha 0, 24 episodes): drawer-close arm-moved ordering accuracy from about 0.67-0.71 to >= 0.85 while
the eval goal stays >= 0.85; reach-wall eval goal stays >= 0.80 (plain DINOv2: 0.838); plate-slide eval goal > 0.6 (plain 0.447). Any setting that must differ
between tasks fails by definition. Two rollout settings are run (H 16 mag 0.7; H 8 mag 1.0) on drawer-close, reach-wall, plate-slide: about 25 min, about $0.15.
Synthetic check of the math (CPU, 5000-d latent with a 20-d "arm" subspace and one "object" direction; Spearman of cost vs true progress): drawer-like plain 0.31,
dperp 0.76, full 1.00, gated 0.38; reach-like plain 0.91, dperp 0.94, full -0.02, gated 0.91. So `dperp` is the candidate; `full` is the control that should break
reach; `gated` is a fallback that depends on rho being informative.

## 2026-10-17 13:50 UTC: stage 4b queued BEFORE candidate 1 (user: "without generalisability of the problem we cannot say it is a valid problem")
The objection is fair: the paired arm-moved drop exists on one task (drawer-close) because the released cost is at the floor on every other object task.
The floor is evidence that the cost is broken, but not a second instance of the mechanism. Stage 4b (`/root/stage4b.sh`, runs after stage 4, then calls
stage 5) measures the mechanism itself on every task, three ways, with rules written now:
- **B. Cost share of the robot** (`cost_share_mw.py`, 9 tasks x 12 episodes): fraction of the start-to-goal patch cost carried by the patches the robot
  occupies (pixel mask from the robot-hidden render), and the fraction of the start-to-goal latent distance that disappears when the robot is hidden in
  both images. Rule: the robot carries >= 50% of the cost on at least 5 of the 7 object tasks (reach-wall should be near 100%: the arm is the task).
- **C. Offline arm-moved test on 6 more tasks** (window-close, door-close, faucet-open have 4 arm variants per episode; drawer-open, handle-press-side,
  button-press-topdown have only the eval goal and the base variant), 5 encoders, 24 episodes, exact dynamics, alpha 0. The offline levels do not need
  the baseline to succeed in closed loop, so the floor does not block them. Rule: arm-moved drop >= 0.10 for DINOv2-S on at least 2 of the 3 tasks with
  variants, and on at least 3 of the 4 other encoders there.
- **A. Oracle robot mask, closed loop** on the 4 floored tasks with true-cost headroom (drawer-open 10/12, handle-press-side 12/12, plate-slide 10/12,
  faucet-open 4/12): exact dynamics, robot hidden in goal and candidates, alpha 0 and alpha 0.1, 12 episodes. Rule: at one alpha, success rises from 0/12
  to >= 4/12 on at least 2 of the 4 tasks. If so, the failure is general across five object tasks and a method has a measured target on each (GO for
  the method on the A100s). If not, the problem is narrow and the direction stops before any training.
Estimated 5-6 h, about $1.5. Stage 5 (candidate 1) follows automatically; ALL_DONE is written at the very end.

Bug found and fixed 16:15 UTC in part B: `robot_free_frame()` restored the snapshot but did not call `mj_forward`, so the renderer drew the geoms where the
last simulation step had left them. In `cost_share_mw.py` the start and goal robot-hidden renders therefore came out identical (`cost_robot_free = 0.0`,
"share removed = 1.0" on every task: wrong). The earlier robot-mask results are not affected: there the robot-hidden goal was rendered right after the expert
rollout or after the variant generator, both of which leave the object geoms at the goal state, and the candidates' robot-hidden frames come from the simulator
workers after stepping. `mj_forward` is now called inside `robot_free_frame`; part B was re-run (`share2` lines in queue.log; the first results are in
`generality/share_v1/`, discarded). The four renders of drawer-close episode 0 were checked by eye (`share/drawer-close_ep0_montage.png`): correct.

**Part B result (16:24 UTC, 12 episodes per task, released DINOv2-S, start image vs eval goal image):**
| task | robot covers (patches) | robot patches carry (share of cost) | removed by hiding the robot in both images |
|---|---|---|---|
| reach-wall (arm is the task) | 49% | 77% | 84% |
| drawer-close | 43% | 72% | 65% |
| button-press-topdown | 47% | 77% | 79% |
| plate-slide | 47% | 76% | 69% |
| drawer-open | 43% | 74% | 69% |
| handle-press-side | 41% | 75% | 57% |
| faucet-open | 49% | 74% | 79% |
| window-close | 45% | 79% | 73% |
| door-close | 53% | 78% | 64% |
Rule (>= 50% on >= 5 of 7 object tasks): **met on 8 of 8.** On every object task about three quarters of the start-to-goal cost sits on the robot, and
hiding the robot removes 57-79% of the latent distance. The reach-wall numbers (77% / 84%) are not separable from the object tasks: the cost looks the same
whether the arm is the task or not, which is the mechanism in one table.

Part C so far: window-close gives no ordering accuracy (no candidate future from the start moves the window within the horizon; the true cost is constant),
faucet-open DINOv2-S: eval goal 0.686, arm-moved 0.585-0.628 (drop 0.08, below the 0.10 rule, and the eval goal itself is weak).
door-close DINOv2-S: only 4 of 24 episodes have candidates that move the door; on those the eval-goal accuracy is 0.35 (below chance, the plate-slide
symptom), arm-moved 0.40-0.53 (n = 3). The summariser crashed on door-close because the variant generator accepted a different number of arm poses in
different episodes; fixed (`_pad` in `goal_equiv_mw.py`, plus a new `episodes_per_variant_index` field) and re-summarised.
Lesson for the offline protocol on contact tasks: candidate futures must start from a state near contact (for example the expert's state a few steps
before the object first moves), otherwise the true cost is constant and the test says nothing. Not changed mid-run; to do for the next round.

**Stage 4b part C, all five encoders done (18:08 UTC).** Eval-goal ordering accuracy, DINOv2-S / SigLIP 2 / CLIP / DINOv2-B / V-JEPA 2:
faucet-open 0.69 / 0.74 / 0.82 / 0.72 / 0.57 (arm-moved variants lower by 0.05-0.15); drawer-open 0.48 / 0.65 / 0.45 / 0.44 / 0.47;
handle-press-side 0.42 / 0.43 / 0.41 / 0.40 / 0.42; door-close (n=4) 0.35 / 0.52 / 0.35 / 0.34 / 0.28; window-close and button-press-topdown
unreadable (no candidate moves the object). Four of five readable unseen tasks at or below chance for every encoder.

**Stage 4b part A, oracle robot mask, closed loop, first two results (alpha 0, 12 episodes):** drawer-open 0/12 (no change); **handle-press-side 0/12 -> 6/12**
[25, 75], success at any replan 67%. Caveat: stage 1 baselines are alpha 0.1, so an alpha-0 unmasked control was added (stage 4c, in `/root/stage5.sh` before
`ALL_DONE`): `run_closed --planners sim_latent --alpha 0 --episodes 12` on handle-press-side, drawer-open, faucet-open -> `generality/base/`. Plate-slide alpha 0
unmasked is already known (stage 3, 0/24). The alpha-0.1 masked runs compare directly to stage 1.

**Stage 4b part A, all alpha-0 mask runs done (19:25 UTC):** drawer-open 0/12, handle-press-side 6/12, plate-slide 1/12, faucet-open 0/12.
Rule (>= 4/12 on >= 2 of 4 tasks at one alpha): **not met at alpha 0.** alpha 0.1: drawer-open 0/12 (19:43 UTC); the other three run next, then
stage 5 (projection probe) and the alpha-0 unmasked controls (stage 4c). Reading for now: removing the robot is necessary (handle-press-side) but not
sufficient; on the other three the robot-free residual cost does not track the object either (hypothesis: small object, few patches, non-monotone latent
change). Direct test to queue next: offline ordering accuracy of the robot-free cost on the same tasks.

**Write-up rewritten (2026-10-18 01:50 local):** `writeup/main.tex` is now a diagnosis note (F1-F8 catalogue, equations for where the failure enters,
protocol limits, implications, future experiments). Method components removed until tested. Two new figures (`unseen_enc.pdf`, `mask_unseen.pdf`),
closed-loop figure and Table 1 updated with the final 6/24. Compiled with pdflatex, 8 pages, no overfull boxes.

## 2026-10-18 (about 20:30 UTC on 10-17): external audit (`research_notes/project_audit_2026-10-17/PROJECT_REVIEW.md`) checked and acted on
Verified true and acted on:
- **Projection rank bug** (`goalset.nuisance_projection`): `svd(U, full_matrices=False).Vh` returns min(K,P) rows regardless of rank, so for a
  rank-deficient U the extra rows are arbitrary null-space directions, which can contain d. Fixed: `_row_space()` keeps singular vectors with
  s > 1e-4 s_max, plus an assert that Q_dperp is orthogonal to d. Pushed to the pod before stage 5 started (backup `goalset.py.bak_prefix_20261017`).
  Audit counterexample now gives cost 1/3 (was 0); generic K=32 case keeps rank 32 and |Q_dperp d_hat| < 3e-7.
- **handle-press-side goal images are not success states.** Replaying the expert episodes: the handle is pressed at steps 31-44 (success flag true
  for 3-28 steps) and then springs back; at the final step obj_to_target is 0.070-0.072 against a 0.02 threshold, success false in 12/12 episodes.
  The goal image therefore shows a released handle. `sim_true` plans to the environment target, not to the image, so its 12/12 is not evidence about the
  image goal, and the robot-hidden 6/12 (alpha 0) and 7/12 (alpha 0.1) are not interpretable as reaching the goal image either (with the robot hidden
  the goal image is nearly the start image). handle-press-side is **excluded** from F5/F6/F7 as a goal-validity failure. Protocol fix for the next
  round: use the expert frame at first success (any-time goal) for spring-loaded tasks, and record the goal's success predicate for every task.
  drawer-open, plate-slide, faucet-open expert goals are valid (success true, obj_to_target 0.001-0.008 on drawer-open).
- **"Unseen tasks" is wrong.** The released Metaworld dataset covers 42 tasks (`TASK_SET["mw42"]`), which include door-close, drawer-open, faucet-open,
  handle-press-side, plate-slide, window-close, button-press-topdown. The six tasks are in the training data and outside the released six-task
  evaluation. Wording changed everywhere; with exact dynamics the predictor is not used and the encoder is frozen, so the measurement itself is unchanged.
- The true-state drawer-close row (24/24, 24/24) was untraced: the measured number is 22/22 at the eval goal (released config, 22 episodes,
  `docs/02`), and no arm-moved true-cost run exists. Table corrected.
- F6 "every encoder" was false for SigLIP 2 on drawer-open (0.65) and door-close (0.52) and CLIP on plate-slide (0.60): now "the released encoder on
  all four, four of five encoders on each". Drift tolerance (3.8 mm eval-to-variant, 4.2 mm between variants, 5 mm allowed; not "<1 mm"), the
  24-episode p = 0.016 vs 48-episode p = 0.0005, reach-wall alpha 0.1 label, all-24-episode encoder correlations (0.80-0.99 vs 0.51-0.73), CLIP
  reach 0.66, JEPA-WMs bibliography entry (Terver, Yang, Ponce, Bardes, LeCun, "What drives success in physical planning with JEPA world models?").
Verified and kept as is: arXiv 1708.01289 is "Independently Controllable Factors" (Thomas, Pondard, E. Bengio et al.); the audit's suggested
1703.07718 is a different, earlier paper. Softened on the audit's argument: "only the third component is at fault", the multiplicative reading of F2,
"no statistic can separate reach" (reach's 84% removal is above the object range 57-79%, its 77% share is inside), "no fixed rule" (a finite grid).
Audit points left for the user: title/positioning relative to Know Thyself (RAC), CF-JEPA and GS-HER as required comparisons; the conceptual
objection to the projection probe (d contains the incidental arm change, so dperp can remove object signal when the arm change dominates d).
Write-up corrected and recompiled after the audit (9 pages, no overfull boxes): summary, Table 1 (true-state row 22/22, alpha per row), F2 softened to a
hypothesis, F3 all-24 correlations and per-encoder episode counts, F4 reach inside/above range wording, F5 "in the training data, outside the released
evaluation" + handle-press-side excluded + CEM-seed caveat, F6 three of four valid tasks with per-encoder exceptions and n caveats, F7 negative result on the
three valid tasks, F8 "tested rules" + alpha labels + goal-set p-values, protocol section gains a goal-validity bullet, limits gain goal-validity and
"interventions not a partition" bullets, implications rewritten around the under-identification point, future experiments gain the oracle relevance
interface and the probe's two caveats, related work repositioned against Know Thyself with CF-JEPA/GS-HER named as required comparisons, bibliography fixed.
Figures `unseen_enc.pdf` and `mask_unseen.pdf` grey out handle-press-side and print per-bar n.

**Stage 4b complete (20:41 UTC).** Oracle mask, alpha 0.1: drawer-open 0/12, handle-press-side 7/12 (goal invalid, not counted), plate-slide 1/12,
faucet-open 0/12. With alpha 0 (0, 6, 1, 0) the pre-stated rule is not met at either alpha. Verdict: hiding the robot recovers none of the three floored tasks
with valid goals. Stage 5 (projection probe, fixed projector) started 20:41 UTC on drawer-close; stage 4c (alpha-0 unmasked controls) follows, then ALL_DONE.

**Stage 5 result (21:09 UTC), change-conditioned nuisance projection, offline, exact dynamics, alpha 0, 24 episodes, K = 32, fixed rank-aware projector.**
Ordering accuracy, columns v0 (eval) / v1 / v2 / v3 / v4; H16 mag 0.7 (H8 mag 1.0 in brackets where different):
| cost | drawer-close | reach-wall (v0) | plate-slide |
|---|---|---|---|
| plain (released) | 0.893 / 0.890 / 0.712 / 0.664 / 0.711 | 0.838 | 0.447 / 0.548 / 0.405 / 0.447 / 0.329 |
| dperp (span(U) with the d direction removed) | 0.900 / 0.896 / **0.678 / 0.612 / 0.662** (worse) | **0.993** | 0.495 / 0.582 / 0.710 / 0.742 / 0.666 |
| full (span(U)) | 0.895 / 0.887 / 0.805 / 0.788 / 0.830 [0.914 / 0.909 / 0.806 / 0.772 / 0.816] | 0.889 [0.943] | 0.490 / 0.546 / 0.654 / 0.680 / 0.604 |
| gated (rho) | 0.901 / 0.898 / 0.752 / 0.707 / 0.756 | 0.897 | 0.485 / 0.547 / 0.501 / 0.536 / 0.379 |
rho (fraction of ||d||^2 inside span(U)): drawer 0.43-0.57 at every variant, reach-wall 0.51 (0.43 at H8), plate-slide 0.11 at the eval goal, 0.45 arm-moved.
Verdict against the pre-stated criteria (drawer arm-moved >= 0.85 with eval >= 0.85; reach >= 0.80; plate-slide eval > 0.6; one setting): **FAIL for all three costs.**
- dperp, the candidate, makes the drawer's arm-moved accuracy *worse* (0.70 -> 0.65). As the audit predicted: d = g - z_start contains the incidental
  start-to-goal arm change, so protecting d protects the arm nuisance. It is near-perfect on reach (0.99), where d is the task.
- full improves the drawer (0.70 -> 0.81, below the 0.85 bar) and, unlike the goal-set in closed loop, does not break reach offline (0.84 -> 0.89): a
  32-direction subspace in a 98k-dim latent removes little, which is also why it cannot reach the bar. Not pursued further without a decision.
- rho does not separate reach (0.51) from the drawer (0.43-0.57): the latent-space version of the cost-share result (F4). The gate idea is dead in this form.
- plate-slide eval goal unchanged by all three (0.45 -> 0.49); the arm-moved variants rise to 0.70-0.74 under dperp, consistent with the decoy reading of F6.
Candidate 1 closed. Stage 4c (alpha-0 unmasked controls) running next; then ALL_DONE and the watcher stops the pod after the lab-6 backup.

**Stage 4c, alpha-0 unmasked controls (22:06 UTC):** handle-press-side 0/12, drawer-open 0/12, faucet-open 0/12. So the alpha-0 mask runs compare to 0/12
baselines at the same alpha; the F7 verdict stands (no valid floored task recovered by the mask at either alpha). `ALL_DONE` written 22:06 UTC; queue
complete. Total spend on this queue about $6; balance about $7-8. Decision pending from the user (options: stop spending and keep the diagnosis for the
next cycle; secure compute and commit to a trained method; or change problem).
Pod B stopped by the watcher at 22:15 UTC (API status EXITED; stopping wipes its disk). Complete results: lab-6 `~/jepa_wms_podB_backup/failure_attribution/`
and the laptop `artifacts/raw_logs/pod_runs/podB_full/` (full copy incl. ep_*.npz) plus `podB/` (small files). No pod is running; nothing is queued.

## 2026-10-19 (UTC 10-18 18:27-20:26): are the failures real? weighting, probes, LeWM (pod C)

Pod C `5xjulqn3c0vbjv` (RTX A4000, $0.17/hr, community, `supportPublicIp: true`; two earlier pods without a public SSH port were terminated
after 3-18 min). Ran 18:59-20:26 UTC, about $0.33 in all incl. the dead pods; stopped (EXITED). Results: `artifacts/raw_logs/pod_runs/podC/`.
Scripts: `code/jepa-wms/evals/failure_attribution/area_probe_mw.py`, `code/lewm_explorations/scripts/exp8_area_weighting.py`,
`exp9_cube_weighting.py`, and the `agent_moved_*` goal conditions added to `code/third_party/stable-worldmodel/scripts/plan/eval_goalshift.py`.
All kinematic (qpos set, mj_forward/pymunk placement, render); arm moves are minimum-joint-change hand displacements (damped Jacobian
pseudo-inverse) after a first version with random joint directions was found to swing the elbow (numbers barely changed: reach 6.2 -> 5.4).

**Exchange rate, frozen DINOv2 ViT-S/14 patch cost** (median over 30 resets; cost of the hand moved 5 cm / cost of the object off by r, the task's
success distance): reach 5.4, reach-wall 4.6, button-press 12.7, door-lock 11.2, drawer-close 4.2, faucet-close 5.5, faucet-open 4.4, box-close
2.4, drawer-open 7.2, plate-slide 3.2, push 7.2 (`mw_ik/`). Hand displacement costing as much as failing the task: 0.4-2.0 cm. Pixel MSE
is more arm-dominated still (6.4-34). OGBench cube (`exp9_ik/`, arm rendered semi-transparent by OGBench): pixel 0.97, DINOv2 4.8, LeWM cube
encoder 0.02.
**PushT** (`exp8/`, 300 states, per px of displacement, agent / block): pixel 0.19-0.32, DINOv2 0.34-0.40 (reproduces the earlier 0.33-0.37),
LeWM 0.02-0.04. Varying agent / block size (area ratio 0.03-0.25): pixel tracks area, DINOv2 barely (0.30-0.50), so "the metric weights by
area" is wrong for DINOv2; it weights the embodiment by something other than size. Offline ordering of block progress with task-equivalent
goals (agent moved 80 px, block same, 200 groups): no drop for any metric (pixel 0.967/0.956, DINOv2 0.947/0.941, LeWM 0.859/0.851).
**Linear probe** (ridge, 5-fold CV, 600 renders, arm in random poses; RMSE of the 3-d object position vs r): DINOv2 patches with the robot
visible resolve the object far below r on all six tasks: drawer-open 0.71 / 3.0 cm, drawer-close 0.77 / 5.5, button-press 0.53 / 2.0,
faucet-open 2.75 / 7.0, plate-slide 3.02 / 7.0, push 3.85 / 5.0 (`mw_probe/`); cube DINOv2 2.85 / 4.0, LeWM 2.28 / 4.0. So Failure B is not
missing information: the object is linearly present; the uniform patch cost does not weight it.
**LeWM PushT closed loop, task-equivalent goals** (`goalshift_results/`, seeds 0 and 42 x 50 goals, block-only success / full success):
identity 98% / 90%; agent moved 40 px 97% / 63%; 80 px 89% / 42% (Fisher vs identity p = 0.018); 160 px 86% / 25% (p = 0.003). The
earlier no-agent goal gave 80.5% block-only (4 seeds). A jointly trained encoder that weights the agent at 0.02-0.04 of the block still
loses 12 pp of block success when only the agent's place in the goal changes.
**Reading.** Failure A holds across all 11 Metaworld tasks and the cube, not only drawer-close. Failure B is a weighting failure, not a
representation failure, so A and B are one problem: the goal metric's weighting of entities is set by the encoder's training, not by the task
(frozen DINOv2 over-weights the robot; LeWM under-weights it), and the information needed for the right weighting is linearly present.
