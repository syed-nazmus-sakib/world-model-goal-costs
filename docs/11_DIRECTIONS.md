# Directions we are pursuing (2026-10-23)

Target: CVPR 2027 main track, deadline 2026-11-16. This file says what the paper is, which directions are open and
closed, the decision rules, and the plan to the deadline. Evidence: docs/01-07 (diagnosis), docs/10 (learned costs).

## 1. The paper

**Working title:** Latent goal costs fail for a reason: diagnosing and fixing the planning objective of frozen
world models.

1. **Diagnosis (done; docs 01-07, `writeup/`).** The released latent goal cost measures image similarity, and the robot arm dominates it:
   - about 75% of the start-to-goal cost sits on robot patches on all 8 object tasks;
   - this holds for 5 encoders;
   - moving only the arm in the goal image costs 12 of 24 episodes with the real planner (McNemar p = 0.004).
2. **Fix: a criterion-aligned cost head (in progress).** A small head on the frozen latents, trained with the task's success distance at training time only, replaces the released cost. The world model, encoder and search are unchanged. Evidence so far:
   - Cube +11.5 points (p = 0.017);
   - plate-slide 0 → 8/12;
   - TwoRoom 58 → 99 with the time target.
3. **Limits (found; 2026-10-23, 9 Metaworld tasks).**
   - The head fixes ranking. It does not fix reaching contact: it wins only plate-slide (0 → 8/12) and stays at 0/12 on push, button-press-topdown, door-open, window-open, faucet-open and drawer-open.
   - On those tasks near-contact ranking is already near-perfect, so the failure is the approach phase and the search, not the cost near success.
   - It also lowers reach-wall (23/24 → 7/12).
   - **Consequence for the paper:** on Metaworld the diagnosis must separate a cost error near success from an approach / search error far from it. The learned cost is a method only where the first dominates (Cube, plate-slide, TwoRoom).

**Position against the closest work.**
- *Supervise What Decides Success* (2610.01224) uses the same kind of labels as an auxiliary world-model loss and gains +3.4 on Cube. We keep the world model frozen and gain +11.5 on Cube.
- TEMPO (2610.04988) and NGC (2610.25013) fix navigation geometry without labels; our time head reproduces TEMPO's TwoRoom result. We do not claim that.
- The claimed contribution is the diagnosis plus the cost that targets what decides success, evaluated on manipulation where temporal / geometric costs inherit the arm bias.

**Honest weakness:** the head needs success labels at training time (simulator states). Reviewers will ask how
this transfers to real robots. Answer to prepare: the labels are the task's own success test, needed only offline,
and the same supervision as the competing paper.

## 2. Open directions, in priority order

| # | Direction | Status | Next step | Decision rule |
|---|---|---|---|---|
| 1 | State head on all 9 Metaworld tasks (closed loop) | **done: FAIL** (wins 1 of 8, reach-wall 23/24 → 7/12; docs/10) | keep as evidence that ranking is not the bottleneck on Metaworld | was: ≥ 3/12 better on ≥ 4 of 8 object tasks, controls ≥ 10/12 |
| 2 | Baselines for the paper | **TEMPO re-implemented on our Metaworld encodings** (`mw_tempo_train.py`, `--cost tempo:<map>:<w>`, w = 0.5 and 1; offline map only), running 2026-10-24 on the same 8 tasks and episodes; released + min done | run; later TEMPO's own code on LeWM Cube / PushT | report all on the same goals |
| 3 | **Reach-then-change cost** (approach phase; docs/10 last section) | **Stage 1 PASSED** (2026-10-23): robot-free change term + oracle reach: button 12/12, push 7/12, drawer-open 6/12, fallback 12/12 (released 0/12). Stage 2 (images only), 12 episodes: v1 button 3/12, push 0/12; **v3 button 3/12, push 0/12, drawer-open 1/12, plate-slide 0/12, faucet-open 0/12, window-open 0/12** (one-camera 3D localisation: start-arm shadows, lift height, drawer-front centroid; docs/10 rows 12-12e); controls drawer-close 10/12, reach-wall 8/12; released controls and door oracles running | v3b (`reachrf3`: start-arm exclusion, relative change strength ≥ 0.5, table-height lift, goal-proprio second phase, 6 cm fallback stand-off) is implemented but its run was cancelled on 2026-10-24 to give the compute to the learned route; the image-only line is reported with v3 | Stage 2 PASS: ≥ 4/12 on ≥ 2 tasks (v1 and v3 so far: FAIL) |
| 4 | **Approach-aware learned head (rtc)**: one head across 8 tasks, target = hand-to-object distance until the object moves, then the success distance (labels at training time only) | offline near-perfect; v1 closed loop button 0/12, push 3/12, drawer-open 0, plate-slide 1, faucet 0, window 2, drawer-close 12/12 (exploited off the training distribution). **Walks + 3-head ensemble (β = 1): button 1/12, push 3/12, drawer-open 1/12, plate-slide 5/12, but the hand reaches the goal hand in 43/48 episodes (v1 button + push: 8/24) and moves the object in 43/48: exploitation solved, object state near the goal not resolved (docs/10)**; the remaining 4 tasks and the β = 0 ablation were cancelled (superseded). **Head + robot-free change term (`costs.HeadChangeObjective`, `headrf:<head>:<robot mask>:1.0:0.5`): button 0/12, push 5/11** (button: gripper open, fingers straddle the button, the button is hidden under the arm so the change term cannot see it); the other 6 tasks running | **H4/H5/H6 queues running**: (H4) aimed near-goal jitter data (`mw_pilot_data --jitter`, kind 4), one task at a time; (H5) the planner's own frames on the training episodes (`run_closed --ep_start 100 --dump_dir`, kinds 5/6, button + push; `mw_merge_planner_frames.py`); (H6) retrain the ensemble on expert + candidates + walks + jitter + planner frames, closed loops head + rf and head alone on button and push, head + rf on the other 6 | PASS: ≥ 6/12 on ≥ 4 of 7 object tasks, controls ≥ 10/12 (v1 FAIL; ensemble alone FAIL) |
| 5 | Does the structure hold outside Metaworld? LeWM PushT with the reach-then-change oracle (`oracle_cost.py` modes `sim_latent_reach`, `sim_true_reach`) | **done: FAIL, scoping result** (2026-10-24): released + reach 36%, true 48%, true + reach 50% (blocks 36 / 54 / 58). The pusher starts at the block; no approach failure to fix | report PushT as the control where the structure does not apply | was: ≥ 10 points over sim_true or over released + min (40%) |
| 6 | Fewer labels | open | rtc head with 10% / 1% of labelled frames | within 2/12 of the full head |
| 7 | Paired statistics | to do | save per-episode success, McNemar / bootstrap over goals | – |

## 3. Closed directions (do not reopen without new evidence)

| Direction | Why closed | Where |
|---|---|---|
| Area-invariant (top-k) patch cost | fixed offline, failed closed loop | docs/03 |
| Goal set from arm-moved goals / world-model goal set | fixes drawer variants, breaks reach-wall (0/24) | docs/07 |
| Nuisance (arm) subspace projection | failed all pre-stated bars | docs/07 stage 5 |
| Oracle robot mask | drawer-open 0/12, plate-slide 1/12, faucet-open 0/12 | docs/07 stage 4b |
| Method A: action guidance | every w ≠ 1 lowers success | docs/10 |
| Time head as the method | covered by TEMPO; arm-biased on manipulation | docs/10 |
| Method D: evid / evid_time (few examples) | 0/12 on object tasks, breaks reach-wall | docs/10 |

## 4. Plan to the deadline

| Dates | Work | Budget |
|---|---|---|
| Oct 23-30 | finish the 9-task state head; TEMPO baseline; approach-phase test (direction 3) on drawer-open / button | about $2 of pod credit |
| Oct 30-Nov 6 | seeds and paired statistics on the tasks that move; label-fraction ablation (direction 4) if budget allows | A100s if available |
| Nov 6-16 | write: diagnosis sections exist in `writeup/main.tex`; add method, results, limits | – |

## 5. Working rules

- Experiments run on a RunPod pod, never on the laptop. Check CUDA, EGL and disk first (`scripts/pod/check_egl.sh`): on 2026-10-23 three pods in a row came up on a host whose GPU could not initialise CUDA (`cuInit` 999) although `nvidia-smi` listed it; terminate such a pod at once.
- Prior art for the reach-then-change direction, checked 2026-10-23: subgoal methods (SAGE 2607.17973, "Mind the Gap" 2607.12547, LAGO 2606.20627, "The Planning Limits of Latent World Models" 2609.39235: expert subgoals 76% vs 23% on Metaworld), change masks (DDP-WM 2602.01780), robot masks (Know Thyself), contact gate in TD-JEPA (2607.25337). None derives reach-then-move shaping from the goal image for a frozen world model.
- Pod runs end with `ALL_DONE`; `scripts/pod/backup_and_stop.sh` copies the results home and stops the pod.
- State a decision rule before each run, and report a failed rule as a failure.
- Never commit or push `artifacts/`, `reports/`, `research_notes/`, `writeup/`, `meeting/`, `candidate_figs/`, the PDFs or `.env`.
