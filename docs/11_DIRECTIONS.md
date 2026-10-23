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
| 2 | Baselines for the paper | to do | TEMPO released code; released + min; *Supervise What Decides Success* if code exists | report all on the same goals |
| 3 | **Reach-then-change cost** (approach phase): a Metaworld reward says "hand to the object, then object to its target"; derive that order from the start and goal images with no labels (robot map from proprio prediction per patch, change map = non-robot patches that differ between start and goal; reach term pulls the hand to the change region until the object moves; fallback to the released cost when nothing but the robot changes) | **Stage 1 running (2026-10-23)**: oracle version with simulator hand and object positions (`run_closed.py --planners sim_latent_reach sim_true_reach`, `costs.ReachThenChangeOracle`). Design change after the one-episode smoke and before the 12-episode runs: the hand reached 1.4 cm from the drawer handle in two replans, nudged the drawer 3-4 mm and never pulled; so the "moved" gate went from 5 mm to 2 mm and the reach term now saturates within 2 cm of the object (a dense reach term collapsed the search onto holding still at the handle) | Stage 2: the two maps from DINOv2 patches | Stage 1 PASS: released cost + oracle reach ≥ 6/12 on ≥ 2 of {drawer-open, button-press-topdown, push, door-open}, reach-wall ≥ 10/12; else the idea is dead. Stage 2 PASS: ≥ 4/12 on ≥ 2 tasks |
| 4 | Fewer labels | open | state head with 10% / 1% of labelled frames, rest time-only | within 2/12 of the full state head |
| 5 | Paired statistics | to do | save per-episode success, McNemar / bootstrap over goals | – |

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
