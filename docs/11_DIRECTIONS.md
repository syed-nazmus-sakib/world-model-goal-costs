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
3. **Limits (found).**
   - The head fixes ranking. It does not fix reaching contact: drawer-open and button-press-topdown stay at 0/12 although near-contact ranking is near-perfect.
   - This separates a cost error from a search / approach error, which the diagnosis alone could not.

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
| 1 | State head on all 9 Metaworld tasks (closed loop) | running (pod G) | finish; add seeds where the head wins | works if ≥ 3/12 better on ≥ 4 of 8 object tasks, controls ≥ 10/12 |
| 2 | Baselines for the paper | to do | TEMPO released code; released + min; *Supervise What Decides Success* if code exists | report all on the same goals |
| 3 | Approach phase (reaching contact) | open question | (a) object-progress time target over frames where the object moves; (b) expert first-contact state as an intermediate goal; (c) longer horizon | drawer-open or button-press-topdown ≥ 4/12 |
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

- Experiments run on a RunPod pod, never on the laptop. Check EGL and disk first (`scripts/pod/check_egl.sh`).
- Pod runs end with `ALL_DONE`; `scripts/pod/backup_and_stop.sh` copies the results home and stops the pod.
- State a decision rule before each run, and report a failed rule as a failure.
- Never commit or push `artifacts/`, `reports/`, `research_notes/`, `writeup/`, `meeting/`, `candidate_figs/`, the PDFs or `.env`.
