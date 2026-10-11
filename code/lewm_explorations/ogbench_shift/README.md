# OGBench-Shift: goal images that look different from the training data

**Working title:** *Same Goal, Different Look: Appearance-Robust Goal Specification for Visual Goal-Conditioned Control*
Target: CVPR 2027 (registration 10 Nov, paper 16 Nov, supp 23 Nov 2026, AoE).
Status (2026-10-05): benchmark renderer implemented and verified; no agents trained yet.

---

## 1. The idea in one paragraph

Visual goal-conditioned agents (offline GCRL on OGBench, JEPA world-model planners) are trained and evaluated
with goal images rendered exactly like the training observations. In practice, a goal photo is taken under
different lighting, with a different background, with a recoloured object, or from a slightly different camera.
The **physical goal is identical; only its appearance changed**. We (i) build **OGBench-Shift**, a benchmark that
re-renders the *goal image only* under controlled, held-out appearance shifts while the observation stream
stays nominal, (ii) show that state-of-the-art visual GCRL agents and world-model planners collapse under it,
and (iii) propose a fix that exploits what goal-conditioned RL already has and generic vision does not: a
**temporal-distance signal that is independent of appearance**. A goal's representation should be defined by
*how far it is from other states* (dual goal representations, temporal distances). Training the goal encoder
to reproduce the same temporal-distance profile for appearance-augmented goal images gives
invariance supervision for free, with no labels and no target-domain data.

## 2. Why this, and why now (evidence)

**Our own preliminary evidence (LeWM on PushT, stable-worldmodel `eval_goalshift.py`, 50 eps × 3 seeds):**

| goal-only shift | success | block-only success |
|---|---|---|
| none | 84–92 % | 92–100 % |
| background colour | 2–10 % | 44–70 % |
| agent colour | 22–34 % | 82–90 % |
| block colour | 8–20 % | 52–68 % |

And a negative result that rules out the cheap fix: the latent offset induced by a colour shift is about 0.8×
a 25-step state change and is spread **isotropically** across the latent (its energy in any rank-r subspace —
PCA, temporal-difference PCA, or action-controllable subspace — equals that of real state progress). No linear
post-hoc projection can separate nuisance from state (`../results/exp3_subspace.txt`). A learned invariance is needed.

**Literature gap (verified by a 5-agent review, see `../LITERATURE_REVIEW.md`):**
- Robustness work in latent world models shifts the *observations* (or both), not the goal alone:
  JEPA-Bisim 2602.18639 (bisimulation, background changes on PointMaze/PushT), CF-JEPA 2610.00727
  (controllable/uncontrollable factorisation, observation distractors only), AdaJEPA 2606.32026 (test-time
  adaptation inside MPC; needs target-domain transitions), Flow-JEPA 2608.29029, Reimagination 2506.16565
  (test-time observation interventions), Better Slots 2608.12078 (object-centric WMs more robust than LeWM).
- Offline GCRL from pixels has **no** appearance-shift study: OGBench visual-* is evaluated only in-distribution;
  the review's arXiv search for OGBench + distractor/background/viewpoint returned nothing.
- Goal representations: Dual Goal Representations 2510.06714 (goal = its temporal distances to all states;
  appearance-agnostic *in principle*), TMD 2509.20478, TRA 2502.05454 (aligns language/image goals with
  successor features), "Do Better Goal Representations Improve GCRL?" 2609.39901 (state pathway is the
  bottleneck — state-based only), Action-Sufficient Goal Reps 2601.22496.
- CVPR fit: visual invariance and benchmarks exposing SOTA failures are classic CVPR material; CVPR 2027
  explicitly lists "Physical AI (world models, VLA, planning)" and "datasets and benchmarks".

## 3. Research questions and hypotheses

- **RQ1 (benchmark):** How much do visual GCRL agents (GCIVL, CRL, HIQL, GCFBC, + dual/TRA/VIP goal reps) and a
  JEPA planner (LeWM) lose when only the goal image's appearance shifts? Which shift families hurt most?
  *H1: large drops (> 20 pt) for colour/lighting/camera at severity ≥ 2; image-space corruptions hurt less.*
- **RQ2 (does temporal structure already help?):** Are dual goal representations more robust than raw-pixel goal
  encoders, because they are trained to encode temporal relations? *H2: only marginally. The encoder still sees
  pixels; nothing in training forces invariance to unseen appearance.*
- **RQ3 (method):** Does supervising goal-side appearance augmentations with a fixed temporal-distance target
  (**TD-anchored goal invariance**, §5) beat generic augmentation and frozen foundation encoders on
  **held-out** shift families? *H3: yes, because the target is appearance-free by construction while generic
  augmentation only regularises the policy input.*

## 4. Benchmark: OGBench-Shift (implemented in `shift.py`)

- Wraps any OGBench visual env (`visual-cube-*`, `visual-scene-*`, `visual-puzzle-*`, `visual-antmaze-*`, …).
- During `reset`, OGBench poses the scene at the goal, renders the goal and restores the start state. The wrapper
  snapshots the **full MjData** at goal-render time and re-renders it with modified appearance. Verified
  bit-exact reproduction of the nominal goal for cube and antmaze (≤ 1/255 on scene, from transparent-glass
  rasterisation); the observation stream is verified unchanged (`test_shift.py`, montage `shift_montage.png`).
- `info['goal']` = shifted goal, `info['goal_nominal']` = original.
- Families × severity {1, 2, 3}:

| family | what changes | type |
|---|---|---|
| `floor_color` | floor/grid material tint | appearance |
| `light` | global light intensity and colour | appearance |
| `robot_color` | robot / ant body materials | appearance |
| `object_color` | cube colour (single-cube tasks only; colour is semantic with several cubes) | appearance |
| `camera` | small viewpoint perturbation of the render camera | geometric |
| `blur`, `noise`, `jpeg` | image-space corruptions of the goal | corruption |

**Protocol (fix before experiments):**
- Split families into **train-augmentation families** and **held-out test families**, so we never train on the test
  shift (a standard reviewer complaint about robustness papers). Proposed split: augment with `floor_color` and
  `light`; test on `robot_color`, `object_color`, `camera`, and corruptions. Report both splits, swapped.
- Metrics: success (OGBench task success), **relative drop** vs nominal, area under the severity curve. Also
  report the goal-representation distance between shifted and nominal goals, normalised by the distance between
  states 1–5 steps apart (how many "steps" the shift is worth).
- Per env: the 5 standard evaluation tasks × 20 episodes × 3 training seeds.

**Possible extensions (only if time allows):** distractor objects (needs XML edits), texture swaps
(`model.tex_data` + texture re-upload), LeWM/PushT version via stable-worldmodel (`eval_goalshift.py` already exists).

## 5. Method candidates (in order of priority)

1. **TD-anchored goal invariance (main proposal).** The dual goal representation φ(g) is trained so that
   φ(g)·ψ(s) ≈ temporal value V(s, g). Add a consistency term on the goal side only:
   `L_inv = E_{s,g,T} || φ(T(g))·ψ(s) − sg[φ(g)·ψ(s)] ||²` (or match φ(T(g)) to sg[φ(g)] directly),
   with T a random appearance augmentation (colour jitter, light jitter, small homography, blur). The target is
   the *temporal* relation of g to states, so invariance is supervised by dynamics, not by labels. Observation
   stream untouched. Variants: (a) match representations, (b) match values to all states in the batch
   (distributional, more faithful to the dual definition).
2. **Generic augmentation baselines (must have):** OGBench's random crop (`p_aug`), colour jitter on goals,
   colour jitter on goals and observations, domain randomisation. If (1) does not beat these on held-out
   families, the method claim dies; the benchmark stays.
3. **Frozen foundation goal encoders:** DINOv2 / SigLIP features of the goal (with a learned head). Are they
   already invariant? Cheap analysis axis.
4. **Training-free test-time canonicalisation:** retrieve the nearest *nominal* dataset frame to the shifted goal
   under an invariant feature (e.g., DINOv2 after per-channel normalisation) and use it as the goal. Strong
   simple baseline for small shifts.

## 6. Kill tests (do these first, about 1 week)

- **K1 (is the benchmark interesting?)** Train GCIVL (`agents/gcivl/original.py`) and GCIVL+dual
  (`agents/gcivl/pixel/dual.py`) on `visual-cube-single-play-v0` and `visual-antmaze-medium-navigate-v0`
  (500k steps, 1 seed first), evaluate with `eval_shift.py`.
  **Kill the benchmark angle** if the average drop at severity 2 is < 15 pt across families for both agents.
- **K2 (does the method work?)** Implement `L_inv` in the dual agent (§5.1). Train augment-on
  {floor_color, light}-style jitter; test on held-out families.
  **GO** if it recovers ≥ 50 % of the drop on held-out families **and** beats goal colour-jitter augmentation by
  ≥ 5 pt. Otherwise pivot to "benchmark + analysis" (still a viable CVPR datasets/benchmark paper if K1 is
  strong and it covers ≥ 4 envs and ≥ 5 agents).

## 7. Setup on another PC

1. Copy this folder (`ogbench_shift/`) and clone your repo:
   `git clone <your dual-goal-representations remote>`; install per its README (JAX, `ogbench==1.2.x`, `mujoco`).
   Rendering needs `MUJOCO_GL=egl` (headless GPU) or `osmesa`.
2. Check the renderer: `cd ogbench_shift && MUJOCO_GL=egl python test_shift.py` → should print `ok` for 3 envs and
   write `shift_montage.png`.
3. Train (OGBench downloads datasets to `~/.ogbench/data` on first use; visual datasets are large, so check
   disk). **Pass `--save_interval` ≤ `--train_steps`**; the repo default (1M) is larger than the 500k used in
   `hyperparameters.sh`, so no checkpoint would be saved:
   ```
   python main.py --env_name=visual-cube-single-play-v0 --agent=agents/gcivl/pixel/dual.py \
     --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 \
     --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 \
     --agent.encoder=impala_small --eval_episodes=15 --save_interval=500000 --seed=0
   ```
   (Baseline: same with `--agent=agents/gcivl/original.py`, dropping the `rep_*`/`goalrep_dim` flags. See
   `hyperparameters.sh` lines ~272–285 for all visual configs.)
4. Evaluate on OGBench-Shift (run from the repo root):
   ```
   PYTHONPATH=/path/to/ogbench_shift MUJOCO_GL=egl python /path/to/ogbench_shift/eval_shift.py \
     --env_name=visual-cube-single-play-v0 --agent=agents/gcivl/pixel/dual.py \
     [the same --agent.* flags as training] \
     --restore_path=exp/goal_representation/Debug/<run_dir> --restore_epoch=500000 \
     --eval_episodes=20 --out=shift_results.csv
   ```
   Note: `eval_shift.py` follows `main.py`'s agent-construction code but has **not been run end-to-end** (no
   JAX env or visual datasets on the machine where it was written). Expect small fixes, e.g. dataset-class names.

## 8. Six-week plan

| week | goal |
|---|---|
| 1 (Oct 6–12) | Setup, train 2 agents × 2 envs, run K1. Freeze the benchmark protocol. |
| 2 (Oct 13–19) | Implement `L_inv` + augmentation baselines; K2 on cube-single. Decide method vs benchmark-only. |
| 3 (Oct 20–26) | Scale: 4 envs (cube-single/double, scene, antmaze-medium) × 5 agents (GCIVL, CRL, HIQL, GCFBC, dual) + LeWM PushT. |
| 4 (Oct 27–Nov 2) | Held-out-family evaluation, 3 seeds; foundation-encoder and retrieval baselines; analysis (shift worth in "steps"). |
| 5 (Nov 3–9) | Figures, ablations, writing; register by Nov 10. |
| 6 (Nov 10–16) | Final runs only for claims; submit Nov 16; supplementary by Nov 23. |

## 9. Files

- `shift.py`: `ShiftedGoalEnv` gymnasium wrapper (goal-only appearance shifts, 8 families × 3 severities).
- `test_shift.py`: fidelity and observation-invariance checks, montage.
- `eval_shift.py`: evaluate a trained dual-goal-representations agent on all shift families → CSV.
- `shift_montage.png`: example goals (rows: cube-single, antmaze-medium, scene; cols: nominal, then each family at sev 1 and 3).

## 10. Risks

- **Augmentation might simply solve it.** Then the paper is benchmark + analysis; make held-out families decisive.
- **Concurrent work:** goal-side shift is thin now, but JEPA robustness papers appear weekly. Cite JEPA-Bisim,
  CF-JEPA, AdaJEPA and position the work as *goal-only*, *offline GCRL*, *held-out families*.
- **OGBench visual data size and training time:** 500k steps of the impala_small agents takes hours per run;
  budget for about 30 runs.
