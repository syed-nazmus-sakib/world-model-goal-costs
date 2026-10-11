# Literature review & direction choice — CVPR 2027 (2026-10-05)

Five parallel reviews, ~200 papers, every ID checked on arXiv by the review agents.
CVPR 2027: registration 10 Nov, paper 16 Nov, supp 23 Nov (AoE). New 2027 topics include
"Physical AI (World Models, VLA, Planning, Simulation)", "Data synthesis, datasets and benchmarks",
"Representation learning". Contemporaneous cutoff: 15 Sep 2026 (later work must still be cited).

## 1. Saturation map (2026)

| Sub-area | 2026 papers | Verdict |
|---|---|---|
| Planning cost / latent geometry on LeWM-type models | ≥11 (TD-JEPA 2607.25337, Objective-Is-Bottleneck 2608.12959, SCALE 2608.16287, LEAP 2609.03294, CGS 2609.35603, ProWorld 2608.01926, Traj-LeWM, DA-LeWM, D-JEPA, RC-aux, Temporal Straightening) | **Saturated** (our idea 1 lived here) |
| Long horizon / hierarchy / amortised planners | ≥9 (HWM 2604.03208, FF-JEPA, Mind the Gap, Var-Length, SAGE, LePlanner, Dual-WM, Anchored Planning, Planning Limits 2609.39235) | **Saturated** |
| Action-discriminative / counterfactual *losses* | ≥9 (AD-WM 2609.30264, Delta-JEPA, ATM, ActSWM, CoCo 2608.04653, CAER, DWM, No-Gaussian-Required, AVL-JEPA, Phys-Grounded JEPA) | **Saturated** |
| Robustness to observation distractors / TTA | 5–6 (JEPA-Bisim 2602.18639, CF-JEPA 2610.00727, AdaJEPA 2606.32026, JEPA-TTT 2610.00722, Flow-JEPA) | Moderate |
| WM evaluation / diagnostics (JEPA + video) | ≥10 (ARC-Bench 2609.05461, ACPC 2608.12939, WorldArena, MiraBench, WorldLens…) | Crowded |
| Video WM architectures / WAMs / policy evaluation | dozens, industry-scale | **Saturated, infeasible on 1 GPU** |
| Long-horizon value learning on state OGBench | ≥9 (TRL, DCRL, GITA, MQE, LAVL…) | Crowded |
| **What data makes a world model plannable** (collection policy, conditional action coverage, budget allocation) | theory 2607.22430 (synthetic 2D); FIRM-WM 2609.22816 (fixed branching, no sweep); PLDM / PlayWorld / WorldSync (confounded, not marginal-matched) | **Thin / empty** |
| Goal-image appearance shift (goal-side only) | none isolates it | **Thin** |
| Pixel OGBench GCRL, frozen video encoders as quasimetric goal distances | few (TMD, Dual Goal Reps, MQE, Ms.PR) | Thin |

## 2. What CVPR rewards in this area
Vision-first contributions with control as the downstream test (NWM oral CVPR'25; GeoWorld CVPR'26);
big efficiency multipliers; representation/interface between video and action (Moto, TraceGen);
benchmarks that expose SOTA failures (WorldLens oral). Plain "better PushT success" papers are weak fits;
only GeoWorld from the JEPA-planning wave reached CVPR'26 main track.

## 3. Our own evidence so far (all in results/)
1. Training-free probe-geometry terminal cost: +2.6 pt held-out (killed; saturated area).
2. Horizon collapse (H5 88 → H6 42 → H8 14 %): real but explained by Planning Limits / Objective-Is-Bottleneck.
3. Goal-appearance shift: LeWM 88 → ~6 %; nuisance offset isotropic in latent → no linear fix (killed).
4. Frozen LeWM: error 1.89× higher for actions off p(a|s) but on p(a) (CI 1.79–1.98), robust to contact controls.
5. Fixed budget (50k windows), matched action marginal, 2 train seeds × 3 eval seeds:
   EXP (expert) 41.3 % | RND1 (decorrelated) **72.3 %** | RND4 67.3 % | RND8 65.7 %;
   counterfactual error 0.254 / **0.065** / 0.074 / 0.079. Decorrelation +31 pt; branching no gain.

## 4. Recommendation — Direction A

**"Demonstrations Are the Wrong Data for World Models: Conditional Action Excitation Governs Visual Planning"**

Thesis: the 2026 wave of action-discriminative/counterfactual *losses* (≥9 papers) is compensating for a
*data* problem — expert data couples actions to states, so the predictor never identifies counterfactual
dynamics (as 2607.22430 predicts in theory). Under a matched action marginal and fixed budget, the
conditional excitation of the collection policy, not the amount or marginal diversity of actions, decides
counterfactual accuracy and planning success; and once actions are decorrelated, per-state branching
(FIRM-WM-style) buys nothing.

Why it is solid: large effects (+31 pt, 3.9× error) with clean controls already replicated; theory link;
review agents independently found the gap empty; cheap (15M-param models, ~17 min each on the 5090).

Planned claims
- C1 mechanism (frozen models): off-p(a|s) error ratio across LeWM / DINO-WM.
- C2 causal dose–response: mix expert/decorrelated actions (0–100 %), measure conditional excitation, show it
  predicts CF error and success monotonically; ≥3 envs (PushT, OGBench-Cube, TwoRoom/Reacher).
- C3 data × loss 2×2: add an action-discriminative loss (inverse dynamics / AD-WM-style) on EXP vs RND data.
  If the loss's gain disappears on decorrelated data, the loss wave is fixing a data problem (headline).
- C4 allocation: state coverage vs branching frontier (negative result, explained by theory).
- C5 vision angle: does data type change robustness to goal/observation appearance shift? (cheap, exploratory)

Nearest competitors and differentiation: 2607.22430 (theory only, synthetic) → pixels, planning, measured
excitation; FIRM-WM 2609.22816 → sweeps branching, shows it is unnecessary; WorldSync 2608.24885 /
PlayWorld 2603.09030 / PLDM 2502.14819 → matched marginal + fixed budget + mechanism; AD-WM / CoCo → data vs loss.

Kill test (3–4 days)
- K1 replication in a second, 3-D contact-rich env (OGBench-Cube): RND1 − EXP ≥ +10 pt at equal budget.
- K2 2×2 on PushT: inverse-dynamics loss gain on EXP vs on RND1.
  GO if K1 holds; K2 decides the framing (data-substitutes-for-loss vs complementary).

## 5. Runners-up
- **B. Goal-appearance shift for visual GCRL ("OGBench-Shift") + temporal-distance-invariant goal encoders.**
  Best CVPR fit and the user's expertise (Dual Goal Reps, TMD); gap empty. Risk: needs a new method; our linear
  fix already failed. Could become the C5 section of A.
- **C. Quasimetric goal distances on frozen video foundation encoders (V-JEPA 2 / DINOv2) for pixel OGBench.**
  Thin, feasible; risk of competition from large-data video value models (RynnValue, VLD, TimeRewarder).

Avoid: new planning costs/geometry, hierarchy, action-discriminative losses as a method, video-WM training.
