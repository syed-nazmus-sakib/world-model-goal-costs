# Literature review v2 and direction choice (CVPR 2027), 2026-10-15

Supersedes `05_LITERATURE_REVIEW.md`. Built from six parallel reviews (full entries, about 280 papers, in `literature_raw/lit_{A..F}_*.md`).
Each reviewer fetched arXiv pages; most judgments are **abstract-level**. Quoted numbers from secondary snippets are marked
UNVERIFIED in the raw files. Before citing any number in the paper, re-check it in the PDF.

## 0. CVPR 2027 facts (cvpr.thecvf.com, verified by reviewer E)

| Item | Value |
|---|---|
| Abstract registration | 2026-11-10 |
| Paper | 2026-11-16 AoE, no extensions |
| Supplementary | 2026-11-23 |
| Reviews / decisions | 2027-01-25 / 2027-02-25 |
| Topics | "Physical AI: World Models, VLA, Planning, Simulation"; "Embodied vision and robotics" |
| Page limit, LLM policy, reproducibility rules | 2027 pages returned 404 / "being finalized". CVPR 2026 proxy: 8 pages + refs, no external links, code encouraged |

Time left today: 26 days to registration, 32 to the paper.

## 1. Corrections to our notes

- **2607.25337 is Temporal-Distance JEPA** (Bai, Xiong), not the temporal-difference TD-JEPA (Bagatella et al., 2510.00739, ICLR 2026). `02`, `03`, `05` conflate them.
- "Anchored Planning" does not exist as a paper (closest: Hi-LeWM anchored waypoint). SPARK is a false match.
- WorldSync (2608.24885) is a robot **video** world model (WorldEcho benchmark); CoCo (2608.04653) is video prediction. Neither is a frozen-encoder planner.
- 2608.12959 is "Objective Is the Bottleneck" (TwoRoom only, no simulator oracle); it is not "ACPC" (2608.12939).
- ARC-Bench (2609.05461): abs page shows submission 12 Aug 2026, inconsistent with the ID; recheck before citing.
- JEPA-Bisim (2602.18639) was retitled in v2 (1 Oct 2026).
- `README`/`03` quote the goal-shift result as 88% to 6% in one place and 90% to 7% in another; the results file says 90.0% to 7.0%.

## 2. Map of the field (Oct 2026)

| Sub-area | Representative work | State |
|---|---|---|
| Aux losses that make LeWM/JEPA latent distance a better cost | SCALE 2608.16287, RC-aux 2605.07278, Temporal-Distance JEPA 2607.25337, CGS 2609.35603, Temporal Straightening 2603.12231, criterion-aligned 2610.01224, D-JEPA 2609.24749, AD-WM 2609.30264, DA-LeWM 2608.18746, AnisoWM 2609.37441, SpecWM 2610.15429 | **Saturated** (about a dozen in 3 months) |
| Learned temporal / reachability costs | TEMPO 2610.04988, TRM 2605.22164, TimeRewarder, quasimetric / Dual Goal Reps 2510.06714 | Crowded; small Push-T gains |
| Hierarchy / subgoals / amortised planners | HWM 2604.03208, H-JEPA 2610.06805, SAGE, GC-IDM 2605.08732, LEAP 2609.03294 | Saturated |
| Failure diagnostics / oracles | Planning Limits 2609.39235, Objective-Is-Bottleneck 2608.12959, TRM 2605.22164, IMWM 2606.01626, ARC-Bench 2609.05461, 2610.05550, 2608.10145 | Crowded (see §3) |
| Observation-side distractor robustness | JEPA-Bisim 2602.18639, CF-JEPA 2610.00727, Better Slots 2608.12078, ACPC 2608.12939 | Moderate |
| Data for world models | FIRM-WM 2609.22816, AD-WM, theory 2607.22430, 2610.19375, CureWM 2610.09134 | Thin but has strong competitors |
| Goal-side invariance (goal image nuisance) | Dual Goal Reps, GS-HER 2606.09476, Better Slots (goal colour on PushT), goal-cond. bisimulation 2204.13060 | **Appears unclaimed as a protocol on frozen-encoder planners** |
| Small-object progress with frozen features | SLIM 2610.03137 (end-to-end LeWM collapse), CostGrad, Better Slots | Partly covered (encoder side); cost-side open |

Core stack: DINO-WM 2411.04983; JEPA-WMs 2512.24497 (cost L_vis + 0.1 L_prop, CEM best; Push-T 70.2 vs 66.0 DINO-WM, Wall 78.8 vs 64.1,
Maze 83.9 vs 81.6; from the paper, partly read); V-JEPA 2 / 2.1; PLDM 2502.14819; LeWM 2603.19312 + stable-worldmodel 2605.21800.

## 3. Novelty verdict per prior finding

| Finding | Verdict | Closest prior | What stays ours |
|---|---|---|---|
| F1 cost is main bottleneck | **Slogan not new; study defensible** | 2608.12959 (cost swap, TwoRoom), 2605.22164 (true cost 7% to 100% TwoRoom), IMWM 2606.01626 (exact dynamics no gain, blames search; excludes Push-T), 2609.39235 (blames goal distance / horizon) | Factorial dynamics x cost x search on one checkpoint; Push-T 64.6 / 64.6 / 90.6. Needs a **search-budget axis** and reconciliation with IMWM / horizon claims |
| F2 task-equivalence violation | **Appears unclaimed**, not proven | Dual Goal Reps, GS-HER, Better Slots, SpecWM (possible concurrent), 2609.39235 (goal reset helps 7 tasks, hurts 5; unconfirmed) | Fixed task outcome, varied arm pose; L1-L4 on frozen-encoder planners; task-dependent equivalence |
| F3 relevance task-dependent | Unclaimed in this form | Denoised MDP taxonomy gives vocabulary (controllable but irrelevant) | The Push-T / drawer / button contrast |
| F4 small-object progress | Cause partly published | SLIM (encoder), CostGrad (token selection) | Exact-dynamics cost ablation; contact-rate analysis; closed-loop fix still missing |
| F5 pusher under-weighting | Not a target | alpha is the known fix | |
| F6 goal-appearance shift | **Overlap risk** | Better Slots perturbs goal colour/background on PushT for LeWM, DINO-WM (read its Fig. 3 before claiming) | Goal-only isolation, test-time fix (AdaJEPA lists as future work) |
| F7 data / counterfactual | Strong competitors | FIRM-WM (Cube 88 vs 70), AD-WM (Cube hard starts 3.7 to 52) | Our Cube null (2610.19375 reports action-swap error only 2.1x on Cube vs 29.1x PushT, snippet only) |
| F8 top-k | Closed | CostGrad (different purpose) | |
| F9 far-goal | Covered | Planning Limits | |

## 4. What CVPR rewards (reviewer E; ICLR 2026 reviews, not CVPR)

- JEPA-WMs itself was rejected at ICLR 2026 (6/6/4) and accepted at TMLR; reviewers: "practitioner's guide", no closed-loop robot, confounded encoder comparison, no failure analysis.
- Recurrent criticisms: only 3 DINO-WM tasks, small 2-D domains, missing MPPI / gradient planners / TD-MPC2, single-seed, no ablations.
- Accepted CVPR world-model papers are either large methods (NWM oral, 1B params), benchmarks with infrastructure (WorldLens, SimScale) or vision-first (GeoWorld). No analysis paper of this type at a top venue yet was found.
- Implied bar for us: 4-6 environments (2+ manipulation), 2-3 encoders, 2+ world-model families, CEM + MPPI/gradient, 3-5 seeds with CIs, DINO-WM / PLDM / LeWM / JEPA-WMs baselines, and one simple fix that works across models. Real robot is not required but its absence is a named weakness: scope claims accordingly.

## 5. Candidate directions

**D1 (recommended spine): a goal-side evaluation protocol plus a cost failure taxonomy, with one task-aware fix.**
Title direction: when the goal image says more than the task. Three parts:
1. Protocol: factorial oracle attribution (dynamics, cost, search budget) and task-equivalent goal families (L1-L4), run on several frozen-encoder planners.
2. Taxonomy of goal-cost failures measured under that protocol: nuisance in the goal (F2/F6), small-object progress (F4), exchange rate (F5), each with its own diagnostic.
3. A fix that works across models: learned goal-equivalence projection (M1 in `04`), against fixed-rule baselines (robot mask, patch subset, proprio alpha, temporal distance, controllability).
Why: fits "analysis + protocol + simple fix" from E; F1 gets its defensible form; F2/F3 are the unclaimed core. Risks: F2 evidence is thin (one task, p=0.031), SpecWM / Better Slots may be concurrent, 32 days.

**D2: frozen-encoder small-object cost study** (encoder x pooling x cost, with a contact-biased CEM fix). Open per D/F, but SLIM is the reviewer's obvious citation and a closed-loop fix on Metaworld push is unproven (all image costs floor at 0-8%).

**D3: data study** (decorrelation, coverage). Open in principle; FIRM-WM and AD-WM are strong and our Cube result is null. Not recommended for CVPR in 32 days.

## 6. First experiments if D1 is chosen (kill tests stated before running)

1. **Proprio confound check (code, hours):** do the equivalent-goal variants change the goal proprio? If yes, repeat drawer L4 with alpha=0 on variants. Kill F2-on-drawer if the effect vanishes.
2. **Read Better Slots Fig. 3 and 2609.39235 goal-reset result** (hours). If either already shows goal-only nuisance on PushT/Metaworld at our level of isolation, narrow the claim.
3. **Pin the protocol:** fixed episodes, seeds, success criterion; add search-budget axis to attribution (Push-T, Metaworld) to answer IMWM.
4. **Replicate on DINO-WM and LeWM / PLDM** (E1 in `04`), more arm variants (5-8) and 48 episodes on drawer; paired tests with Holm correction across variants.
5. **Add tasks without floor/ceiling** (E4): need 30-80% base success, plus the button counter-example.
6. **Fixed-rule baseline table** (E5) to show no fixed rule is right everywhere.
7. Only then train M1; kill if it does not restore drawer variants to base success while leaving Push-T and button unchanged.

Decision point: after items 1-2 (about 2 days) choose between D1 and D2.
