# Lit review C: failure analysis, diagnostics, benchmarks, planner/search design (verified 2026-10-15)

Verification key: VERIFIED-FULL = abs page fetched AND HTML body read for the oracle/protocol; VERIFIED-ABS = arXiv abs page fetched (title/authors/date/abstract confirmed), body not read; UNVERIFIED = only seen in search snippets.
Caveat: WebFetch summarises pages with a small model; numbers quoted from body reads should be re-checked against the PDF before they go into the paper.

## 0. Status of the IDs in our internal notes

| ID | Exists? | What it actually is | Note |
|---|---|---|---|
| 2609.39235 | YES | Planning Limits of Latent World Models (Alrasheed, ..., Akhtar) | VERIFIED-FULL (oracle section read). Has BOTH oracle-dynamics+oracle-cost and oracle-dynamics+state-distance. |
| 2608.12959 | YES | The Objective Is the Bottleneck (Joyjeet Singh) | VERIFIED-FULL. NOT called ACPC (name never appears). No oracle experiment at all; TwoRoom only. |
| 2609.05461 | YES | ARC-Bench (Zhang, Li) | VERIFIED-FULL. No oracle-planner ablation; fixed-candidate ranking audit. Note arXiv ID month (2609) vs "submitted 12 Aug 2026" mismatch on abs page; flag when citing. |
| 2608.16287 | YES | Reperesentation [sic] Geometry Matters for Planning with JEPA World Models (SCALE) (Hu, ..., Cui) | VERIFIED-ABS. v3 revised 6 Oct 2026. |
| 2609.03294 | YES | LEAP = Latent Energy Action Planning (Pham, Bera) | VERIFIED-ABS. |
| 2609.24749 | YES | D-JEPA: A Decision-Aligned Latent World Model (Liu, ..., Su) | VERIFIED-ABS. It is a training-time geometry refinement, not (per abstract) a test-time reranker. |
| 2609.35603 | YES | CGS = Control-Geometry Straightening (Fu, Ning) | VERIFIED-ABS. |
| 2610.05550 | YES | When Low Prediction Error Misleads Planning (Min, ..., Du) | VERIFIED-FULL (first 100k chars). |
| "TD-JEPA" | AMBIGUOUS | (a) Bagatella et al. 2510.00739 (ICLR 2026) is zero-shot-RL TD-JEPA, unrelated to planning cost; (b) Bai and Xiong 2607.25337 is "Temporal-Distance JEPA". Our notes probably mean (b). | Check which one the notes mean. |

## 1. Most important papers for novelty of F1 (read these first)

### 1.1 The Planning Limits of Latent World Models (VERIFIED-FULL)
- arXiv 2609.39235, Ali Alrasheed ... Naveed Akhtar, 30 Sep 2026, preprint. Code: none found.
- Method: action-conditioned WMs on 5 frozen backbones (V-JEPA 2, V-JEPA 2.1, VideoMAEv2, VideoPrism, DINOv2-L), CEM planner, cosine latent cost; Meta-World (16 tasks) + BridgeData V2.
- Oracle protocol (Sec 5.5): replace WM with real simulator (exact dynamics) AND score by true distance of hand/objects to target (oracle dynamics + oracle cost jointly, MPC, 5-step rollout). Success 92% -> 41% as target goes from 5 to 20 steps ahead; learned predictor gets 40% at 20 steps, i.e. the planning horizon, not model error, limits it. Appendix E: exact dynamics + state-distance cost on 8 tasks: 3.8% -> 46.3% at equal budget, 55.0% at full search; learned predictor attains ~29% / 39% of gains. Authors themselves note the gap mixes cost (cosine vs true) and prediction error.
- Overlap with F1: HIGH on the claim "dynamics are not the bottleneck", but they swap dynamics and cost together (no factorial). They attribute failure to horizon / goal distance (search/subgoal), not cost per se. Nearby expert subgoals help most.

### 1.2 Beyond Euclidean Proximity: Horizon-Matched Trajectory Reachability Metrics (VERIFIED-FULL)
- arXiv 2605.22164, Liangyu Li, Shengzhi Wang, Qingwen Liu (first Li, last Liu), May 2026 preprint. Code: "supplement", no URL.
- Method: post-hoc pairwise head (TRM) trained on temporal separations; replaces Euclidean terminal cost; LeWM / PLDM.
- Oracle protocol: ORACLE COST ONLY. Learned LeWM encoder+predictor kept; terminal cost = true simulator-state geodesic distance. TwoRoom high-distance, CEM 300x10x30, 100 eps x 3 seeds: raw latent MSE 7.0%, oracle geodesic 100.0%, TRM 97.0%, decoded-XY cost 94.2%; raw latent with 1000 samples/20 iters still 2.5% (so not search budget). PushT: task-state heads as auxiliary cost only modest (offset 50: 40.0 -> 52.7%).
- Overlap with F1: VERY HIGH for TwoRoom. This is the closest prior art to "swap true cost into frozen WM, keep learned dynamics". No oracle-dynamics counterpart; PushT not oracle-tested.

### 1.3 IMWM: Intuition Models Complement World Models for Latent Planning (VERIFIED-FULL)
- arXiv 2606.01626, Baoqi Gao ... Song Wang, 1 Jun 2026. Code: none (artifact release planned).
- Oracle protocol: ORACLE DYNAMICS ONLY (true env rollout replaces predictor; encoder, terminal latent-MSE cost, CEM 300/30/30 fixed). Two-Room: oracle 85.5% vs WM-only 87.7% vs IMWM 99.2%. OGBench-Cube: 76.8 / 66.2 / 94.7. Push-T and Reacher excluded (physics confound / baseline already good). Failure analysis: 98.9% (86/87) Two-Room and 91.4% (127/139) Cube failed episodes had zero goal-reaching candidates in the CEM population => attributed to SEARCH.
- Overlap with F1: This is the oracle-dynamics half of our result (exact dynamics gives no gain: matches our 64.6 -> 64.6). But it concludes SEARCH, whereas 1.2 concludes COST. They never swap in true cost. Their Push-T oracle cell is a side result with the baseline collapsing 94% -> 30% under their physics modification.

### 1.4 When Low Prediction Error Misleads Planning (VERIFIED-FULL)
- arXiv 2610.05550, Rui Min, Xianyao Li, Fang Xu, Jing Du, 4 Oct 2026. Code: supplementary bundle only.
- Protocol: per-start 300 shared CEM candidates; endpoint error split into candidate-pool "center" and action-relative "response"; oracle C (swap realized center) vs oracle S (swap realized responses); rank vs physical endpoint score (Spearman, top-30 elite quality). Models LeWM, PLDM, PreJEPA, DreamerV3, DINO-WM; tasks PushT, TwoRoom, Reacher, Cube. Center dominates MSE in 14/16 cells but response repair wins ranking in 6. Matched-candidate: encoding realized endpoints raises Spearman 0.464->0.975 (Reacher), 0.193->0.631 (PushT). Oracle-selected-action LeWM study favours center repair on PushT. 72-run objective study: closed-loop gains unconfirmed.
- Overlap: MEDIUM-HIGH. Oracle is on ranking quality of dynamics outputs, not full closed-loop swap of dynamics or cost. Directly relevant: PushT latent ranking is poor (rho 0.19) even given realized endpoints, i.e. consistent with cost/representation as the culprit on PushT.

### 1.5 ARC-Bench (VERIFIED-FULL)
- arXiv 2609.05461, Zhengshu Zhang, Zhiyuan Li. Code: promised, none.
- Fixed-candidate audit of official JEPA-WM checkpoints (PointMaze, Wall, Push-T, MetaWorld reach-wall, + U-Maze, FourRooms). Candidate GT cost = distance of executed terminal state to goal terminal state. Push-T official: top-1 wrong on 96.8% of anchors, Hit@1 3.2%, regret 0.243; MetaWorld 100% wrong; encoder swap (DINOv2, V-JEPA 1/2) still 91-97% wrong. Closed-loop replanning masks it: PointMaze 89.6 -> 61.8%, Push-T 36.1 -> 12.5% as replan interval 1 -> 6. Prediction error nearly uncorrelated with true cost (Spearman 0.21, 0.027).
- Overlap: HIGH on the symptom (cost ranking is broken); no oracle planner. Strong supporting evidence and a protocol we can reuse/compare (use their regret/Hit@1 metrics).

### 1.6 The Objective Is the Bottleneck (VERIFIED-FULL)
- arXiv 2608.12959, Joyjeet Singh (single author), 13 Aug 2026. Code: github.com/joyjeet-singh/tinylab.
- LeWM reproduction on TwoRoom, CEM 300/30/30. Rollout error at 75 steps is 0.189 of static baseline, yet published cost success at offset 100 is 26% (ours) / 14% (released). Latent vs true distance r=0.426 (inverts beyond ~120 units); position probe R2=0.992. Swapping ONLY cost: decoded-position cost 88%, learned temporal head 98% (ours). Released weights: 14 -> 70% / 34%.
- Oracle: none (no true-simulator oracle); uses a decoded-position proxy. TwoRoom only, no PushT.
- Overlap: HIGH for the thesis statement "objective, not predictor". Title is nearly our F1 slogan.

## 2. Oracle / attribution adjacent

### 2.1 VisualPatchWorld (VERIFIED-ABS, oracle detail UNVERIFIED)
arXiv 2607.25236, Jiaxin Bai, Jiaxuan Xiong, 28 Jul 2026, code github.com/HKBU-KnowComp/VisualPatchWorld. Code-world-model planner; abstract: contact-rich pushing retains a gap that engine re-checking of a shortlist mostly closes. Search snippet claimed PushT 22% -> 96% with "Oracle Hybrid" (simulator rescoring); the abs page did not describe it, so UNVERIFIED. Relevance: simulator rescoring (true cost + true dynamics on shortlist) as an upper bound on Push-T.

### 2.2 Decision-Metric Alignment in Latent World Models (DA-LeWM) (VERIFIED-ABS)
arXiv 2608.18746, Jiawei Wang ... Minglei Li, 19 Aug 2026. Defines Plan-Real Spearman and CEM-stage Spearman; DA-LeWM adds inverse-dynamics/goal-action heads. Snippet (UNVERIFIED) says PushT 49.3 -> 92.7 in a 1-epoch budget table. No oracle. Overlap: diagnostics for cost-ranking alignment.

### 2.3 Supervise What Decides Success (VERIFIED-ABS)
arXiv 2610.01224, Takumi Hara, Kanata Suzuki, 1 Oct 2026. In all 4 models, end-effector position encoded with more error than the success criterion tolerates; auxiliary linear head on criterion quantities: +3.5 pp PushT, +3.4 pp Cube. Supports "latent cost cannot resolve the success criterion" (precision, not just ranking).

### 2.4 Predictive but Not Plannable: RC-aux (VERIFIED-ABS)
arXiv 2605.07278, Wenyuan Li ... Miki Haseyama, 8 May 2026. Code github.com/Guang000/RC-aux. Multi-horizon + budget-conditioned reachability auxiliary on LeWM; reachability-aware planner at test time. Title is the same message as F1.

### 2.5 Temporal-Distance JEPA (VERIFIED-ABS)
arXiv 2607.25337, Jiaxin Bai, Jiaxuan Xiong, Jul 2026, code github.com/HKBU-KnowComp/Temporal-Distance-JEPA. Directed temporal cost, TwoRoom 100.0 vs LeWM 97.4, Cube +14.2, Push-T improved (no number in abstract).

### 2.6 Traj-LeWM (VERIFIED-ABS)
arXiv 2608.14125, Xiaodi Huang ... Tao Huang, 14 Aug 2026. Trajectory-level latent cost added to endpoint cost; gains over LeWM: +3 PushT, +14 Cube, +7 Reacher, +7 TwoRoom. Same dynamics/budget/CEM. Cost-side fix, small on PushT.

### 2.7 Anisotropic Representations Improve Planning (AnisoWM) (VERIFIED-ABS)
arXiv 2609.37441, Mingu Kang ... Joonseok Lee, 29 Sep 2026. Learnable diagonal covariance in SIGReg-style target; latent cost agrees better with task outcomes; 4 envs. Page: rkdrn79.github.io/AnisoWM-page.

### 2.8 Toward Physically Grounded JEPA WMs for Goal-Conditioned Planning (VERIFIED-ABS)
arXiv 2609.03565, Muyuan Liu ... Xiang Gao, 3 Sep 2026. Inverse dynamics + state alignment; TwoRoom 100, PushT 98, Cube 87. Privileged-state alignment, same family as SCALE.

### 2.9 Order-Consistent Latent Geometry (VERIFIED-ABS)
arXiv 2610.18570, Junlin Wu, Wen Fan, Dandan Zhang, 13 Oct 2026. Theory: endpoint-Euclidean ranking only needs self-approaching condition; acute bends in latent path cause mis-ranking even with exact rollouts. Useful theory for why cost fails even with exact dynamics.

### 2.10 ATM and AC-consistency diagnostics
- ATM: Action-Consistency Transfer Matrix, arXiv 2606.09028, Jiaheng Chen, Jun 2026. UNVERIFIED (search snippet only): probe-based diagnosis, >100x faster than CEM eval, PushT/TwoRoom/Cube.
- Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency, arXiv 2608.12939, Guo An ... Qi Tian, 13 Aug 2026. VERIFIED-ABS. Invariance Radius and Separation Rate; no code, no numbers in abstract.

## 3. Representation / geometry papers (cost landscape)

- Temporal Straightening for Latent Planning, arXiv 2603.12231, Ying Wang ... Mengye Ren, ICML 2026 camera-ready; code agenticlearning.ai/temporal-straightening. VERIFIED-ABS. Curvature regularizer; Euclidean latent distance becomes better geodesic proxy; gradient planning stabilised.
- SCALE (2608.16287), VERIFIED-ABS: adds term correlating sampled pairwise latent distances with task-state distances (privileged state in training only) to LeWM; improves success over LeWM across solvers. Closest "fix the cost geometry via state" method.
- CGS (2609.35603), VERIFIED-ABS: aligns pairwise cosine similarity of actions with that of latent differences; guarantees for MPPI/CEM/GD under linear dynamics; up to +20 pp over LeWM at 128 candidates, +12.6 over LeWM+TS; 4 envs; includes DINO-WM comparison and planner ablations.
- Value-Guided Action Planning with JEPA World Models, arXiv 2601.00844, Matthieu Destrade ... Yann LeCun, World Modeling Workshop 2026. VERIFIED-ABS. Shape embedding so that negative goal-conditioned value ~ embedding distance.
- D-JEPA (2609.24749), VERIFIED-ABS: "decision-local prediction gap"; refines predictive geometry from executed outcomes; PushT 87.89%, RoboTwin +15.04, real robot +17. Not clearly a reranker at test time.
- TD-JEPA (Bagatella ... Tirinzoni) arXiv 2510.00739, ICLR 2026. VERIFIED via search only (not abs fetched): zero-shot RL representations, tangential to planning-cost diagnosis.
- Latent World Models with Monotone Planning Costs for Image-Goal Navigation, arXiv 2608.09073, Amirhosein Chahe ... Lifeng Zhou, Aug 2026. VERIFIED-ABS. Monotone Cost Ranking loss; GNM navigation.
- Beyond the Next Step: Variable-Length Latent World Models (VLWM), arXiv 2606.21775, Tianqi Du ... Yisen Wang, Jun 2026. VERIFIED-ABS. +13% avg over LeWM.

## 4. Planner / search design

- GRASP: Parallel Stochastic Gradient-Based Planning for World Models, arXiv 2602.00475, Michael Psenka ... Amir Bar, Jan 2026. VERIFIED-ABS. Lifted states, noise, stop-gradient through state input; beats CEM / GD on long horizon. Snippet (UNVERIFIED) Push-T H=50 43.4%.
- Latent Geometry Beyond Search: Amortizing Planning in World Models, arXiv 2605.08732, Hoang Nguyen, Xiaohao Xu, Xiaonan Huang, May 2026, code github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models. VERIFIED-ABS. GC-IDM on frozen LeWM matches/exceeds CEM in 7/8 settings, 100-130x cheaper; compares CEM/MPPI/iCEM/gradient. Snippet: 12 CEM configs; Push-T best CEM 90% vs GC-IDM 84.2% at 35x cost.
- LEAP (2609.03294), VERIFIED-ABS: quasi-Newton on full horizon via frozen LeWM, terminal-window energy, goal-conditioned proposal init; 4 domains 77.5 -> 94.8% mean. Snippet: Push-T 90.0 vs 88.0 controlled LeWM+CEM (UNVERIFIED).
- Reinforced Planning with Latent World Models (RP1), arXiv 2608.18669, Armin Sommer, Jannik Schilling, Aug 2026. VERIFIED-ABS. Learned critic + plan-improvement operator, 1000x fewer rollouts than CEM.
- LeFlow, arXiv 2608.24855, Hsiang-Wei Huang ... Jenq-Neng Hwang, Aug 2026, code github.com/hsiangwei0903/LeFlow (from search; not fetched on abs). Search-verified only; rectified-flow latent path + inverse dynamics, frozen WM verifies. Snippet: PushT 86% vs 84% with DINO-WM.
- Hierarchical Planning with Latent World Models (HWM), arXiv 2604.03208, Wancong Zhang ... Nicolas Ballas, v3 12 Oct 2026. VERIFIED-ABS. Real Franka pick-and-place 70% vs 0% single level; up to 3x less planning compute. Project: kevinghst.github.io/HWM.
- What Drives Success in Physical Planning with JEPA-WMs, arXiv 2512.24497, Basile Terver ... Yann LeCun, TMLR (v4 2 Sep 2026), code github.com/facebookresearch/jepa-wms. VERIFIED-ABS. Compares CEM/NeverGrad/gradient planners, L1 vs L2 cost (L2 consistently better; snippet-level), architecture; no oracle.
- LeWorldModel, arXiv 2603.19312, Lucas Maes ... Randall Balestriero, 13 Mar 2026. VERIFIED-ABS. ~15M params, SIGReg, plans 48x faster than foundation-model WMs. Backbone for most 2026 fixes.
- DINO-WM, arXiv 2411.04983, Gaoyue Zhou ... Lerrel Pinto, 2024 (venue not on abs page; ICML 2025 per my memory, UNVERIFIED). VERIFIED-ABS. PushT 90% in its own setup (snippet).

## 5. Benchmarks / evaluation
- ARC-Bench (above).
- Singh, "The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWM on TwoRoom", arXiv 2608.10145, 10 Aug 2026, code tinylab. VERIFIED-ABS. Goal construction alone moves released checkpoint 84% -> 8%; one-step error does not predict long-horizon success. Warns that our Push-T numbers are protocol-sensitive.
- WorldArena (arXiv 2602.08971; 2.0 = 2605.17912): VERIFIED via search only (not abs-fetched); video-quality + functional utility (planner, policy evaluator) for generative embodied WMs on RoboTwin; weakly relevant to latent-cost diagnostics.
- How Should World Models Be Evaluated for Embodied Decision-Making?, arXiv 2606.15032, Yang Yu ... Haoxin Lin, Jun 2026. VERIFIED-ABS. Position paper: L0-L7 ladder, metrics such as policy-ranking agreement, model exploitability.

## 6. Classical MBRL objective mismatch (all VERIFIED via search, abs pages not individually fetched)
- Objective Mismatch in Model-based RL, arXiv 2002.04523, Nathan Lambert ... Roberto Calandra, L4DC 2020. One-step likelihood not correlated with control performance.
- A Unified View on Solving Objective Mismatch in MBRL, arXiv 2310.06253, Ran Wei ... Roberto Calandra, TMLR 2024. Taxonomy: distribution correction, control-as-inference, value equivalence, differentiable planning. (This is the "Wei et al." in the brief; it is a survey, not an oracle-ablation paper.)
- The Value Equivalence Principle for MBRL, arXiv 2011.03506, Christopher Grimm ... Satinder Singh, NeurIPS 2020.
- Benchmarking Model-Based RL, arXiv 1907.02057, Tingwu Wang ... Jimmy Ba, 2019. Names "dynamics bottleneck" and "planning horizon dilemma" as challenges.

## 7. How recent papers quantify cost vs dynamics vs search (summary matrix)

| Paper | Oracle dynamics | Oracle cost | Search budget sweep | Verdict |
|---|---|---|---|---|
| 2609.39235 | yes (joint) | yes (joint) | yes (full search) | horizon / goal distance, not WM |
| 2605.22164 | no | yes (TwoRoom, 100%) | yes (1000 samples, still 2.5%) | cost |
| 2606.01626 | yes (Two-Room, Cube) | no | CEM population analysis | search |
| 2610.05550 | partial (oracle-corrected predictions) | partial (realized-endpoint cost) | no | dynamics error magnitude misleads |
| 2608.12959 | no | proxy (decoded pos) | budget 50 vs 150 | cost/objective |
| 2609.05461 | no | no (audits GT cost) | replan interval | cost ranking |
| ours (F1) | yes (Push-T 64.6 -> 64.6) | yes (Push-T -> 90.6) | (check) | cost, on manipulation |

## 8. Honest novelty assessment of F1

- F1 as a qualitative statement ("cost, not the world model, is the bottleneck") is NOT novel. 2608.12959 (title), 2605.22164 (oracle-cost 7% -> 100%), 2609.05461, 2610.05550, 2609.37441/2608.16287/2610.01224/2608.18746 and the whole reachability/temporal-distance family (RC-aux, TD-JEPA, TRM, Traj-LeWM) all make it, mostly on TwoRoom/navigation with LeWM.
- Partly contested: IMWM (2606.01626) and Planning Limits (2609.39235) argue search/horizon is the bottleneck. A fair paper must reconcile them. Our own search-budget evidence is needed.
- What remains novel and defensible: (i) a clean FACTORIAL oracle attribution (dynamics x cost x search) with every cell measured on the same model and tasks; prior work swaps at most one factor, or both jointly (2609.39235), and 2605.22164 / 2606.01626 each cover one half; (ii) manipulation (Push-T) with DINO-WM / JEPA-WM, where prior oracle evidence is thin: 2605.22164 PushT only auxiliary heads, IMWM excluded Push-T; (iii) the exact-dynamics-gives-zero-gain on Push-T (64.6 -> 64.6) paired with true-cost 90.6 on the SAME checkpoint; (iv) a cross-benchmark/ cross-encoder pattern showing where the attribution flips (nav vs manip).
- Do not claim: first to show cost bottleneck; first oracle ablation; "dynamics are fine" (IMWM, 2609.39235 already show). Position as a controlled attribution study that unifies and adjudicates conflicting 2026 claims (cost vs search vs horizon), with protocol sensitivity (2608.10145) controlled.
- Risks: oracle-cost uses privileged state so it upper-bounds what any learned cost (SCALE, TRM, D-JEPA) can reach; also check Push-T oracle numbers against 2609.39235 Appendix E style (matched budgets) and report a search-budget axis, since a reviewer will cite IMWM.

## 9. Gaps / not done
- Bodies of 2609.39235 (last ~38k chars), 2610.05550, 2606.01626 (>100k chars) not read; 2608.16287, 2609.35603, 2609.03294, 2609.24749 only abstracts read (no oracle ablation inspected). Recommend reading the SCALE and D-JEPA experiments for baselines (Push-T numbers) before the writing phase.
- No separate paper found titled "Wei et al. planning with imperfect models" other than 2310.06253; no classical paper found that runs an oracle-model-vs-oracle-reward ablation exactly like ours.
