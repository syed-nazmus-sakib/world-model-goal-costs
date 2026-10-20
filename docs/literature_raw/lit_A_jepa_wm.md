# Lit review A: JEPA / latent / DINO-feature world models for planning (2023 - Oct 2026)
Compiled 2026-10-15. VERIFIED = arXiv abs page (or search result naming that arXiv ID, title, authors) retrieved this session. Numbers are author-reported from abstract/page unless stated; items marked [abs-only] have not had their full tables read.

## 0. Verification of internal-note IDs
| Internal note | Result |
|---|---|
| 2607.25337 "TD-JEPA" | EXISTS but is **Temporal-Distance JEPA** (Bai & Xiong, Jul 2026), NOT Temporal-Difference JEPA. The "TD-JEPA" of Bagatella et al. is a different paper, 2510.00739 (ICLR 2026). Avoid name collision in citations. |
| 2609.39235 | EXISTS: "The Planning Limits of Latent World Models" (Alrasheed, Azam, Akhtar). |
| 2609.35603 "CGS" | EXISTS: "Control-Geometry Straightening for Sampling-Based Latent Planning" (Ziang Fu, Ning Ning), v2 7 Oct 2026. |
| 2610.00727 "CF-JEPA" | EXISTS: Byrd, Wright, Ha, 30 Sep 2026. |
| 2602.18639 "JEPA-Bisim" | EXISTS; v1 title was "Learning Invariant Visual Representations for Planning with Joint-Embedding Predictive World Models", v2 (1 Oct 2026) retitled "JEPA-Bisim: Learning Robust Visual Representations for Planning with ...". Cite the v2 title but mention v1. |
None of the five is fabricated.

## 1. Foundational / core (the stack we build on)

### 1. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning — VERIFIED
Gaoyue Zhou, ..., Lerrel Pinto (authors: Zhou, Pan, LeCun, Pinto). arXiv 2411.04983; ICML 2025. Predicts future frozen DINOv2 patch features (ViT predictor, action-conditioned, optional proprio) from offline trajectories; no reward/decoder. Cost: distance between predicted final latent and encoded goal image (MSE on patch features); planner CEM (GD also tried, CEM better). Benchmarks: PointMaze, PushT, Wall, Rope, Granular, (6 envs). Code: github.com/gaoyuezhou/dino_wm. Relevance: our base model; the goal-embedding L2 cost is exactly what we critique. 

### 2. What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? (JEPA-WMs) — VERIFIED
Basile Terver, ..., Yann LeCun (Terver, Yang, Ponce, Bardes, LeCun). arXiv 2512.24497 (30 Dec 2025), preprint, no venue listed. Code/weights: github.com/facebookresearch/jepa-wms.
- Cost: L(o_t, a, o_g) = (L_vis + alpha*L_prop)(G(o_t,a), E(o_g)); G = predictor unrolled with planning context W_p=2; L1 or L2 (L2 consistently better); alpha=0.1 sim, 0 for DROID/Robocasa. Goal = single goal-image embedding (goal frames from expert policy/dataset/random state sampler). Frozen patch-level encoder features; proprio via shallow encoder.
- Planners: CEM (best overall w/ L2), NGOpt (Nevergrad; ties CEM on DROID/Robocasa, weaker on 2D nav), GD/Adam (good on Metaworld, poor on nav/contact due to local minima).
- Encoders: DINOv2/v3, V-JEPA/V-JEPA-2; DINO > V-JEPA; DINOv3 > v2 on Robocasa/DROID. Scaling encoder/predictor does not help sim, helps DROID.
- Table 1 (success %, DINO-WM / V-JEPA-2-AC / proposed): Maze 81.6/-/83.9; Wall 64.1/-/78.8; Push-T 66.0/-/70.2; MW-Reach 44.8/-/58.2; MW-Reach-Wall 35.1/-/41.6; Robocasa Reach 19.1/16.2/25.4; Robocasa Place 21.7/33.1/30.7; DROID action score 39.4/42.9/48.2.
- Relevance: the paper explicitly studies planner/architecture/encoder choices; it does NOT (from the 100k chars read) study goal-image content or cost-function alternatives beyond L1/L2/cosine. Our "goal cost is the failure source + irrelevant factors in goal image" angle is not covered here. Note: only the first ~100k of 176k chars were read.

### 3. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning — VERIFIED
Mido Assran, ..., (30 authors, FAIR/Mila). arXiv 2506.09985, Jun 2025. V-JEPA 2-AC: action-conditioned predictor post-trained on <62 h DROID; zero-shot Franka pick-and-place in 2 labs from image goals; cost = L1 distance between predicted and goal-image embedding (per V-JEPA 2 paper; not re-checked here), planned with CEM. SSv2 77.3, EK100 39.7 R@5. Code: github.com/facebookresearch/vjepa2. Relevance: original image-goal L1 latent cost on real robot; baseline in JEPA-WMs.

### 4. V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning — VERIFIED
Lorenzo Mur-Labadia, ..., Adrien Bardes. arXiv 2603.14482 (Mar 2026). Dense predictive loss on visible+masked tokens, deep self-supervision; reports +20-point real-robot grasping success over V-JEPA-2-AC. Relevance: stronger encoder for goal-image features; "Planning Limits" finds it the most consistent backbone.

### 5. PLDM: Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models — VERIFIED
Vlad Sobal, ..., Yann LeCun. arXiv 2502.14819 (v4 Oct 2025; NeurIPS 2025 per search source). End-to-end JEPA latent dynamics (VICReg-style regularisers) + MPC; goal-image latent distance; compared with GCIQL, HIQL, CRL, etc. on 23 datasets across navigation envs (Two-Room, Diverse maze, etc.); best on generalisation to unseen layouts, low data, stitching. Code: github.com/vladisai/PLDM. Relevance: baseline; plans with latent distance to goal.

### 6. LeWorldModel (LeWM): Stable End-to-End JEPA from Pixels — VERIFIED
Lucas Maes, ..., Randall Balestriero (Maes, Le Lidec, Scieur, LeCun, Balestriero). arXiv 2603.19312 (v3 Jun 2026), preprint. Two losses: next-embedding prediction + SIGReg (isotropic Gaussian regulariser); ~15M params, 1 GPU, plans up to 48x faster than foundation-model WMs. Cost: terminal latent L2 to goal-image embedding with CEM/MPPI. Benchmarks: Two-Room, PushT, Reacher, OGBench-Cube. In stable-worldmodel paper reproductions (secondary): PushT LeWM 94, DINO-WM 92, PLDM 78, GCBC 75, TD-MPC2 12; OGB-Cube DINO-WM 86, LeWM 72, PLDM 62, GCBC 84. Code: github.com/lucas-maes/le-wm. Relevance: dominant 2026 JEPA-WM baseline; nearly all 2026 follow-ups patch LeWM.

### 7. stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation — VERIFIED
Lucas Maes, ..., Randall Balestriero (12 authors). arXiv 2605.21800 (May 2026). (Earlier "stable-worldmodel-v1" is arXiv 2602.08968, listed by search only, UNVERIFIED by fetch.) Lance data layer, LeWM/DINO-WM baselines, CEM/MPPI/GD solvers, envs with controllable visual/geometric/physical factors of variation (useful for our task-irrelevant-factor study). Code: github.com/galilai-group/stable-worldmodel.

## 2. Cost / objective / latent geometry for planning (closest to our thesis)

### 8. The Planning Limits of Latent World Models — VERIFIED
Alrasheed, Azam, Akhtar (full names not retrieved). arXiv 2609.39235. Action-conditioned predictors on five frozen SSL backbones, Meta-World + BridgeData V2. Ranking is reliable only for goals 5-10 steps ahead; task goals 16-53 steps; 81x larger predictor and longer-rollout training do not extend range; even with true simulator, success 92% -> 41% as target goes 5 -> 20 steps. Pure imagination 23%, MPC 30%, imagine-as-far-as-goal 47%, nearby expert subgoals 76%. V-JEPA 2.1 most consistent. Improves a VLA 65 -> 77% via best-of-8 selection. Relevance: HIGH. Supports "goal cost/horizon, not dynamics" with frozen encoders; threat to novelty of that claim (they attribute to horizon, we attribute to goal-cost content). 

### 9. Representation Geometry Matters for Planning with JEPA World Models (SCALE) — VERIFIED
Jiaming Hu, ..., Qingjun Cui. arXiv 2608.16287 (v3 6 Oct 2026). Argues decodable-but-barely-distance-affecting task state breaks the L2 cost; SCALE correlates pairwise latent distances with task-state distances (privileged state in training only). Added to LeWM. Per search snippet: Push-T latent-to-state distance alignment DINO-WM .70, LeWM .13, SCALE .77 (secondary; UNVERIFIED in full text). Relevance: HIGH; same diagnosis (cost misrepresents closeness), fix is training-time.

### 10. How Long, Not How Close: A Learned Temporal Metric for Planning in Latent World Models (TEMPO) — VERIFIED
Lama Moukheiber, Haotian Xue, Yongxin Chen. arXiv 2610.04988 (v2 7 Oct 2026). Learns small MLP map of frozen latent space where distance ~ step count; blends into planner cost; works on LeWM and PLDM; 11 sim envs. TwoRoom goal 3 plans away: 36% -> 99%. Push-T gain small (22->28%, secondary). Relevance: HIGH threat; learned goal cost on frozen latents, no retraining.

### 11. World Model Control by Trajectory Reachability Metrics (TRM) — VERIFIED
Liangyu Li, ..., Qingwen Liu. arXiv 2605.22164 (v2 31 Aug 2026). Small temporal pairwise cost ranks predicted endpoints vs goal; TwoRoom LeWM 7.0 -> 96.7%, PLDM 32.7 -> 84.0%; reweights few latent directions (XY rowspace <1% of terminal MSE but carries decision info); fails on contact-rich PushT and unseen wall orientations. Relevance: HIGH; shows Euclidean goal cost underweights control-relevant directions.

### 12. Temporal-Distance JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control (the note's "TD-JEPA") — VERIFIED
Jiaxin Bai, Jiaxuan Xiong. arXiv 2607.25337 (Jul 2026). Directed temporal-cost head from same-trajectory ordering, cross-trajectory negatives, rollout-consistency term on LeWM backbone. Two-Room 100% vs 97.4% LeWM; OGB-Cube +14.2 pts shared Euclidean planning; Push-T 86.0 vs LeWM 83.6 (secondary); learned cost used only on Two-Room/Reacher. Code: github.com/HKBU-KnowComp/Temporal-Distance-JEPA.

### 13. Predictive but Not Plannable: RC-aux for Latent World Models — VERIFIED
Wenyuan Li, ..., Miki Haseyama. arXiv 2605.07278 (May 2026). Multi-horizon open-loop training + budget-conditioned reachability head with temporal hard negatives, optional reachability-guided planner; on LeWM. Secondary: TwoRoom 88.8->98.0, Reacher 82.8->87.2, Wall 50.4->83.6. Also LIBERO-Goal. Code: github.com/Guang000/RC-aux.

### 14. Control-Geometry Straightening for Sampling-Based Latent Planning (CGS) — VERIFIED
Ziang Fu, Ning Ning. arXiv 2609.35603 (v2 7 Oct 2026). Aux loss matching pairwise cosine of actions to those of latent differences; theory for MPPI/CEM/GD; 4 envs; up to +20 pts vs LeWM and +12.6 vs LeWM+TS at 128 samples.

### 15. Temporal Straightening for Latent Planning — VERIFIED
Ying Wang, ..., Mengye Ren (Wang, Bounou, Zhou, Balestriero, Rudner, LeCun, Ren). arXiv 2603.12231, ICML 2026 camera-ready comment. Curvature regulariser makes Euclidean distance a better geodesic proxy; helps GD planning. Abstract notes pretrained encoders carry distracting information (directly relevant to us).

### 16. Supervise What Decides Success: Criterion-Aligned Auxiliary Losses for Latent World-Model Planning — VERIFIED
Takumi Hara, Kanata Suzuki. arXiv 2610.01224 (1 Oct 2026, under review). In four latent WMs end-effector position is encoded with error larger than the success tolerance; auxiliary regression of success-criterion quantities; PushT +3.5, Cube +3.4 pts (significant). Relevance: HIGH; supports "latent cannot separate successful candidates".

### 17. D-JEPA: A Decision-Aligned Latent World Model — VERIFIED
Shuaijun Liu, ..., Ningxin Su. arXiv 2609.24749 (Sep 2026). "Decision-local prediction gap": closest-to-goal prediction need not succeed; refines predictor using executed outcomes. PushT 87.89%, +15.04 on RoboTwin, +17 real robot.

### 18. Keeping JEPA World Models Plannable When Little of the Frame Moves (SLIM) — VERIFIED
Florian Strohm, ..., Marco Huber. arXiv 2610.03137 (2 Oct 2026). Small-object pushing benchmark with paired visual and language goals; vanilla LeWM 0.3% on SLIM; inverse-dynamics aux 35%; language-goal head plans from sentences (nav 0.84 vs visual-goal oracle 1.00). Relevance: goal-modality comparison (visual vs language goal), partial overlap with goal-spec question.

### 19. Spectral-Target Physical Latent Structuring for JEPA-Style World Models — VERIFIED
Penghao Zhu, ..., Aneesh Jonelagadda. arXiv 2609.04264 (v3 16 Sep 2026). "Physical representation laziness"; Fourier aux head. [abs-only, no numbers.]

### 20. Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning — VERIFIED
Muyuan Liu, ..., Xiang Gao. arXiv 2609.03565; IROS 2026 PWMS workshop. IDM + state-alignment on LeWM; TwoRoom 100, PushT 98, OGB-Cube 87.

### 21. Beyond Euclidean-style: Latent World Models with Monotone Planning Costs for Image-Goal Navigation — VERIFIED
Amirhosein Chahe, Siwei Cai, Lifeng Zhou. arXiv 2608.09073 (Aug 2026). Cosine goal cost on frozen DINO latents; Monotone Cost Ranking loss + AR rollout loss; GNM data; beats NWM, DINO-WM, OmniVLA, NoMaD; 2.7x lower orientation error vs same-encoder DINO-WM.

### 22. No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models (AC-MTM) — VERIFIED
Jack Boylan, Chris Hokamp. arXiv 2608.17542. Replaces SIGReg with Action-NCE inverse head; OGBench Visual Scene 80.0 vs 58.0 SIGReg. Code linked.

### 23. Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models — VERIFIED
Kai Zhao, ..., Dan Zeng. arXiv 2605.09241. Code github.com/intcomp/Sub-JEPA. Anti-collapse variant; no numbers read.

## 3. Robustness to irrelevant visual factors (directly our topic)

### 24. JEPA-Bisim — VERIFIED (see section 0)
Leonardo F. Toso, ..., James Anderson. arXiv 2602.18639. Bisimulation encoder in DINO-WM-style objective; PointMaze and PushT with background changes and distractors; latent up to 10x smaller than DINO-WM; works with DINOv2, SimDINOv2, iBOT. Relevance: HIGH threat on the "irrelevant factors hurt latent planning" claim, but addresses encoder/dynamics side, not goal-image cost specifically.

### 25. CF-JEPA: Improving Robustness of JEPA World Models via Controllability Factorization — VERIFIED
Morgan Byrd, Robert Wright, Sehoon Ha. arXiv 2610.00727 (30 Sep 2026). Controllable/uncontrollable latent split; distractor info routed to uncontrollable part; only model without latent collapse under distractors; 2D/3D control + sim robot task. Relevance: HIGH.

### 26. LRC-JEPA: Disentangling Dynamics and Residual Context for Efficient World Models — VERIFIED
Luzhe Huang, ..., Yuhuan Zhao. arXiv 2609.34375. Compact predictive latent z for planning plus residual context u; +9 pts over parameter-matched JEPA on 4 sims; 5.5M encoder beats DINO-WM (22.1M) and V-JEPA2 (303.9M) encoders on Bridge-v2.

### 27. Beyond Task Success: Stage-Wise Reliability of World Model Planning under Sensing Degradation — VERIFIED
Geonmyeong Lee, Byoung-Tak Zhang. arXiv 2609.07126. 10 visual/temporal degradations traced across representation, prediction, planner preference, outcome; large representation shift can fade downstream. Relevance: HIGH methodology precedent for our "cost/plan/success" analysis; degrades observations, not goal-image factors specifically (UNVERIFIED whether goal is perturbed).

### 28. VJEPA: Variational JEPAs as Probabilistic World Models — VERIFIED
Yongchao Huang (single author). arXiv 2601.14354. Claims filtering of high-variance distractors; theoretical.

## 4. Long horizon / hierarchy / planner efficiency

### 29. Hierarchical Planning with Latent World Models (HWM) — VERIFIED
Wancong Zhang, ..., Nicolas Ballas (10 authors incl. Terver, Assran, Bar, LeCun). arXiv 2604.03208; CoRL 2026. Multi-timescale WMs in shared latent space; long-horizon predictions are subgoals via latent matching; macro-action encoder. Real Franka pick-and-place single goal image: 70% vs 0% flat; up to 3x less planning compute; improves V-JEPA2-AC, DINO-WM, PLDM. Project page kevinghst.github.io/HWM.

### 30. H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning — VERIFIED
Wancong Zhang, ..., Randall Balestriero. arXiv 2610.06805 (Oct 2026). Stack of action-conditioned JEPAs; top-down subgoals. Visual AntMaze 18% -> 73%.

### 31. FF-JEPA: Long-Horizon Planning in World Models with Latent Planners — VERIFIED
Sergi Masip, ..., Tinne Tuytelaars. arXiv 2606.09311. Action-free latent subgoal planner; removes need for goal image; preliminary PushT. Relevance: goal-image-free alternative.

### 32. SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning — VERIFIED
Letian Cheng, Qi Zhang, Yisen Wang. arXiv 2607.17973. Offset 150: PushT 4.7 -> 64.7, OGB Cube 20.7 -> 67.3, LIBERO Scene2 0 -> 48.7.

### 33. Beyond the Next Step: Variable-Length Latent World Models for Long-Horizon Planning — VERIFIED
Tianqi Du, ..., Yisen Wang. arXiv 2606.21775. +13% avg over LeWM.

### 34. Parallel Stochastic Gradient-Based Planning for World Models (GRASP) — VERIFIED
Michael Psenka, ..., Amir Bar. arXiv 2602.00475 (Jan 2026). Virtual states, noise, action-only gradients; DINO-WM; Push-T 80-step 10.4% vs CEM 2.8%, GD 6.4% (secondary). Code github.com/michael-psenka/grasp.

### 35. Closing the Train-Test Gap in World Models for Gradient-Based Planning — VERIFIED
Rohun Agrawal, ..., Micah Goldblum. arXiv 2512.09929 (v2 1 Oct 2026). Train-time data synthesis; matches/exceeds CEM at ~10% time.

### 36. Latent Geometry Beyond Search: Amortizing Planning in World Models — VERIFIED
Hoang Nguyen, Xiaohao Xu, Xiaonan Huang. arXiv 2605.08732. GC-IDM replaces search on LeWM; matches/exceeds CEM in 7/8 settings, 100-130x cheaper.

### 37. LeFlow: Generative Latent Flow Planning for World Models — VERIFIED
Hsiang-Wei Huang, ..., Jenq-Neng Hwang. arXiv 2608.24855. Rectified-flow latent path + inverse dynamics; code released.

### 38. Representation World Model — VERIFIED
Yijun Yuan, ..., Hang Zhao. arXiv 2609.29171. Plans via inverse dynamics along latent path, no rollouts or search. [abs-only]

## 5. Other latent / geometric / navigation WMs and contrast methods

### 39. GeoWorld: Geometric World Models — VERIFIED
Zeyu Zhang, Danning Li, Ian Reid, Richard Hartley. arXiv 2602.23058; CVPR 2026. Hyperbolic JEPA + Geometric RL, CEM energy planning; CrossTask and COIN; ~+3% SR 3-step, ~+2% 4-step over V-JEPA 2 (small). Note: procedural video planning, not control benchmarks.

### 40. ProWorld: Progress-Aware Hyperbolic World Models for Long-Horizon Visual Goal Reaching — VERIFIED
Zihan Liu, ..., Menglin Yang. arXiv 2608.01926. Progress-order hyperbolic learning, planning objective of goal proximity + sustained progress; +9.67 avg over LeWM on 4 tasks; code after acceptance.

### 41. Navigation World Models (NWM) — VERIFIED
Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun. arXiv 2412.03572; CVPR 2025. CDiT video diffusion WM (1B), plans by simulating candidate trajectories and scoring against goal (LPIPS/DreamSim-type score; not re-verified); weights HF facebook/nwm (CC-BY-NC). Pixel-space contrast to latent WMs.

### 42. Predicting Consequences and Reinforcing Navigation Policies with Latent World Models — VERIFIED
Zengmao Wang, Wei Gao, Shuhan Shen. arXiv 2608.26190; ECCV 2026 spotlight. Latent compatibility model; RL in WM.

### 43. DUET-DINO: Simultaneous Cross-View World Modeling for Latent Planning in Robot Manipulation — VERIFIED
Nisarga Nilavadi, ..., Wolfram Burgard. arXiv 2609.10506. Static+wrist DINOv3 latent WM on DROID/RoboArena, 7-DoF planning; Reach 92, Angled-reach 72.5, Grasp-and-lift 60.0. Finds DINOv3 captures action-conditioned change better than V-JEPA 2 wrist.

### 44. TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning (Temporal-Difference) — VERIFIED (via search results naming ID/venue; abs page not fetched)
Marco Bagatella, ..., Andrea Tirinzoni. arXiv 2510.00739; ICLR 2026 (oral per listing). TD-learned latent predictor + task encoders, zero-shot reward optimisation; ExoRL and OGBench, 13 datasets, strongest from pixels. Not an image-goal CEM planner (policy-based, successor-feature style). Contrast only.

### 45. TD-MPC2: Scalable, Robust World Models for Continuous Control — VERIFIED (search)
Nicklas Hansen, Hao Su, Xiaolong Wang. arXiv 2310.16828; ICLR 2024. Decoder-free latent WM, MPPI planning with learned value/reward; 104 tasks, 317M agent on 80 tasks. Contrast: reward/value-based cost vs our goal-image cost. Appears as a weak baseline (PushT 12, OGB-Cube 4) in stable-worldmodel.

### 46. Dreamer 4: Training Agents Inside of Scalable World Models — VERIFIED (search)
Danijar Hafner, Wilson Yan, Timothy Lillicrap. arXiv 2509.24527. Offline-only diamonds in Minecraft; policy trained in imagination, reward-based. Contrast. (Dreamer v1-3 not searched.)

## 6. Not found / not covered
- No paper found that studies how task-irrelevant factors in the GOAL IMAGE (arm pose, background colour) change the goal-embedding cost/plan/success. Closest: JEPA-Bisim and CF-JEPA (observation-side distractors), Stage-Wise Reliability (sensing degradation of observations), SLIM (goal modality), Temporal Straightening abstract (mentions distracting info in pretrained encoders). One search only; absence is not proof. Suggest the other reviewers/Google Scholar check "goal-conditioned" + "appearance gap" + "goal image masking / inpainting".
- GeoWorld numbers (CrossTask/COIN) are procedural-video, not control.
- Not checked: Dreamer v1-3, DINO-world (LeCun tweet link only), PWM, OA-WAM (2605.06481), EV-WM (2606.13053) -- UNVERIFIED, not included.
