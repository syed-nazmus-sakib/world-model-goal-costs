# Lit review E: CVPR fit, calibration, contribution types (as of 2026-10-15)
Legend: VERIFIED = fetched primary page (cvpr.thecvf.com / arXiv abs / OpenReview content) this session. UNVERIFIED = only secondary/search-snippet evidence.

## 1. CVPR 2027 official facts
Source: https://cvpr.thecvf.com/Conferences/2027/Dates and /CallForPapers (VERIFIED, fetched)
- Paper registration (abstract): Nov 10, 2026 AoE. VERIFIED
- Paper submission: Nov 16, 2026 AoE, no extensions. VERIFIED
- Supplementary: Nov 23, 2026. VERIFIED
- Reviews released Jan 25, 2027; final decisions Feb 25, 2027. VERIFIED (rebuttal window not listed on fetched page)
- Workshops/tutorials Jun 20-21; main Jun 22-25, 2027. VERIFIED. Location: home page says Seattle WA (map link inconsistently points to a San Francisco hotel); treat Seattle as probable.
- Topic list on Call for Papers includes: "Embodied vision and robotics", "Agentic AI", "Physical AI: World Models, VLA, Planning, Simulation", "Autonomous driving", "Reasoning & Reinforcement Learning", "Multimodal Generative and Foundation Models". VERIFIED. => Physical AI/World Models is an explicit named topic. No separate tracks defined on the page.
- LLM policy: CFP says CVPR 2027 will use LLMs in selected parts of the conference/review process; scope "currently being finalized". VERIFIED (via fetch + search). Not final.
- Reviewer duty: all authors must serve as reviewer/AC/SAC when invited; exemptions need PC approval. VERIFIED.
- Dual submission: no substantially similar paper elsewhere during review (Nov 16, 2026 - Feb 22, 2027). VERIFIED.
- Page limit / Author Guidelines 2027: the 2027 AuthorGuidelines and ReviewerGuidelines URLs returned 404 = NOT YET PUBLISHED. Proxy: CVPR 2026 Author Guidelines (VERIFIED fetch): 8 pages incl. figures/tables, references-only extra pages; non-anonymous/over-length/wrong template = reject without review; supplementary optional, code encouraged not required; no external links in paper/supp/rebuttal; 1-page rebuttal, no new experiments unless reviewers ask; authors may use LLMs but are responsible for citations (fake citations can lead to rejection); hidden prompt-injection text = ethics violation, desk reject; dataset claimed as contribution must be public by camera-ready; negative societal impact discussion expected; OpenReview profiles complete by abstract deadline or desk-reject; author list frozen after abstract deadline (reorder only until paper deadline). 2026 reviewers barred from using LLMs to write reviews (search-sourced, UNVERIFIED on primary page).
- Reproducibility: no explicit requirement on 2027 CFP page. VERIFIED absent. (No checklist found.)
- Dataset/benchmark expectations: no special benchmark track found on 2027 page. Datasets only count as contributions if public by camera-ready (2026 rule).
ACTION: re-check cvpr.thecvf.com/Conferences/2027/AuthorGuidelines weekly; assume 8 pages + 2026 template rules.

## 2. Calibration table: papers (venue, type, scale)
### CVPR 2025 / 2026
1. Navigation World Models (Bar, Zhou, Tran, Darrell, LeCun) arXiv 2412.03572. CVPR 2025 ORAL (also award candidate on CVPR virtual page). VERIFIED. Method; 1B-param CDiT video diffusion trained on egocentric human+robot video; planning by simulating/ranking trajectories. Large compute (not one-GPU).
2. WorldLens: Full-Spectrum Evaluations of Driving World Models, arXiv 2512.10958. CVPR 2026 ORAL (per GitHub repo worldbench/WorldLens and oral-list repo; CVPR page not directly fetched). PARTLY VERIFIED. Benchmark; 5 aspects, 20+ metrics, plus WorldLens-26K dataset and Agent. Driving-focused, no robot needed.
3. GeoWorld: Geometric World Models (Zhang, Li, Reid, Hartley) arXiv 2602.23058. "Accepted to CVPR 2026" in arXiv page. VERIFIED. Method (Hyperbolic JEPA + geometric RL); CrossTask and COIN, baseline V-JEPA 2; gains only ~3% (3-step) and ~2% (4-step) SR. => small gains over one strong baseline on two datasets were enough for CVPR poster. No real robot.
4. CompACT / Planning in 8 Tokens (Kim, Seo, Lee, Cho, Kwak) arXiv 2603.05438. Comments: "CVPR 2026". VERIFIED. Method; tokenizer for fast latent-WM planning (speed-for-accuracy tradeoff; planning "orders of magnitude" faster). Benchmarks not read.
5. SimScale (Learning to Drive via Real-World Simulation at Scale) arXiv 2511.23369. CVPR 2026 ORAL stated in arXiv page. VERIFIED. Method/data-scaling; driving, benchmarks navhard/navtest.
6. PAI-Bench (Zhou, ..., Ramanan, Shi) arXiv 2512.01989. On CVPR 2026 oral list repo (UNVERIFIED as oral; arXiv page lists no venue). Benchmark, 2,808 real cases.
7. PointWorld (Huang, ..., Fei-Fei Li) arXiv 2601.03782. CVPR 2026 per a 2026 survey citation only. UNVERIFIED venue. Method/large pretrain; ~2M trajectories/500h, Franka + humanoid real robots; MPC at 0.1 s.
8. Other CVPR 2026 orals on community list (oral list repo; UNVERIFIED individually): AT-VLA, Structural Action Transformer for 3D Dexterous Manipulation (2603.03960), SocialNav (2511.21135), SparseWorld-TC (2511.22039), ProPhy (2512.05564), NitroGen (2601.02427). Observation: orals in embodied/WM space are dominated by driving, benchmarks, and large-scale/real-robot works.
9. CVPR 2025 Oral Session 4B "Embodied Computer Vision" exists (VERIFIED via search listing of cvpr.thecvf.com/virtual/2025/session/35394); NWM is in it alongside PDFactor (policy paper).
### Robotics / WM, other venues
10. DINO-WM (Zhou, Pan, LeCun, Pinto) arXiv 2411.04983. ICML 2025 poster (icml.cc page seen in search; arXiv abs shows no venue). Method; 6 envs (maze, push, multi-particle...). Single-GPU-friendly; the baseline suite for your project.
11. PLDM: Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models (Sobal, ..., LeCun) arXiv 2502.14819. Venue not on arXiv page (v4 Oct 2025). VERIFIED arXiv; venue UNVERIFIED. Analysis/comparison (RL vs latent planning) on navigation; data-quality studies.
12. JEPA-WMs: What Drives Success in Physical Planning with JEPA World Models (Terver, Yang, Ponce, Bardes, LeCun) arXiv 2512.24497. ICLR 2026 REJECTED (OpenReview 5744, forum TuYC5Fpp7M), later accepted at TMLR with Reproducibility Certification. VERIFIED (OpenReview search content). Analysis + recipe. ICLR reviews 6/6/4 (conf 3).
13. LeWorldModel (Maes, Le Lidec, Scieur, LeCun, Balestriero) arXiv 2603.19312, v3 Jun 2026; no peer-reviewed venue listed. VERIFIED. Method; 15M params, single GPU few hours, plans 48x faster than foundation-model WMs; tasks Push-T, Two-Room, Reacher, OGBench-Cube per follow-ups.
14. V-JEPA 2 (Assran et al.) arXiv 2506.09985; no venue listed on arXiv. VERIFIED. Method/foundation; >1M hours video, AC post-train on <62h DROID, zero-shot Franka in two labs (real robot).
15. EB-JEPA library (Terver ... Bar) arXiv 2602.03604; ICLR 2026 World Models workshop camera-ready. VERIFIED. Library; single GPU few hours; Two Rooms 97%.
16. Causal-JEPA (Nam, Le Lidec, Maes, LeCun, Balestriero) arXiv 2602.11389; ICML 2026. VERIFIED. Method, object-level masking; uses 1% latent features for planning.
17. TD-JEPA (Bagatella, ..., Tirinzoni) arXiv 2510.00739; ICLR 2026 ORAL (iclr.cc oral page exists in search; UNVERIFIED label, strong). 13 datasets across ExoRL + OGBench. Method.
18. Sparse Imagination for Efficient Visual World Model Planning (Chun, Jeong, Kim) arXiv 2506.01392; ICLR 2026 POSTER (OpenReview note). VERIFIED. Method; LIBERO-10, Meta-World + real robot. Reviewer score seen: 4 (soundness 3, contrib 3).
19. Closing the Train-Test Gap in World Models for Gradient-Based Planning (OpenReview By4u7V6tAx). ICLR 2026 REJECTED. VERIFIED (review text). Method; DINO-WM suite 3 tasks.
20. Benchmarking World-Model Learning (HuNIgYhBoy) ICLR 2026 rejected; 4D Latent World Model for Robot Planning (iB9qx28gv4) rejected; MoWM (4NYdDrRPSc, arXiv 2509.21797) rejected. VERIFIED status (OpenReview listing), no review text obtained.
21. Ctrl-World (Guo, Shi, Chen, Finn) arXiv 2510.10125; venue ICLR 2026 claimed only by a search snippet = UNVERIFIED. VERIFIED abstract: DROID 95k traj/564 scenes; policy eval + SFT (+44.7%).
22. DreamGen (Jang et al.) arXiv 2505.12705; no venue on arXiv. VERIFIED. Method+DreamGen Bench; humanoid, 22 new behaviors; NVIDIA scale.
23. Cosmos World Foundation Model Platform arXiv 2501.03575 (NVIDIA, 77 authors); no venue listed. VERIFIED. Platform/foundation.
24. WorldGym (Quevedo, ..., Percy Liang, Sherry Yang) arXiv 2506.00613; no venue listed. VERIFIED. Policy evaluation in a video WM; correlation with real success.
25. WorldArena (Shang, ..., Yong Li) arXiv 2602.08971; preprint, no venue. VERIFIED. Benchmark; 14 world models, 16 perception metrics, 3 functional roles (data engine/policy evaluator/action planner); finding: perception-functionality gap.
26. Dreamer 4 (Hafner, Yan, Lillicrap) arXiv 2509.24527; no venue listed. VERIFIED. Method; offline Minecraft diamonds; real-time on single GPU inference.
27. LingBot-VA, Causal World Modeling for Robot Control arXiv 2601.21998; preprint. VERIFIED. World-action model; sim + real.
28. VLA-JEPA arXiv 2602.10098; preprint. VERIFIED. LIBERO, LIBERO-Plus, SimplerEnv, real tasks.
29. H-JEPA (Zhang, Terver, Rabbat, LeCun, Balestriero) arXiv 2610.06805 (5 Oct 2026). VERIFIED preprint. 4 sim nav/manip envs (Visual AntMaze 18%->73%) + DROID offline. Direct competitor to your direction.
### Fresh LeWM/DINO-WM-style follow-ups (all arXiv preprints, no venue on abs page; VERIFIED)
30. Traj-LeWM 2608.14125 (Push-T, OGBench-Cube, Reacher, Two-Room; only LeWM baseline).
31. Temporal-Distance JEPA 2607.25337 (Two-Room, OGB-Cube, Push-T; baselines LeWM + RC-aux; seeds not stated).
32. Delta-JEPA 2606.31232 (4 tasks; LeWM 84.53, PLDM 76.13 on Push-T per search snippet).
33. ARC-Bench 2609.05461 (analysis/benchmark): top-scored candidate almost always suboptimal in frozen JEPA-WMs; closed-loop replanning masks it; released checkpoints + V-JEPA 1/2 ViT-L/G swaps. 15pp.
34. ACPC, Diagnosing JEPA WMs 2608.12939 (analysis; 4 tasks, 2 models: LeWM, PLDM).
35. Stage-wise reliability under sensing degradation 2609.07126 (analysis; 10 degradations, 50 tasks, 8 pages).
36. What Makes Video WM Latents Action-Relevant 2606.07687 (analysis; V-JEPA vs VideoMAE, CALVIN).
=> The field is flooded with LeWM +x-point method tweaks on the same 4 tasks (Push-T/Two-Room/Reacher/OGB-Cube) and a growing analysis/diagnostic strand.

## 3. What reviewers criticised (OpenReview, ICLR 2026; text fetched via OpenReview search endpoint)
- JEPA-WMs (rejected; 6/6/4): "more of a practitioner's guide than a scientific leap"; novelty "mainly empirical"/composes known parts; real-robot eval only offline action matching on 16 Franka videos, "No closed loop robot trials are reported"; underperforms V-JEPA 2-AC on Robocasa Reach (wants failure analysis); no gradient-based/hybrid planner compared; encoder comparison confounded (two-frame encoding), wants fairness table; narrative "list of results" not prioritized; unclear whether "JEPA-WM" is algorithm or recipe.
- Closing the Train-Test Gap (rejected; 6/4/2): only three DINO-WM-suite sim tasks; "small 2-D domains, no high-dimensional or real-robot experiments"; incomplete baselines (CEM+gradient hybrids, iLQR/DDP, MPPI, TD-MPC2); no ablations for hyperparameters; incremental novelty (DAgger/adversarial training); wall-clock reported for 1 task; assumes simulator reset.
- Sparse Imagination (accepted poster; score 4 in one review): readability, key baselines in appendix, narrow tasks/encoders, only MPC/CEM, manual drop ratio, coarse compute breakdown. (A related "Sparse World Models" review: "limited datasets and weak baselines... no real-world evaluation... only one encoder size".)
- Benchmarks with narrow domain/ GPT-5 dependence/reproducibility drew 4-ish scores (CubeBench, GUI Knowledge Bench 2).
Caveat: these are ICLR reviews, not CVPR; CVPR reviewers weigh visual/vision framing more. No meta-reviews were retrievable (OpenReview API returns challenge from curl/WebFetch except through the search endpoint).

## 4. Lessons / recommendations for one-GPU, simulator-only group
Verified patterns:
a) Simulator-only is not disqualifying: GeoWorld (CrossTask/COIN, +2-3%), CompACT, WorldLens (benchmark, driving) are CVPR 2026; Sparse Imagination poster at ICLR had real robot though. But ICLR reviewers of purely sim JEPA papers repeatedly ask for real-robot/closed-loop evidence or flag "small 2-D domains". Expect the same from CVPR; mitigate with 3D/pixel-rich envs (Metaworld, OGBench-Cube/visual scene, LIBERO) and frank scoping rather than claiming general physical planning.
b) Breadth bar suggested by reviews: >=4-6 environments including at least 2 non-trivial 3D/manipulation; >=2-3 frozen encoders (DINOv2/v3, V-JEPA 2, plus a non-DINO e.g. SigLIP/pixel/LeWM end-to-end) and >=2 predictor/WM families (DINO-WM, JEPA-WMs/AdaLN, PLDM, LeWM); multiple planners (CEM, MPPI, gradient, iLQR/hybrid); >=3-5 seeds with CIs (many follow-ups report none - a differentiator); compute/wall-clock table across all tasks.
c) Required baselines: DINO-WM, PLDM, LeWM, JEPA-WMs, V-JEPA 2-AC, plus a model-free/goal-conditioned (GCBC/IQL/TD-MPC2) reference.
d) Contribution type: the +x-point LeWM tweak is saturated (>=4 near-identical preprints Jun-Oct 2026, mostly same four tasks). Analysis/diagnostic papers (ARC-Bench, ACPC, sensing-degradation, action-relevance) exist but are all preprints so far (no verified top-venue acceptance). Strong analysis needs a surprising, actionable finding plus a fix or protocol (e.g., ranking-quality metric that predicts planning success; closed-loop replanning masking). Pure benchmarks that got CVPR orals (WorldLens, PAI-Bench) had large metric suites and human/leaderboard infrastructure; WorldArena-style (14 models, 16 metrics) is beyond a one-GPU group unless the models evaluated are small JEPA-WMs.
e) Realistic sweet spot: a rigorous analysis+protocol paper (e.g., planning-relevance of frozen encoder features across many encoders/envs, ranking vs closed-loop diagnostics) with one simple remedy that improves several WMs, with seeds/CIs and code (JEPA-WMs got TMLR Reproducibility Certification even after ICLR rejection - reproducibility is valued but not sufficient at ICLR).
f) Real robot: not formally required (CFP/guidelines silent), but most accepted embodied papers listed here (PointWorld, V-JEPA 2, DreamGen, Sparse Imagination, VLA-JEPA) include it, and reviewers explicitly called its absence a weakness. Hedge: pick sim benchmarks with realistic pixels (LIBERO, Metaworld, DROID offline planning fidelity as H-JEPA did) and avoid overclaim.
g) Timeline: abstract in 26 days (Nov 10), paper Nov 16, supp Nov 23. Competition risk: H-JEPA (Oct 5), Traj-LeWM, Delta-JEPA overlap with LeWM-improvement directions.

## 5. Not verified / gaps
- CVPR 2027 Author/Reviewer Guidelines (404); exact page limit, LLM rules for authors, reproducibility checklist unknown.
- No CVPR 2025/2026 official acceptance-type (poster/highlight) data for most papers; only oral claims noted above. CVPR 2026 papers accepted-list not enumerated.
- Reviews for LeWM (no venue listed), Sparse Imagination text, 4D Latent WM, Benchmarking World-Model Learning not retrieved.
- Direct openreview.net / api2 notes?forum= access blocked by bot challenge; only /notes/search endpoint worked.
