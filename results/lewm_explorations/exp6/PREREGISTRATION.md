# Direction A, kill test K1: Cube replication (pre-registered 2026-10-05, before generating data)

Env: OGBench cube-single (MuJoCo), 112px, frame-skip 5 (25-d action block), our simulator
convention (mj_forward after every step). Expert = OGBench CubeMarkovOracle + N(0,0.1) noise
(OGBench play-data recipe). 400 expert trajectories x 500 steps; 40 held out for evaluation.
Datasets (50k windows each, identical budget): EXP (expert's own actions), RND1 (20-step action
sequences copied from random other expert windows: same marginal, decorrelated), RND4 (4 branches/state).
Training: identical to PushT exp5 (LeWM arch, 30k steps, bs128, lr 3e-4), train seeds 0,1.
Planning eval: start = held-out expert state, goal = state 25 steps later with cube displacement
>= 5 cm; success = cube within 4 cm of goal cube position at any step within 50 steps;
CEM 300x30, H=5, receding 5; eval seeds 0,1,2 x 50 tasks.
GO (direction A survives) iff RND1 - EXP >= +10 pt planning success (mean over train x eval seeds)
AND counterfactual error(RND1) < counterfactual error(EXP).
INCONCLUSIVE if EXP and RND1 both < 10% success (budget too small) -> rerun with more data/steps.
KILL otherwise.
