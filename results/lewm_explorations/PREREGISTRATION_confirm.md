# Pre-registered confirmation (written 2026-10-05, before any confirmation run)

Frozen: LeWM `quentinll/lewm-pusht`, code as of this commit; no changes to scores/planner.
Protocol: PushT, goal offset D=50 env steps, eval budget 100 steps, CEM 30 iters,
H=5 blocks, receding 5, matched budget B=1500 WM transitions per CEM iteration per env.
Seeds: 100..109 (never used in development), 100 episodes per seed; identical tasks per
seed across methods (paired), deterministic simulator.

Methods (hyperparameters fixed from development):
  geometry  = cf_ridge:K1:M8:rho0.1   (n=107 samples)
  L2        = l2                      (n=300)
  bestprobe = probemin:K1:M8          (n=107)

Primary:   Δ_L2    = SR(geometry) - SR(L2)
Secondary: Δ_probe = SR(geometry) - SR(bestprobe)
Statistic: paired episode-level difference, 95% CI by cluster bootstrap over seeds
           (10 clusters) and, as a check, episode-level paired bootstrap.
GO   iff Δ_L2 > 5 pts with CI excluding 0  AND  Δ_probe > 5 pts with CI excluding 0.
KILL otherwise (in particular if Δ_L2 <= 3).
