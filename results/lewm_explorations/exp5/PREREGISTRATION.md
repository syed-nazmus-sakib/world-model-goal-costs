# Idea 3 step 2 — fixed-budget branching (pre-registered 2026-10-05, before any full training)

Datasets (50k windows each = identical env-interaction budget, 112px, test episodes held out):
  EXP (B=1 expert actions), RND1 (B=1 marginal), RND4 (B=4), RND8 (B=8); RND* share one
  marginal action pool, so p(a) matches and only p(a|s) differs.
Training: identical LeWM arch/loss (pred MSE + 0.09 SIGReg), 30k steps, bs 128,
  AdamW lr 3e-4 one-cycle, wd 1e-3, bf16. Training seeds 0 (then 1, 2 if any signal).
Metrics (model-invariant; anchors from held-out test episodes):
  M1 counterfactual identification: 16 alternative 5-step action blocks per anchor (8 marginal,
     8 Gaussian), truth from simulator; top-1 accuracy matching each prediction to its true
     outcome among the 16, and normalised error ||pred-true||^2 / mean pairwise ||true_i-true_j||^2.
  M2 action ranking: Spearman(pred latent dist to goal latent, true task dist to goal) over the 16.
  M3 factual: same normalised error for the expert's own action (held-out windows).
  M4 planning: standard PushT protocol (goal +25, budget 50), CEM 300x30, eval seeds 0,1,2 x 50 eps.
PRIMARY contrast: best of {RND4, RND8} vs RND1 (pure branching, same p(a)).
GO iff  M1 normalised error improves >= 20% relative  AND  M4 improves >= +5 points
        (averaged over eval seeds and >= 2 training seeds).
KILL if RND4/RND8 within noise of RND1 on M1 and M4.
SECONDARY: RND1 vs EXP (decorrelation without branching).
