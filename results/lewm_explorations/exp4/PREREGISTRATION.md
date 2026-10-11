# Idea 3 step 1 (frozen LeWM) — pre-registered 2026-10-05, before running

Question: does LeWM's counterfactual prediction error depend on CONDITIONAL action
coverage p(a|s) beyond marginal action coverage p(a) and state density?

Anchors: 800 dataset states. Neighbours: k=64 nearest dataset states (true sim
state features, other episodes only). Probes per anchor (model step = 5 env steps):
  LOCAL : 8 action blocks taken from the neighbours        (on conditional support)
  MARG  : 8 action blocks taken from random dataset states (same marginal, off conditional support)
  GAUSS : 8 blocks u~N(0,I) normalised                    (CEM proposal)
Error: ||Phi_1(z_s,u) - enc(render(sim(s,u)))||^2.

PRIMARY (within-state, density held fixed by design):
  ratio = mean err(MARG)/err(LOCAL), paired over anchors, after regressing out true
  latent displacement ||z_true - z_s||. GO needs ratio >= 1.3 with 95% CI excl. 1
  AND partial Spearman(err, local Mahalanobis | global Mahalanobis, displacement,
  anchor fixed effects) >= 0.15.
SECONDARY (across states): partial Spearman(rho(s)=lambda_min Cov(a|s), mean GAUSS
  error | log kNN radius, stratum) <= -0.15.
KILL if primary fails.
