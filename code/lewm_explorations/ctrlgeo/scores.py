"""Terminal-value scores for an imagined/observed latent z w.r.t. goal latent zg.

Every score is "lower = better". Each returns (scores: dict[name -> (N,) tensor],
cost: dict[name -> world-model transition-equivalents per candidate]).

Family:
  l2            ||z - zg||^2                                  (LeWM default)
  wl2           Mahalanobis with dataset latent covariance
  drift         ||e||^2, e = zg - Phi_K(z, u_nom)             (look K further, nominal action)
  probemin      min_i ||Phi_K(z, u_i) - zg||^2                (look K further, M random probes)
  lookcem       latent CEM over K steps, best residual         (look K further, optimised)
  jac_ridge     e^T (I + W/rho)^{-1} e,  W = J J^T             (= linearised lookahead with
                                                                action-energy penalty rho)
  jac_perp      ||(I - U U^T) e||^2                            (rho -> 0; unreachable residual)
  jac_energy    e_par^T W^+ e_par                              (min control energy)
  cf_ridge/perp same as jac_* with W from M counterfactual probes (nonlinear)
  l2+b*perp     L2 plus unreachable-residual penalty
"""
import torch


def _proj_parts(e, U, s, rho):
    """e:(N,D) U:(N,D,r) s:(N,r). Returns ridge score, perp, energy."""
    c = torch.einsum('ndr,nd->nr', U, e)  # coords of e in reachable basis
    e2 = e.pow(2).sum(-1)
    s2 = s.pow(2)
    out = {}
    for r_ in rho:
        out[f'ridge{r_:g}'] = e2 - (s2 / (s2 + r_) * c.pow(2)).sum(-1)
    out['perp'] = e2 - c.pow(2).sum(-1)
    out['energy'] = (c.pow(2) / s2.clamp_min(1e-12)).sum(-1)
    out['perpfrac'] = out['perp'] / e2.clamp_min(1e-12)
    return out


def _svd_basis(J, tau):
    """J:(N,D,P). Left singular vectors with s > tau * s_max."""
    D, P = J.shape[-2:]
    if P < D:  # thin case: eigendecompose the small P x P Gram matrix (much faster)
        ev, V = torch.linalg.eigh(J.transpose(-1, -2) @ J)
        ev, V = ev.flip(-1), V.flip(-1)
        s = ev.clamp_min(0).sqrt()
        U = (J @ V) / s.clamp_min(1e-8)[..., None, :]
    else:
        U, s, _ = torch.linalg.svd(J, full_matrices=False)
    keep = s > tau * s[:, :1]
    s = torch.where(keep, s, torch.zeros_like(s))
    U = U * keep[:, None, :]
    return U, s


@torch.no_grad()
def compute_scores(wm, z, zg, K_list=(1, 2), M_list=(16, 64), rho=(0.01, 0.1, 1.0),
                   tau=0.05, betas=(0.5, 1.0, 2.0), Sinv=None, seed=0, cem_iters=3):
    N, D = z.shape
    dev = z.device
    g = torch.Generator(device=dev).manual_seed(seed)
    S, C = {}, {}
    d0 = z - zg
    S['l2'] = d0.pow(2).sum(-1); C['l2'] = 0
    if Sinv is not None:
        S['wl2'] = torch.einsum('nd,de,ne->n', d0, Sinv, d0); C['wl2'] = 0

    for K in K_list:
        u0 = torch.zeros(N, K, 10, device=dev)
        y0 = wm.phi(z, u0)
        e = zg - y0
        S[f'drift_K{K}'] = e.pow(2).sum(-1); C[f'drift_K{K}'] = K

        # differential geometry (Jacobian at nominal)
        with torch.enable_grad():
            J = wm.jacobian(z, u0)
        U, s = _svd_basis(J, tau)
        P = K * 10
        for k, v in _proj_parts(e, U, s, rho).items():
            S[f'jac_{k}_K{K}'] = v; C[f'jac_{k}_K{K}'] = K * (1 + P)
        for b in betas:
            S[f'l2+{b:g}perp_jac_K{K}'] = S['l2'] + b * S[f'jac_perp_K{K}']
            C[f'l2+{b:g}perp_jac_K{K}'] = K * (1 + P)
            S[f'drift+{b:g}perp_jac_K{K}'] = S[f'drift_K{K}'] + b * S[f'jac_perp_K{K}']
            C[f'drift+{b:g}perp_jac_K{K}'] = K * (1 + P)

        for M in M_list:
            u = torch.randn(N, M, K, 10, device=dev, generator=g)
            Y = wm.phi(z.repeat_interleave(M, 0), u.view(N * M, K, 10)).view(N, M, D)
            r = (Y - zg[:, None]).pow(2).sum(-1)
            S[f'probemin_K{K}_M{M}'] = torch.minimum(r.min(1).values, S[f'drift_K{K}'])
            C[f'probemin_K{K}_M{M}'] = K * (M + 1)
            S[f'probemean_K{K}_M{M}'] = r.mean(1); C[f'probemean_K{K}_M{M}'] = K * M

            # counterfactual reachable covariance around the nominal endpoint
            Dv = (Y - y0[:, None]).transpose(1, 2) / M ** 0.5  # (N,D,M): W_cf = Dv Dv^T
            Ucf, scf = _svd_basis(Dv, tau)
            for k, v in _proj_parts(e, Ucf, scf, rho).items():
                S[f'cf_{k}_K{K}_M{M}'] = v; C[f'cf_{k}_K{K}_M{M}'] = K * (M + 1)
            for b in betas:
                S[f'l2+{b:g}perp_cf_K{K}_M{M}'] = S['l2'] + b * S[f'cf_perp_K{K}_M{M}']
                C[f'l2+{b:g}perp_cf_K{K}_M{M}'] = K * (M + 1)

            # optimised lookahead: small latent CEM, same samples per iter
            mu = torch.zeros(N, K, 10, device=dev); sd = torch.ones_like(mu)
            best = S[f'drift_K{K}'].clone()
            topk = max(2, M // 8)
            for it in range(cem_iters):
                uu = mu[:, None] + sd[:, None] * torch.randn(N, M, K, 10, device=dev, generator=g)
                Yc = wm.phi(z.repeat_interleave(M, 0), uu.view(N * M, K, 10)).view(N, M, D)
                rc = (Yc - zg[:, None]).pow(2).sum(-1)
                best = torch.minimum(best, rc.min(1).values)
                idx = rc.topk(topk, dim=1, largest=False).indices
                el = torch.gather(uu, 1, idx[:, :, None, None].expand(-1, -1, K, 10))
                mu, sd = el.mean(1), el.std(1) + 1e-3
            S[f'lookcem_K{K}_M{M}'] = best; C[f'lookcem_K{K}_M{M}'] = K * M * cem_iters
    return S, C
