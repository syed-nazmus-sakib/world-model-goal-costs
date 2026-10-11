"""Batched CEM over LeWM with a pluggable terminal value, mirroring
stable_worldmodel.planning.CEMSolver + GoalMSE (PushT config: H=5 blocks,
receding 5, 300 samples, 30 iters, top-30, init N(0,1), candidate 0 = mean).
"""
import torch

from ctrlgeo.scores import _svd_basis


class TerminalValue:
    """Callable (zT:(N,D), zg:(N,D)) -> (N,) cost; .extra = transitions/candidate."""

    def __init__(self, wm, kind='l2', K=1, M=16, rho=0.1, beta=1.0, tau=0.05, seed=0):
        self.wm, self.kind, self.K, self.M = wm, kind, K, M
        self.rho, self.beta, self.tau = rho, beta, tau
        self.g = torch.Generator(device=wm.dev).manual_seed(seed)
        self.extra = {'l2': 0, 'probemin': K * (M + 1), 'cf_ridge': K * (M + 1),
                      'cf_perp': K * (M + 1), 'l2+cf_perp': K * (M + 1),
                      'jac_ridge': K * (1 + 10 * K), 'drift': K}[kind]

    @torch.no_grad()
    def __call__(self, zT, zg):
        wm, K, M, N = self.wm, self.K, self.M, zT.shape[0]
        if self.kind == 'l2':
            return (zT - zg).pow(2).sum(-1)
        u0 = torch.zeros(N, K, 10, device=zT.device)
        y0 = wm.phi(zT, u0)
        e = zg - y0
        e2 = e.pow(2).sum(-1)
        if self.kind == 'drift':
            return e2
        if self.kind == 'jac_ridge':
            U, s = _svd_basis(wm.jacobian(zT, u0, chunk=512), self.tau)
        else:
            u = torch.randn(N, M, K, 10, device=zT.device, generator=self.g)
            Y = wm.phi(zT.repeat_interleave(M, 0), u.view(N * M, K, 10)).view(N, M, -1)
            if self.kind == 'probemin':
                return torch.minimum((Y - zg[:, None]).pow(2).sum(-1).min(1).values, e2)
            U, s = _svd_basis((Y - y0[:, None]).transpose(1, 2) / M ** 0.5, self.tau)
        c = torch.einsum('ndr,nd->nr', U, e)
        if self.kind in ('cf_ridge', 'jac_ridge'):
            s2 = s.pow(2)
            return e2 - (s2 / (s2 + self.rho) * c.pow(2)).sum(-1)
        perp = e2 - c.pow(2).sum(-1)
        if self.kind == 'cf_perp':
            return perp
        return (zT - zg).pow(2).sum(-1) + self.beta * perp  # l2+cf_perp


@torch.no_grad()
def cem_plan(wm, z0, zg, value, H=5, n_samples=300, n_iters=30, topk=30, gen=None,
             rerank=0):
    """z0, zg: (B,D). Returns mean action plan (B,H,10) and its cost.
    rerank>0: score all samples with L2, then re-score only the best `rerank`
    of them with `value` (the rest keep +inf); elites come from the re-scored set."""
    B, D = z0.shape
    dev = z0.device
    A = wm.block_dim
    mu = torch.zeros(B, H, A, device=dev)
    sd = torch.ones_like(mu)
    zrep = z0.repeat_interleave(n_samples, 0)
    grep = zg.repeat_interleave(n_samples, 0)
    for _ in range(n_iters):
        u = mu[:, None] + sd[:, None] * torch.randn(B, n_samples, H, A, device=dev, generator=gen)
        u[:, 0] = mu
        zT = wm.phi(zrep, u.view(B * n_samples, H, A))
        if rerank:
            l2 = (zT - grep).pow(2).sum(-1).view(B, n_samples)
            top = l2.topk(rerank, dim=1, largest=False).indices  # (B,R)
            flat = (top + torch.arange(B, device=dev)[:, None] * n_samples).view(-1)
            cost = torch.full_like(l2, float('inf'))
            cost.scatter_(1, top, value(zT[flat], grep[flat]).view(B, rerank))
        else:
            cost = value(zT, grep).view(B, n_samples)
        idx = cost.topk(topk, dim=1, largest=False).indices
        el = torch.gather(u, 1, idx[:, :, None, None].expand(-1, -1, H, A))
        mu, sd = el.mean(1), el.std(1)
    return mu, cost.gather(1, idx[:, :1]).squeeze(1)
