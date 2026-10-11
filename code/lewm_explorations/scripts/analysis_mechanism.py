"""Mechanism analysis on the exp1 candidate set (no planning; does not touch the
pre-registered confirmation).

(a) Probe-scale sweep: cf_ridge (K1, M8, rho 0.1) with probes u ~ N(0, sigma^2 I).
    If the score is a finite-difference/secant Gauss-Newton lookahead, sigma -> 0
    should recover the Jacobian score and accuracy should peak at a finite sigma.
(b) "Lucky probe" test: on pairs where best-probe and geometry disagree, is the
    best-probe-preferred candidate characterised by one outlying probe
    (large gap between best and median probe residual)?
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from exp1_ranking import pair_stats, boot  # noqa: E402
from ctrlgeo.wm import FrozenLeWM  # noqa: E402
from ctrlgeo.scores import _svd_basis, _proj_parts  # noqa: E402
from ctrlgeo import sim  # noqa: E402

D = dict(np.load('results/exp1/sim.npz'))
v, strat = D['vstar'], D['strat']
G, C = v.shape
wm = FrozenLeWM()
z = wm.encode(D['img'].reshape(G * C, 224, 224, 3))
zg = wm.encode(D['goal_img']).repeat_interleave(C, 0)
N, K, M, rho = len(z), 1, 8, 0.1


@torch.no_grad()
def probe_set(sigma, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    u0 = torch.zeros(N, K, 10, device='cuda')
    y0 = wm.phi(z, u0)
    u = sigma * torch.randn(N, M, K, 10, device='cuda', generator=g)
    Y = wm.phi(z.repeat_interleave(M, 0), u.view(N * M, K, 10)).view(N, M, -1)
    return y0, Y


def acc(score):
    ps = pair_stats(score.reshape(G, C), v, strat, 0.25)
    return boot(ps['all'])[0], boot(ps['contactish'])[0], boot(ps['free-free'])[0]


print('(a) probe-scale sweep, cf_ridge K1 M8 rho0.1  [all / contact-ish / free-free]')
for sigma in [0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
    res = []
    for seed in range(3):
        y0, Y = probe_set(sigma, seed)
        e = zg - y0
        # rho scales with sigma^2 so the action-energy penalty is scale-consistent
        U, s = _svd_basis((Y - y0[:, None]).transpose(1, 2) / M ** 0.5, 0.05)
        sc = _proj_parts(e, U, s / sigma, [rho])[f'ridge{rho:g}'].cpu().numpy()
        r_min = torch.minimum((Y - zg[:, None]).pow(2).sum(-1).min(1).values,
                              e.pow(2).sum(-1)).cpu().numpy()
        res.append(acc(sc) + acc(r_min))
    m = np.mean(res, 0)
    print(f'  sigma={sigma:4.2f}  geometry {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f}   '
          f'best-probe {m[3]:.3f}/{m[4]:.3f}/{m[5]:.3f}')

with torch.enable_grad():
    J = wm.jacobian(z, torch.zeros(N, K, 10, device='cuda'), chunk=512)
U, s = _svd_basis(J, 0.05)
e = zg - wm.phi(z, torch.zeros(N, K, 10, device='cuda'))
print('  Jacobian ridge (sigma->0 limit):', np.round(acc(_proj_parts(e, U, s, [rho])['ridge0.1'].cpu().numpy()), 3))

print('\n(b) lucky-probe test at sigma=1 (seed 0)')
y0, Y = probe_set(1.0, 0)
e = zg - y0
Ub, sb = _svd_basis((Y - y0[:, None]).transpose(1, 2) / M ** 0.5, 0.05)
geo = _proj_parts(e, Ub, sb, [rho])['ridge0.1'].cpu().numpy().reshape(G, C)
r = (Y - zg[:, None]).pow(2).sum(-1).cpu().numpy().reshape(G, C, M)
r = np.concatenate([r, e.pow(2).sum(-1).cpu().numpy().reshape(G, C, 1)], -1)
bp = r.min(-1)
luck = (np.median(r, -1) - bp) / np.median(r, -1)  # how much the best probe stands out
iu = np.triu_indices(C, 1)
stats = {'bp_right': [], 'geo_right': []}
for gi in range(G):
    dv = v[gi][iu[0]] - v[gi][iu[1]]
    db = bp[gi][iu[0]] - bp[gi][iu[1]]
    dg = geo[gi][iu[0]] - geo[gi][iu[1]]
    m = (np.abs(dv) > 0.25) & (np.sign(db) != np.sign(dg))
    for a, b, dvv, dbb in zip(iu[0][m], iu[1][m], dv[m], db[m]):
        bp_pick = a if dbb < 0 else b  # candidate preferred by best-probe
        other = b if bp_pick == a else a
        key = 'bp_right' if np.sign(dbb) == np.sign(dvv) else 'geo_right'
        stats[key].append((luck[gi, bp_pick], luck[gi, other], strat[gi, bp_pick] != sim.FREE))
for k, x in stats.items():
    x = np.array(x, dtype=float)
    print(f'  disagreements where {k:9s}: n={len(x):5d}  luck(best-probe pick)={x[:, 0].mean():.3f}  '
          f'luck(other)={x[:, 1].mean():.3f}  pick in contact-ish={x[:, 2].mean():.2f}')
