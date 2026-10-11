"""Idea 2, step 1: does a goal-appearance shift move the goal latent mostly
OUTSIDE the action-controllable subspace (while real state changes lie inside)?

Subspaces (all training-free, rank r, from the frozen LeWM + offline data):
  ctrl   : top-r eigvecs of E_s[ Δ Δ^T ], Δ = Phi_1(z_s,u) - Phi_1(z_s,0), u~N(0,I)
  pca    : top-r PCA of dataset latents z_s
  dpca   : top-r PCA of dataset temporal differences z_{t+5} - z_t
Measures, per condition, for 400 dataset goal states g:
  nuis_in  = ||P c||^2 / ||c||^2,  c = z(render_shift(g)) - z(render(g))
  state_in = ||P d||^2 / ||d||^2,  d = z(render(g)) - z(render(g_-25))   (real progress)
  rank-acc : fraction of candidate pairs ranked the same with shifted vs clean goal
             under cost ||P(z - zg)||^2 (uses exp1 candidates/goals)
"""
import sys, os, json
import numpy as np
import torch
import h5py

from ctrlgeo.paths import SWM_ROOT

sys.path.insert(0, str(SWM_ROOT / 'scripts' / 'plan'))
from eval_goalshift import GOAL_CONDITIONS, GoalRenderer  # noqa: E402
from ctrlgeo.wm import FrozenLeWM, H5  # noqa: E402

# held-out appearance shifts (never looked at during development of this idea)
HELDOUT = {
    'bg_dark': {'background.color': np.array([60, 60, 70], dtype=np.uint8)},
    'block_blue': {'block.color': np.array([30, 90, 220], dtype=np.uint8)},
    'agent_green': {'agent.color': np.array([40, 180, 60], dtype=np.uint8)},
}
GOAL_CONDITIONS.update(HELDOUT)
CONDS = ['bg_color', 'agent_color', 'block_color', 'no_agent'] + list(HELDOUT)
RANKS = [8, 16, 32, 64, 96]

wm = FrozenLeWM()
rng = np.random.default_rng(0)
with h5py.File(H5, 'r') as f:
    n = len(f['state'])
    st_idx = np.sort(rng.choice(np.arange(n), 4000, replace=False))
    px = np.stack([f['pixels'][i] for i in st_idx])
    ep, step = f['episode_idx'][:], f['step_idx'][:]
    ok = np.where(step >= 25)[0]
    gidx = np.sort(rng.choice(ok, 400, replace=False))
    gstate = f['state'][gidx]; pstate = f['state'][gidx - 25]
    nxt = np.minimum(st_idx + 5, n - 1)
    same = ep[nxt] == ep[st_idx]
    px5 = np.stack([f['pixels'][i] for i in nxt])

zs = wm.encode(px)
z5 = wm.encode(px5)


def topr(M, r):
    ev, V = torch.linalg.eigh(M)
    return V[:, ev.argsort(descending=True)[:r]]


# controllable subspace from probes
with torch.no_grad():
    u0 = torch.zeros(len(zs), 1, 10, device='cuda')
    y0 = wm.phi(zs, u0)
    W = torch.zeros(192, 192, device='cuda')
    for _ in range(16):
        d = wm.phi(zs, torch.randn_like(u0)) - y0
        W += d.T @ d
W /= 16 * len(zs)
zc = zs - zs.mean(0)
dd = (z5 - zs)[torch.as_tensor(same, device='cuda')]
basis = {'ctrl': lambda r: topr(W, r),
         'pca': lambda r: topr(zc.T @ zc, r),
         'dpca': lambda r: topr(dd.T @ dd, r)}


def render(cond, states):
    R = GoalRenderer(cond, 224)
    return np.stack([R(s) for s in states])


z_clean = wm.encode(render('identity', gstate))
z_prev = wm.encode(render('identity', pstate))
d_state = z_clean - z_prev
out = {}
print(f'{"cond":12s} {"sub":5s} ' + ' '.join(f'r={r:<3d}(nuis/state)' for r in RANKS))
for cond in CONDS:
    c = wm.encode(render(cond, gstate)) - z_clean
    for name, B in basis.items():
        row = []
        for r in RANKS:
            U = B(r)
            fn = lambda x: ((x @ U).pow(2).sum(-1) / x.pow(2).sum(-1)).mean().item()
            row.append((fn(c), fn(d_state)))
        out[f'{cond}/{name}'] = row
        print(f'{cond:12s} {name:5s} ' + ' '.join(f'   {a:.2f}/{b:.2f}      ' for a, b in row))
    print(f'{"":12s} |c|/|d| = {(c.norm(dim=-1) / d_state.norm(dim=-1)).median().item():.2f}')
os.makedirs('results/exp3', exist_ok=True)
json.dump(out, open('results/exp3/subspace.json', 'w'))
torch.save({'W': W.cpu(), 'zs_mean': zs.mean(0).cpu(), 'pcaM': (zc.T @ zc).cpu(),
            'dpcaM': (dd.T @ dd).cpu()}, 'results/exp3/subspaces.pt')
