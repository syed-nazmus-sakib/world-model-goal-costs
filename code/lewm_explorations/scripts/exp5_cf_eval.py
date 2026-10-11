"""Idea 3 step 2: model-invariant counterfactual metrics (M1-M3) for trained models.

  python scripts/exp5_cf_eval.py --build            # simulator ground truth (once)
  python scripts/exp5_cf_eval.py --models results/exp5/models/*_s0
"""
import argparse, glob, json, os
from multiprocessing import Pool

import h5py
import numpy as np
import torch

from ctrlgeo import sim
from ctrlgeo.wm import FrozenLeWM, H5

GT = 'results/exp5/cf_groundtruth.npz'
A_MEAN = np.array([-0.00752025, 0.00777495]); A_STD = np.array([0.2097516, 0.20697816])
N_ANCH, N_MARG, N_GAUSS = 600, 8, 8


def sim_job(args):
    s7, g7, raw, res = args
    e = sim.make_env(resolution=res)
    e._set_state(g7); gimg = e.render()
    e._set_state(s7); snap = sim.snapshot(e); img0 = e.render(); gap = sim.agent_block_gap(e)
    imgs, tdist = [], []
    for blk in raw:
        sim.restore(e, snap)
        for a in blk:
            sim.step_raw(e, a)
        imgs.append(e.render())
        tdist.append(sim.task_dist(sim.snap_to_state7(sim.snapshot(e)), g7))
    return img0, gimg, np.stack(imgs), np.array(tdist), gap


def build(res=112):
    rng = np.random.default_rng(7)
    test_eps = np.load('results/exp5/data/test_episodes.npy')
    with h5py.File(H5, 'r') as f:
        state = f['state'][:]; act = np.nan_to_num(f['action'][:])
        ep = f['episode_idx'][:]; step = f['step_idx'][:]; L = f['ep_len'][:]
    ok = step <= L[ep] - 31
    anchors = rng.choice(np.where(ok & np.isin(ep, test_eps))[0], N_ANCH, replace=False)
    pool = np.where(ok & ~np.isin(ep, test_eps))[0]
    jobs, U = [], []
    for a in anchors:
        fact = act[a:a + 5]
        marg = np.stack([act[i:i + 5] for i in rng.choice(pool, N_MARG)])
        gauss = rng.standard_normal((N_GAUSS, 5, 2)) * A_STD + A_MEAN
        raw = np.concatenate([fact[None], marg, gauss])           # (17,5,2): 0=factual
        U.append(((raw - A_MEAN) / A_STD).reshape(len(raw), 1, 10))
        jobs.append((state[a], state[a + 25], raw, res))
    with Pool(30) as p:
        res_ = p.map(sim_job, jobs, chunksize=8)
    np.savez(GT, img0=np.stack([r[0] for r in res_]), gimg=np.stack([r[1] for r in res_]),
             imgs=np.stack([r[2] for r in res_]), tdist=np.stack([r[3] for r in res_]),
             gap=np.array([r[4] for r in res_]), U=np.stack(U).astype(np.float32))
    print('built', GT)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


@torch.no_grad()
def evaluate(path):
    D = np.load(GT)
    wm = FrozenLeWM(path=path)
    n, A = D['U'].shape[:2]
    z0 = wm.encode(D['img0']); zg = wm.encode(D['gimg'])
    zt = wm.encode(D['imgs'].reshape(n * A, *D['imgs'].shape[2:])).view(n, A, -1)
    U = torch.tensor(D['U'], device='cuda')
    zp = wm.phi(z0.repeat_interleave(A, 0), U.view(n * A, 1, 10)).view(n, A, -1)
    pair = torch.cdist(zt, zt).pow(2)                              # (n,A,A)
    scale = pair.sum((1, 2)) / (A * (A - 1))                       # mean pairwise sq dist
    err = (zp - zt).pow(2).sum(-1) / scale[:, None]                # normalised
    cf = slice(1, A)                                               # counterfactual alternatives
    d_pt = torch.cdist(zp[:, cf], zt[:, cf])                       # pred i vs true j
    top1 = (d_pt.argmin(-1) == torch.arange(A - 1, device='cuda')).float().mean().item()
    pd = (zp - zg[:, None]).pow(2).sum(-1).cpu().numpy()
    rho = np.nanmean([spearman(pd[i, cf], D['tdist'][i, cf]) for i in range(n)])
    strat = np.array([sim.classify(g) for g in D['gap']])
    e = err.cpu().numpy()
    gm = lambda x: float(np.exp(np.log(x + 1e-8).mean()))
    out = dict(model=path, cf_err_geo=gm(e[:, cf]), cf_err=float(e[:, cf].mean()), cf_err_median=float(np.median(e[:, cf])),
               cf_top1=top1, rank_spearman=float(rho), factual_err_geo=gm(e[:, 0]),
               marg_err_geo=gm(e[:, 1:1 + N_MARG]), gauss_err_geo=gm(e[:, 1 + N_MARG:]))
    for k, nm in sim.STRATA.items():
        out[f'cf_err_geo_{nm}'] = gm(e[strat == k][:, cf])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--models', nargs='*', default=[])
    ap.add_argument('--out', default='results/exp5/cf_eval.jsonl')
    args = ap.parse_args()
    if args.build:
        build()
    for m in args.models:
        r = evaluate(m)
        print(json.dumps(r), flush=True)
        with open(args.out, 'a') as f:
            f.write(json.dumps(r) + '\n')
