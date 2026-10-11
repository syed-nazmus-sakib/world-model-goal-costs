"""Idea 3, step 1: conditional action coverage vs counterfactual error of the
frozen LeWM. See results/exp4/PREREGISTRATION.md.
"""
import json, os
from multiprocessing import Pool

import h5py
import numpy as np
import torch

from ctrlgeo import sim
from ctrlgeo.wm import FrozenLeWM, H5

OUT = 'results/exp4'
N_ANCH, KNN, P = 800, 64, 8
rng = np.random.default_rng(0)


def feats(st):
    return np.concatenate([st[:, :4], 50 * np.cos(st[:, 4:5]), 50 * np.sin(st[:, 4:5])], 1)


def sim_probe(args):
    """Execute each 5-step raw action block from state s; return renders + gap."""
    s7, blocks = args
    e = sim.make_env()
    e._set_state(s7)
    snap = sim.snapshot(e)
    gap = sim.agent_block_gap(e)
    img0 = e.render()
    imgs, bmove, cont = [], [], []
    for b in blocks:
        sim.restore(e, snap)
        c = 0
        for a in b:
            sim.step_raw(e, a)
            c += e.n_contact_points
        s1 = sim.snapshot(e)
        imgs.append(e.render())
        bmove.append(np.linalg.norm(s1[4:6] - snap[4:6]) + 50 * abs(s1[6] - snap[6]))
        cont.append(c)
    return img0, np.stack(imgs), gap, np.array(bmove), np.array(cont)


def main():
    os.makedirs(OUT, exist_ok=True)
    wm = FrozenLeWM()
    with h5py.File(H5, 'r') as f:
        state = f['state'][:]; act = f['action'][:]
        ep = f['episode_idx'][:]; step = f['step_idx'][:]; L = f['ep_len'][:]
    valid = np.where(step <= L[ep] - 6)[0]                     # has a full 5-step block
    blocks = np.stack([act[valid + i] for i in range(5)], 1)  # (V,5,2) raw
    blocks = np.nan_to_num(blocks)
    nblk = wm.norm_actions(blocks).reshape(len(valid), 10)    # normalised 10-d
    F = torch.tensor(feats(state[valid]), device='cuda', dtype=torch.float32)
    Ev = torch.tensor(ep[valid], device='cuda')
    gmu = nblk.mean(0); gcov = np.cov(nblk.T) + 1e-3 * np.eye(10)
    gci = np.linalg.inv(gcov)

    anchors = rng.choice(len(valid), N_ANCH, replace=False)
    rec, jobs = [], []
    for a in anchors:
        d = (F - F[a]).pow(2).sum(1)
        d[Ev == Ev[a]] = float('inf')
        dk, nn = d.topk(KNN, largest=False)
        nn = nn.cpu().numpy(); rk = dk[-1].sqrt().item()
        la = nblk[nn]
        lmu, lcov = la.mean(0), np.cov(la.T) + 1e-3 * np.eye(10)
        ev = np.linalg.eigvalsh(lcov)
        loc = la[rng.choice(KNN, P, replace=False)]
        marg = nblk[rng.choice(len(valid), P, replace=False)]
        gauss = rng.standard_normal((P, 10))
        U = np.concatenate([loc, marg, gauss])                 # (3P,10) normalised
        raw = wm.denorm_actions(U.reshape(3 * P, 1, 10))       # (3P,5,2)
        jobs.append((state[valid[a]], raw))
        rec.append(dict(U=U, lmu=lmu, lci=np.linalg.inv(lcov), rho=ev[0],
                        logdet=np.log(ev).sum(), rk=rk))
    with Pool(30) as pool:
        res = pool.map(sim_probe, jobs, chunksize=4)

    z0 = wm.encode(np.stack([r[0] for r in res]))
    zt = wm.encode(np.concatenate([r[1] for r in res])).view(N_ANCH, 3 * P, -1)
    U = torch.tensor(np.stack([r['U'] for r in rec]), device='cuda', dtype=torch.float32)
    with torch.no_grad():
        zp = wm.phi(z0.repeat_interleave(3 * P, 0), U.view(-1, 1, 10)).view(N_ANCH, 3 * P, -1)
    err = (zp - zt).pow(2).sum(-1).cpu().numpy()
    disp = (zt - z0[:, None]).pow(2).sum(-1).sqrt().cpu().numpy()
    Un = U.cpu().numpy()
    lmah = np.stack([np.einsum('pi,ij,pj->p', Un[i] - r['lmu'], r['lci'], Un[i] - r['lmu'])
                     for i, r in enumerate(rec)])
    gmah = np.einsum('npi,ij,npj->np', Un - gmu, gci, Un - gmu)
    np.savez(f'{OUT}/raw.npz', err=err, disp=disp, lmah=lmah, gmah=gmah,
             rho=[r['rho'] for r in rec], logdet=[r['logdet'] for r in rec],
             rk=[r['rk'] for r in rec], gap=[r[2] for r in res],
             bmove=np.stack([r[3] for r in res]), cont=np.stack([r[4] for r in res]))
    analyse()


def rankdata(x):
    return np.argsort(np.argsort(x)).astype(float)


def resid(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def analyse():
    D = np.load(f'{OUT}/raw.npz')
    err, disp, lmah, gmah = D['err'], D['disp'], D['lmah'], D['gmah']
    n = err.shape[0]
    grp = {'LOCAL': slice(0, P), 'MARG': slice(P, 2 * P), 'GAUSS': slice(2 * P, 3 * P)}
    lines = []
    W = lines.append
    for k, s in grp.items():
        W(f'{k:6s} err mean={err[:, s].mean():.3f} median={np.median(err[:, s]):.3f}  '
          f'disp={disp[:, s].mean():.2f}  localMaha={np.median(lmah[:, s]):.1f}  globalMaha={np.median(gmah[:, s]):.1f}')
    # primary 1: paired ratio after regressing out displacement (log-space)
    le, ld = np.log(err + 1e-6), np.log(disp + 1e-6)
    r_adj = resid(le.ravel(), np.column_stack([ld.ravel(), ld.ravel() ** 2])).reshape(n, -1)
    diff = r_adj[:, grp['MARG']].mean(1) - r_adj[:, grp['LOCAL']].mean(1)
    bs = [diff[rng.integers(0, n, n)].mean() for _ in range(5000)]
    W(f'\nPRIMARY ratio MARG/LOCAL (disp-adjusted, geometric) = {np.exp(diff.mean()):.3f} '
      f'CI[{np.exp(np.percentile(bs, 2.5)):.3f},{np.exp(np.percentile(bs, 97.5)):.3f}]  '
      f'raw ratio of means = {err[:, grp["MARG"]].mean() / err[:, grp["LOCAL"]].mean():.3f}')
    # primary 2: partial Spearman with anchor fixed effects
    def fe(x):
        return (x - x.mean(1, keepdims=True)).ravel()
    R = lambda x: rankdata(x.ravel()).reshape(x.shape)
    y = fe(R(err)); xl = fe(R(lmah))
    C = np.column_stack([fe(R(gmah)), fe(R(disp))])
    pr = np.corrcoef(resid(y, C), resid(xl, C))[0, 1]
    pg = np.corrcoef(resid(y, np.column_stack([xl, fe(R(disp))])),
                     resid(fe(R(gmah)), np.column_stack([xl, fe(R(disp))])))[0, 1]
    W(f'PRIMARY partial Spearman(err, localMaha | globalMaha, disp, anchor FE) = {pr:+.3f}')
    W(f'  (reference) partial Spearman(err, globalMaha | localMaha, disp, anchor FE) = {pg:+.3f}')
    # secondary: across states
    ge = np.log(err[:, grp['GAUSS']].mean(1))
    strat = np.array([sim.classify(g) for g in D['gap']])
    Cs = np.column_stack([rankdata(np.log(D['rk'])), strat == 1, strat == 2])
    ps = np.corrcoef(resid(rankdata(ge), Cs), resid(rankdata(D['rho']), Cs))[0, 1]
    pd = np.corrcoef(resid(rankdata(ge), Cs), resid(rankdata(D['logdet']), Cs))[0, 1]
    W(f'SECONDARY partial Spearman(GAUSS err, rho | density, stratum) = {ps:+.3f}   (logdet: {pd:+.3f})')
    W(f'  plain Spearman(GAUSS err, log kNN radius) = {np.corrcoef(rankdata(ge), rankdata(D["rk"]))[0, 1]:+.3f}')
    for k, name in sim.STRATA.items():
        m = strat == k
        W(f'  stratum {name:11s} n={m.sum():4d}  LOCAL={err[m][:, grp["LOCAL"]].mean():.3f} '
          f'MARG={err[m][:, grp["MARG"]].mean():.3f} GAUSS={err[m][:, grp["GAUSS"]].mean():.3f}')
    # confound control: block motion / contact caused by the probe action
    bm, ct = D['bmove'], D['cont']
    W('\nCONFOUND: probe-induced contact / block motion')
    for k, s in grp.items():
        W(f'  {k:6s} frac contact={np.mean(ct[:, s] > 0):.2f}  mean block move={bm[:, s].mean():.2f}')
    Cc = np.column_stack([fe(R(gmah)), fe(R(disp)), fe(R(bm)), fe((ct > 0).astype(float))])
    pr2 = np.corrcoef(resid(y, Cc), resid(xl, Cc))[0, 1]
    W(f'partial Spearman(err, localMaha | globalMaha, disp, blockmove, contact, anchor FE) = {pr2:+.3f}')
    r2 = resid(le.ravel(), np.column_stack([ld.ravel(), ld.ravel() ** 2, np.log1p(bm.ravel()),
                                            (ct.ravel() > 0)])).reshape(n, -1)
    for nm, mask in [('all', np.ones_like(ct, bool)), ('no-contact probes', ct == 0),
                     ('contact probes', ct > 0)]:
        a = np.array([r2[i, grp['MARG']][mask[i, grp['MARG']]].mean() - r2[i, grp['LOCAL']][mask[i, grp['LOCAL']]].mean()
                      if mask[i, grp['MARG']].any() and mask[i, grp['LOCAL']].any() else np.nan for i in range(n)])
        a = a[~np.isnan(a)]
        bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(3000)]
        W(f'  ratio MARG/LOCAL adj. for disp+blockmove+contact [{nm:17s}] n={len(a):3d} = {np.exp(a.mean()):.3f} '
          f'CI[{np.exp(np.percentile(bs, 2.5)):.3f},{np.exp(np.percentile(bs, 97.5)):.3f}]')
    txt = '\n'.join(lines); print(txt)
    open(f'{OUT}/report.txt', 'w').write(txt)


if __name__ == '__main__':
    main()
