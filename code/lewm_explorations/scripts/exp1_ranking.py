"""Kill test 1: does local reachable-subspace structure rank states by true
continuation value better than L2 / simply looking farther?

Stage A (CPU, simulator, privileged): for G (start, goal) groups from the expert
  dataset, generate C candidate endpoints by executing 25 env steps of noisy
  expert / random actions; label each with
    V*  = best true task distance reachable in the next L env steps
          (simulator CEM, two restarts), and its contact stratum.
Stage B (GPU, frozen LeWM only): encode candidate & goal renders, compute scores.
Stage C: within-group Spearman, pairwise accuracy by stratum, top-1 regret,
  bootstrap CIs over groups.

Usage:
  python scripts/exp1_ranking.py --groups 200 --out results/exp1
  python scripts/exp1_ranking.py --stage C --out results/exp1   # re-analyse only
"""
import argparse, json, os, time
from multiprocessing import Pool

import numpy as np

from ctrlgeo import sim

from ctrlgeo.paths import H5
A_STD = np.array([0.2098, 0.2070])


# ----------------------------------------------------------------- stage A
def sim_cem(e, snap0, goal7, mean0, L, n, iters, elite, std0, rng):
    mean = mean0.copy(); std = np.full_like(mean, std0)
    best = np.inf
    for _ in range(iters):
        A = mean[None] + std[None] * rng.standard_normal((n, L, 2))
        A[0] = mean
        cost = np.empty(n)
        for i in range(n):
            sim.restore(e, snap0)
            c = np.inf
            for t in range(L):
                sim.step_raw(e, A[i, t])
                c = min(c, sim.task_dist(sim.snap_to_state7(sim.snapshot(e)), goal7))
            cost[i] = c
        best = min(best, cost.min())
        el = A[np.argsort(cost)[:elite]]
        mean, std = el.mean(0), el.std(0) + 1e-3
    return best


def work_group(args):
    gi, state_t, goal7, act_plan, act_cont, cfg = args
    rng = np.random.default_rng(cfg['seed'] * 100003 + gi)
    e = sim.make_env()
    e._set_state(goal7); goal_img = e.render()
    e._set_state(state_t); start = sim.snapshot(e)
    C, nexp = cfg['C'], cfg['C_expert']
    sig = np.linspace(0.0, cfg['max_noise'], nexp)
    out = dict(snap=[], img=[], gap=[], strat=[], d_now=[], vstar=[], kind=[])
    for j in range(C):
        if j < nexp:
            A = act_plan + sig[j] * A_STD * rng.standard_normal(act_plan.shape)
            kind = sig[j]
        else:  # CEM-prior-like random actions (normalised N(0,1))
            A = rng.standard_normal(act_plan.shape) * A_STD
            kind = -1.0
        sim.restore(e, start)
        for a in A:
            sim.step_raw(e, a)
        snap = sim.snapshot(e)
        gap = sim.agent_block_gap(e)
        img = e.render()
        v1 = sim_cem(e, snap, goal7, act_cont, cfg['L'], cfg['n'], cfg['iters'],
                     cfg['elite'], 0.15, rng)
        v2 = sim_cem(e, snap, goal7, np.zeros_like(act_cont), cfg['L'], cfg['n'],
                     cfg['iters'], cfg['elite'], 0.3, rng)
        d_now = sim.task_dist(sim.snap_to_state7(snap), goal7)
        for k, v in zip(out, [snap, img, gap, sim.classify(gap), d_now,
                              min(v1, v2, d_now), kind]):
            out[k].append(v)
    out = {k: np.stack(v) if k == 'img' else np.array(v) for k, v in out.items()}
    out['goal_img'] = goal_img
    return gi, out


def stage_a(cfg, out_dir):
    import h5py
    rng = np.random.default_rng(cfg['seed'])
    with h5py.File(H5, 'r') as f:
        off, L = f['ep_offset'][:], f['ep_len'][:]
        need = cfg['plan_steps'] + cfg['L'] + 1
        eps = rng.choice(np.where(L >= need + 1)[0], cfg['groups'], replace=False)
        jobs = []
        for gi, ep in enumerate(eps):
            t = rng.integers(0, L[ep] - need)
            o = off[ep] + t
            st = f['state'][o:o + need]
            ac = f['action'][o:o + need]
            P = cfg['plan_steps']
            jobs.append((gi, st[0], st[P + cfg['L']], ac[:P], ac[P:P + cfg['L']], cfg))
    t0 = time.time()
    res = [None] * len(jobs)
    with Pool(cfg['workers']) as pool:
        for k, (gi, r) in enumerate(pool.imap_unordered(work_group, jobs)):
            res[gi] = r
            if k % 10 == 0:
                print(f'  sim group {k}/{len(jobs)}  {time.time() - t0:.0f}s', flush=True)
    keys = [k for k in res[0] if k != 'goal_img']
    data = {k: np.stack([r[k] for r in res]) for k in keys}
    data['goal_img'] = np.stack([r['goal_img'] for r in res])
    np.savez(os.path.join(out_dir, 'sim.npz'), **data)
    return data


# ----------------------------------------------------------------- stage B
def stage_b(data, out_dir, args):
    import torch, h5py
    from ctrlgeo.wm import FrozenLeWM
    from ctrlgeo.scores import compute_scores
    wm = FrozenLeWM()
    G, C = data['vstar'].shape
    z = wm.encode(data['img'].reshape(G * C, 224, 224, 3))
    zg = wm.encode(data['goal_img']).repeat_interleave(C, 0)
    # dataset latent covariance for whitened L2
    with h5py.File(H5, 'r') as f:
        idx = np.sort(np.random.default_rng(0).choice(len(f['pixels']), 4000, replace=False))
        px = np.stack([f['pixels'][i] for i in idx])
    zd = wm.encode(px)
    Sig = torch.cov(zd.T) + 1e-4 * torch.eye(zd.shape[1], device=zd.device)
    Sinv = torch.linalg.inv(Sig)
    S_all, Cost = {}, None
    bs = args.bs
    t0 = time.time()
    for i in range(0, G * C, bs):
        S, Cost = compute_scores(wm, z[i:i + bs], zg[i:i + bs], Sinv=Sinv,
                                 K_list=args.K, M_list=args.M)
        for k, v in S.items():
            S_all.setdefault(k, []).append(v.float().cpu().numpy())
    torch.cuda.synchronize()
    print(f'  scores: {time.time() - t0:.1f}s for {G * C} candidates')
    S_all = {k: np.concatenate(v).reshape(G, C) for k, v in S_all.items()}
    np.savez(os.path.join(out_dir, 'scores.npz'), **S_all)
    json.dump(Cost, open(os.path.join(out_dir, 'cost.json'), 'w'), indent=1)
    return S_all, Cost


# ----------------------------------------------------------------- stage C
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def pair_stats(score, v, strat, margin):
    """Per-group arrays of (correct, category) over informative pairs."""
    G, C = v.shape
    iu = np.triu_indices(C, 1)
    cats = {'free-free': [], 'mixed': [], 'contactish': [], 'all': []}
    for g in range(G):
        dv = v[g][iu[0]] - v[g][iu[1]]
        ds = score[g][iu[0]] - score[g][iu[1]]
        m = np.abs(dv) > margin
        corr = (np.sign(dv) == np.sign(ds))[m]
        s1, s2 = strat[g][iu[0]][m], strat[g][iu[1]][m]
        ff = (s1 == sim.FREE) & (s2 == sim.FREE)
        cc = (s1 != sim.FREE) & (s2 != sim.FREE)
        cats['all'].append((corr.sum(), len(corr)))
        cats['free-free'].append((corr[ff].sum(), ff.sum()))
        cats['contactish'].append((corr[cc].sum(), cc.sum()))
        cats['mixed'].append((corr[~ff & ~cc].sum(), (~ff & ~cc).sum()))
    return {k: np.array(x, dtype=float) for k, x in cats.items()}


def l2_wrong_pairs(score, l2, v, margin):
    """Accuracy on pairs that L2 gets wrong (rescue rate)."""
    G, C = v.shape
    iu = np.triu_indices(C, 1)
    out = []
    for g in range(G):
        dv = v[g][iu[0]] - v[g][iu[1]]
        m = (np.abs(dv) > margin) & (np.sign(dv) != np.sign(l2[g][iu[0]] - l2[g][iu[1]]))
        ds = score[g][iu[0]] - score[g][iu[1]]
        out.append(((np.sign(dv) == np.sign(ds))[m].sum(), m.sum()))
    return np.array(out, dtype=float)


def boot(per_group, B=2000, seed=0):
    """per_group: (G,2) [correct, n]. Returns mean acc and 95% CI."""
    rng = np.random.default_rng(seed)
    G = len(per_group)
    acc = per_group[:, 0].sum() / max(per_group[:, 1].sum(), 1)
    bs = []
    for _ in range(B):
        i = rng.integers(0, G, G)
        bs.append(per_group[i, 0].sum() / max(per_group[i, 1].sum(), 1))
    return acc, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def stage_c(data, S, Cost, out_dir, margin):
    v, strat = data['vstar'], data['strat']
    G, C = v.shape
    lines = []
    P = lines.append
    P(f'groups={G} candidates/group={C}  strata counts: ' +
      ', '.join(f'{sim.STRATA[k]}={int((strat == k).sum())}' for k in sim.STRATA))
    P(f'V* mean={v.mean():.2f}  d_now mean={data["d_now"].mean():.2f}  pair margin={margin}')
    rows = []
    l2 = S['l2']
    for name, sc in S.items():
        rho = np.nanmean([spearman(sc[g], v[g]) for g in range(G)])
        ps = pair_stats(sc, v, strat, margin)
        regret = np.mean([v[g][np.argmin(sc[g])] - v[g].min() for g in range(G)])
        rw = l2_wrong_pairs(sc, l2, v, margin)
        rows.append(dict(name=name, cost=Cost.get(name, 0), spearman=rho, regret=regret,
                         **{k: boot(x) for k, x in ps.items()},
                         rescue=rw[:, 0].sum() / max(rw[:, 1].sum(), 1)))
    rows.sort(key=lambda r: -r['all'][0])
    hdr = f'{"score":34s} {"cost":>5s} {"spear":>6s} {"regret":>6s} ' \
          f'{"all":>17s} {"free-free":>17s} {"mixed":>17s} {"contactish":>17s} {"L2wrong":>7s}'
    P(hdr); P('-' * len(hdr))
    f = lambda t: f'{t[0]:.3f}[{t[1]:.3f},{t[2]:.3f}]'
    for r in rows:
        P(f'{r["name"]:34s} {r["cost"]:5d} {r["spearman"]:6.3f} {r["regret"]:6.2f} '
          f'{f(r["all"]):>17s} {f(r["free-free"]):>17s} {f(r["mixed"]):>17s} '
          f'{f(r["contactish"]):>17s} {r["rescue"]:7.3f}')
    # paired bootstrap: each score vs the best pure-lookahead baseline at <= its cost
    P('\nPaired bootstrap Δacc (all pairs) vs strongest lookahead baseline with cost <= own cost')
    base_names = [n for n in S if n.startswith(('drift', 'probemin', 'lookcem')) and '+' not in n]
    rng = np.random.default_rng(1)
    idx = [rng.integers(0, G, G) for _ in range(1000)]
    pg = {n: pair_stats(S[n], v, strat, margin) for n in S}
    def acc(n, cat, ii=None):
        x = pg[n][cat] if ii is None else pg[n][cat][ii]
        return x[:, 0].sum() / max(x[:, 1].sum(), 1)
    for r in rows[:15]:
        n = r['name']
        if n in base_names or n in ('l2', 'wl2'):
            continue
        cands = [b for b in base_names if Cost[b] <= max(Cost[n], 1)] or ['l2']
        b = max(cands, key=lambda b: acc(b, 'all'))
        for cat in ('all', 'contactish', 'free-free'):
            d = [acc(n, cat, ii) - acc(b, cat, ii) for ii in idx]
            P(f'  {n:30s} vs {b:22s} [{cat:10s}] Δ={acc(n, cat) - acc(b, cat):+.3f} '
              f'CI[{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]')
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(out_dir, 'report.txt'), 'w').write(txt)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/exp1')
    ap.add_argument('--stage', default='ABC')
    ap.add_argument('--groups', type=int, default=200)
    ap.add_argument('--C', type=int, default=32)
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--L', type=int, default=25, help='env steps of continuation for V*')
    ap.add_argument('--n', type=int, default=96, help='sim CEM samples')
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--K', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--M', type=int, nargs='+', default=[16, 64])
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--margin', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = dict(seed=args.seed, groups=args.groups, C=args.C, C_expert=args.C * 3 // 4,
               max_noise=2.0, plan_steps=25, L=args.L, n=args.n, iters=args.iters,
               elite=max(4, args.n // 8), workers=args.workers)
    json.dump(dict(cfg, **vars(args)), open(os.path.join(args.out, 'config.json'), 'w'), indent=1)
    if 'A' in args.stage:
        data = stage_a(cfg, args.out)
    else:
        data = dict(np.load(os.path.join(args.out, 'sim.npz')))
    if 'B' in args.stage:
        S, Cost = stage_b(data, args.out, args)
    else:
        S = dict(np.load(os.path.join(args.out, 'scores.npz')))
        Cost = json.load(open(os.path.join(args.out, 'cost.json')))
    if 'C' in args.stage:
        stage_c(data, S, Cost, args.out, args.margin)
