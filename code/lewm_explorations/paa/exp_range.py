"""exp-20: how far does the latent goal cost point the way, for each world model, against a model-independent distance?
The reference graph uses true positions only: temporal edges + edges between nodes closer than the median recorded
five-step move (k=10 nearest in position).  For goal g and state v at true graph distance h, rho(h) is the share of pairs
of v's neighbours whose latent order to g agrees with their true order (0.5 = chance).  Also the Spearman correlation of
the latent cost with true graph distance over all nodes (global) and within 6 hops, and one cost map for the figure.
Concentration: median squared latent distance per hop against 2 x total variance, the value for two unrelated samples.
usage: python exp_range.py --out res.npz --n_goals 200 name1=nodes1.npz name2=nodes2.npz ...   (first file gives the states)"""
import argparse, json
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from exp_ref import ref_graph

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--n_goals', type=int, default=200); ap.add_argument('nodes', nargs='+')
args = ap.parse_args()
files = dict(x.split('=') for x in args.nodes)
ref = np.load(next(iter(files.values())))
xy, ep, step = ref['state'][:, :2].astype(np.float32), ref['ep'], ref['step']
emb = {k: np.load(v)['emb'].astype(np.float32) for k, v in files.items()}
n = len(xy); assert all(len(e) == n for e in emb.values()), 'all models must encode the same frames'
G = ref_graph(xy, ep, step); GT = G.T.tocsr()
rng = np.random.default_rng(0); goals = rng.choice(n, args.n_goals, replace=False)
H = list(range(1, 41)); agree = {k: {h: [0, 0] for h in H} for k in emb}; sp_all = {k: [] for k in emb}; sp_loc = {k: [] for k in emb}
cost_map = None
d2 = {k: {h: [] for h in H} for k in emb}
for gi, g in enumerate(goals):
    dg = dijkstra(GT, directed=True, indices=int(g)); fin = np.isfinite(dg)
    if fin.sum() < 100:
        continue
    lat = {k: ((e - e[g]) ** 2).sum(-1) for k, e in emb.items()}
    if cost_map is None and dg[fin].max() > 30:
        cost_map = dict(goal=int(g), dg=dg.astype(np.float32), **{f'lat_{k}': v.astype(np.float32) for k, v in lat.items()})
    s = rng.choice(np.nonzero(fin)[0], min(4000, int(fin.sum())), replace=False)
    for k in emb:
        for h in H:
            m = dg[s] == h
            if m.any():
                d2[k][h].append(float(np.median(lat[k][s][m])))
        sp_all[k].append(spearmanr(dg[s], lat[k][s]).statistic)
        near = s[dg[s] <= 6]
        if len(near) > 20:
            sp_loc[k].append(spearmanr(dg[near], lat[k][near]).statistic)
    for h in H:
        cand = np.nonzero(dg == h)[0]
        if len(cand) == 0:
            continue
        for v in rng.choice(cand, min(4, len(cand)), replace=False):
            nb = G.indices[G.indptr[v]:G.indptr[v + 1]]; nb = nb[np.isfinite(dg[nb])]
            if len(nb) < 2:
                continue
            i, j = np.triu_indices(len(nb), 1); m = dg[nb][i] != dg[nb][j]
            if not m.any():
                continue
            for k in emb:
                good = np.sign(dg[nb][i][m] - dg[nb][j][m]) == np.sign(lat[k][nb][i][m] - lat[k][nb][j][m])
                agree[k][h][0] += int(good.sum()); agree[k][h][1] += int(m.sum())
    if gi % 50 == 0:
        print(gi, flush=True)
res = {k: dict(hops=H, rho=[agree[k][h][0] / agree[k][h][1] if agree[k][h][1] else None for h in H],
               pairs=[agree[k][h][1] for h in H], spearman_all=float(np.nanmean(sp_all[k])),
               spearman_6hops=float(np.nanmean(sp_loc[k])) if sp_loc[k] else None,
               d2_hop=[float(np.median(d2[k][h])) if d2[k][h] else None for h in H],
               two_var=float(2 * emb[k].var(0).sum()), dim=int(emb[k].shape[1])) for k in emb}
print(json.dumps({k: dict(rho1=v['rho'][0], rho3=v['rho'][2], rho6=v['rho'][5], rho12=v['rho'][11], sp=v['spearman_all'],
                          sp6=v['spearman_6hops'], d2_h40=v['d2_hop'][-1], two_var=v['two_var']) for k, v in res.items()}), flush=True)
json.dump(res, open(args.out + '.json', 'w'))
if cost_map:
    s = rng.choice(n, min(20000, n), replace=False)
    np.savez_compressed(args.out, xy=xy[s], goal_xy=xy[cost_map['goal']], **{k: v[s] for k, v in cost_map.items() if k != 'goal'})
print('saved', args.out, flush=True)
