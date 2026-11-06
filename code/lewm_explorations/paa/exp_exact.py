"""exp-28: the cost alone, with the predictor and the search removed.  CPU, needs only encoded nodes.
Exact dynamics: from node v, the planner may reach exactly the recorded nodes within R reference-graph hops (R = H = 10 blocks).
Exact search: it takes the reachable node with the smallest squared latent distance to the target, moves `step` hops
towards it (receding horizon, 5 blocks) and repeats.  Budget: as the real planner, 2 x the shortest path, in the same blocks.
target=goal: the latent goal cost.  target=anchor: K hops along the latent graph (temporal + 10-NN under tau, as in the paper).
If the goal cost still fails here, neither the predictor nor CEM is the cause.  Killed if goal reaches the success of anchors.
Success: within 2 hops of the goal.  usage: python exp_exact.py --out res.json [--pairs 300] name=nodes.npz ... (first gives states)"""
import argparse, json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
from exp_ref import ref_graph

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--pairs', type=int, default=300); ap.add_argument('--K', type=int, default=6)
ap.add_argument('--R', type=int, default=10); ap.add_argument('--step', type=int, default=5); ap.add_argument('nodes', nargs='+')
a = ap.parse_args()
files = dict(x.split('=') for x in a.nodes)
ref = np.load(next(iter(files.values())))
xy, ep, step = ref['state'][:, :2].astype(np.float32), ref['ep'], ref['step']
G = ref_graph(xy, ep, step); GT = G.T.tocsr(); n = len(xy)


def latent_graph(z):
    o = np.lexsort((step, ep)); x, y = o[:-1], o[1:]; nx = (ep[x] == ep[y]) & (step[y] - step[x] == 5)
    tx, ty = x[nx], y[nx]
    thr = float(np.median(((z[tx] - z[ty]) ** 2).sum(-1)))
    d, idx = NearestNeighbors(n_neighbors=11).fit(z).kneighbors(z)
    ok = d[:, 1:] ** 2 < thr; ri = np.repeat(np.arange(n)[:, None], 10, 1)
    r = np.concatenate([tx, ri[ok], idx[:, 1:][ok]]); c = np.concatenate([ty, idx[:, 1:][ok], ri[ok]])
    g = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n)); g.data[:] = 1.0
    return g


def run(z, LG, s, g, dg, T, mode):
    v = int(s)
    if mode == 'anchor':
        _, lp = dijkstra(LG.T.tocsr(), directed=True, indices=int(g), return_predecessors=True)   # tree towards the goal
    for _ in range(T):
        if dg[v] <= 2:
            return True
        target = z[g]
        if mode == 'anchor':
            u, h = v, 0
            while h < a.K and lp[u] >= 0:
                u = int(lp[u]); h += 1
            target = z[u]
        D, pred = dijkstra(G, directed=True, indices=v, limit=a.R, return_predecessors=True)
        ball = np.nonzero(np.isfinite(D))[0]
        u = int(ball[((z[ball] - target) ** 2).sum(-1).argmin()])
        path = [u]
        while path[-1] != v:
            path.append(int(pred[path[-1]]))
        v = path[::-1][min(a.step, len(path) - 1)]
    return bool(dg[v] <= 2)


rng = np.random.default_rng(0); pairs = []
while len(pairs) < a.pairs:
    g = int(rng.integers(n)); dg = dijkstra(GT, directed=True, indices=g)
    c = np.nonzero(np.isfinite(dg) & (dg >= 5))[0]
    if len(c):
        for s in rng.choice(c, min(4, len(c)), replace=False):
            pairs.append((int(s), g, dg))
res = {}
BINS = [(5, 10), (10, 20), (20, 30), (30, 99)]
for name, f in files.items():
    z = np.load(f)['emb'].astype(np.float32); LG = latent_graph(z); res[name] = {}
    for mode in ('goal', 'anchor'):
        ok = {b: [] for b in BINS}
        for s, g, dg in pairs:
            h = dg[s]; b = next((b for b in BINS if b[0] <= h < b[1]), None)
            if b:
                ok[b].append(run(z, LG, s, g, dg, int(np.ceil(2 * h / a.step)) + 1, mode))
        res[name][mode] = {f'{b[0]}-{b[1]}': [float(np.mean(v)) if v else None, len(v)] for b, v in ok.items()}
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
