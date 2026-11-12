"""Metro-WM-style sub-goals (Lee et al., arXiv 2609.31868), re-implemented on our graph nodes for a matched comparison.

From the paper (Sec. 5, Alg. 1) and the released metro/graph.py:
  vertices   encoded recorded frames; the query's own episode is removed (done by the caller)
  temporal   directed edge frame -> next frame of the episode, cost = the frames between them
  Lambda(d)  the alpha-quantile (alpha = 0.25) of the latent distance between two frames d apart in one episode, made
             non-decreasing in d; kappa(gap) reads a latent gap back as frames (linear between the knots, at least 1 frame)
  bridges    undirected edges to the k = 4 nearest vertices of OTHER episodes that lie within eps = Lambda(H), cost kappa(gap)
  goal       a new vertex joined to its k nearest vertices within eps (the nearest one if none is that close)
  J(u)       cost of the cheapest route from u to the goal (Dijkstra backward from the goal)
  attach     the current latent goes to the vertex u0 among its k nearest within eps that minimises kappa(gap) + J(u0)
  sub-goal   the goal if J <= H; else walk the route from u0 while the summed cost (attachment included) stays <= H, the first
             move always taken; the last vertex reached is the sub-goal.  H is the waypoint budget in frames.

Differences from the paper, on purpose:
  * our vertices are every `stride`-th frame (5 = one action block), not every frame; a temporal edge costs `stride` frames and
    Lambda is measured at d = stride, 2 stride, ... with a knot (0 frames, 0 gap) added below the first
  * when no route exists the sub-goal is the goal (the paper first tries the neighbour nearest the goal)
Distances are L2 (not squared).  The class holds no model and runs on the device of `emb`.
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.csgraph import dijkstra


class MetroGraph:
    def __init__(self, emb, ep, step, k=4, alpha=0.25, budget=8.0, stride=5, chunk=1024, pairs=500_000):
        self.emb, self.k, self.alpha, self.budget, self.stride = emb, int(k), float(alpha), float(budget), int(stride)
        n, dev = len(emb), emb.device
        self.n = n
        order = np.lexsort((step, ep))
        eo, so = ep[order], step[order]
        # Lambda at d = stride, 2 stride, ... up to one knot beyond the budget
        lat, frm, rng = [0.0], [0.0], np.random.default_rng(0)
        for j in range(1, int(np.ceil(self.budget / self.stride)) + 2):
            ok = np.nonzero((eo[:-j] == eo[j:]) & (so[j:] - so[:-j] == self.stride * j))[0]
            if len(ok) == 0:
                break
            if len(ok) > pairs:
                ok = np.sort(rng.choice(ok, pairs, replace=False))
            ia, ib = torch.as_tensor(order[ok], device=dev), torch.as_tensor(order[ok + j], device=dev)
            q = float(np.quantile((emb[ia] - emb[ib]).float().norm(dim=-1).cpu().numpy(), self.alpha))
            lat.append(max(q, lat[-1] * (1 + 1e-6) + 1e-12))  # running maximum, kept strictly increasing for the interpolation
            frm.append(float(self.stride * j))
        assert len(lat) > 1, 'no two consecutive nodes of one episode: Lambda cannot be measured'
        self.lat, self.frm = np.asarray(lat), np.asarray(frm)
        self.eps = float(np.interp(self.budget, self.frm, self.lat))  # bridge radius Lambda(H)
        # temporal edges
        a, b = order[:-1], order[1:]
        nxt = (ep[a] == ep[b]) & (step[b] - step[a] == self.stride)
        ta, tb = a[nxt], b[nxt]
        T = csr_matrix((np.full(len(ta), float(self.stride), np.float32), (ta, tb)), shape=(n, n))
        # bridges: k nearest vertices of other episodes within eps
        ep_t = torch.as_tensor(np.unique(ep, return_inverse=True)[1], device=dev)
        rows, cols, ws = [], [], []
        for s in range(0, n, chunk):
            dd = torch.cdist(emb[s:s + chunk].float(), emb.float())
            dd[ep_t[s:s + chunk, None] == ep_t[None]] = float('inf')
            v, j = dd.topk(min(self.k, n), largest=False)
            v, j = v.cpu().numpy(), j.cpu().numpy()
            ok = v <= self.eps
            rows.append(np.repeat(np.arange(s, s + len(v))[:, None], v.shape[1], 1)[ok])
            cols.append(j[ok])
            ws.append(self.cost(v[ok]))
            del dd
        r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(ws).astype(np.float32)
        B = csr_matrix((w, (r, c)), shape=(n, n))
        B = B.maximum(B.T)  # undirected, each bridge with one cost
        G = (T + B).tocsr()
        self.n_temporal, self.n_bridge = len(ta), B.nnz // 2
        # transposed graph with an empty last row and column for the goal vertex (filled per goal)
        self._top = hstack([G.T.tocsr(), csr_matrix((n, 1), dtype=np.float32)]).tocsr()

    def cost(self, gap):
        """kappa: a latent gap read as frames, between 1 and the budget"""
        return np.clip(np.interp(gap, self.lat, self.frm), 1.0, max(1.0, self.budget))

    def _near(self, z):
        """the k nearest vertices within eps (the nearest one if none is that close): indices, attachment costs"""
        v, j = (self.emb.float() - z[None].float()).norm(dim=-1).topk(min(self.k, self.n), largest=False)
        v, j = v.cpu().numpy(), j.cpu().numpy()
        ok = v <= self.eps
        if not ok.any():
            ok[0] = True
        return j[ok], self.cost(v[ok])

    def goal_tree(self, g):
        """(J, next vertex toward the goal, vertex nearest the goal); the goal itself is vertex n"""
        us, w = self._near(g)
        row = csr_matrix((w.astype(np.float32), (np.zeros(len(us), np.int64), us)), shape=(1, self.n + 1))
        dist, pred = dijkstra(vstack([self._top, row]).tocsr(), directed=True, indices=self.n, return_predecessors=True)
        return dist, pred, int(us[0])

    def subgoal(self, z0, tree):
        """(sub-goal vertex or None = the goal itself, attachment vertex u0, J of the current latent in frames)"""
        dist, pred, _ = tree
        us, c = self._near(z0)
        tot = c + dist[us]
        i = int(np.argmin(tot))
        u0, j0 = int(us[i]), float(tot[i])
        if not np.isfinite(j0) or j0 <= self.budget + 1e-6:  # no route, or the goal is within one budget
            return None, u0, j0
        spent, cur, first = float(c[i]), u0, True
        while True:
            nx = int(pred[cur])
            if nx < 0:
                break
            w = float(dist[cur] - dist[nx])
            if not first and spent + w > self.budget + 1e-6:
                break
            spent, cur, first = spent + w, nx, False
            if cur == self.n:
                return None, u0, j0
        return cur, u0, j0
