"""exp-24 / exp-25: test-time graph search with a learned value function, on top of a frozen offline policy.
No world model anywhere: both the graph and the subgoals come from the policy's own value head.

Distance (TTGS, Opryshko et al. 2026): expected steps from V, log(V(1 - gamma) + 1) / log(gamma), in training-clip units.
mode=value (exp-24, TTGS-style): V is GCIVL's learned value.  mode=hilp (exp-25, GAS-style, Baek et al. 2025): V is
-||phi(s) - phi(g)|| of HILP, and nodes are thinned by coverage in phi space (GAS's key nodes without its TE filter).
Graph: n nodes drawn from the dataset's 5-step grid; edges with distance < tau among each node's `cand` nearest neighbours in
image features (all pairs for hilp; TTGS scores all pairs, which a transformer value head cannot afford here) + temporal edges.
Path: Dijkstra once from the node nearest the start to the node nearest the goal.  Subgoal, every `every` steps: furthest
path node within threshold = 2 tau (TTGS's ratio), the real goal once it is within threshold.  Replans when the subgoal
drifts past replan x threshold (off when replan <= 0).  Same pairs, budget and success test as every other method."""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from graph_policy import GraphWaypointPolicy


class ValueGraph:
    def __init__(self, model, nodes_npz, dataset, prepare, mode='value', tau=4.0, n_nodes=2000, cand=50, gamma=0.99, seed=0, batch=256):
        self.model, self.mode, self.tau, self.gamma, self.batch = model, mode, float(tau), float(gamma), int(batch)
        self.device = next(model.parameters()).device
        d = np.load(nodes_npz)
        ep, step = d['ep'], d['step']
        pick = np.sort(np.random.default_rng(seed).choice(len(ep), min(int(n_nodes), len(ep)), replace=False))
        self.ep_all, self.step_all = ep[pick], step[pick]
        E = []
        for i in range(0, len(pick), batch):
            E.append(self._embed_rows(dataset, prepare, pick[i:i + batch], ep, step).cpu())
        self.emb, self.cand, self.G = torch.cat(E), int(cand), None                   # (N, P, D) in the policy's own embedding

    def set_exclude(self, eps):
        """Drop the evaluation pairs' episodes, as PAAGoalCost does, then build the graph: no plan can replay a test episode."""
        keep = ~np.isin(self.ep_all, np.asarray(list(eps)))
        self.ep_all, self.step_all, self.emb = self.ep_all[keep], self.step_all[keep], self.emb[keep]
        if self.mode == 'hilp':
            k = self._cover(self._phi(self.emb), 0.5 * self.tau)
            if len(k) >= 50:   # below that phi has collapsed (untrained model): keep every node rather than a useless graph
                self.ep_all, self.step_all, self.emb = self.ep_all[k], self.step_all[k], self.emb[k]
        self._build(self.cand)

    def _embed_rows(self, dataset, prepare, idx, ep, step):
        rows = dataset.load_chunk(ep[idx], step[idx], step[idx] + 1)
        info = {}
        for col in ('pixels', 'proprio'):
            if col in rows[0]:
                v = [r[col][-1:] for r in rows]
                v = torch.stack(v) if torch.is_tensor(v[0]) else torch.as_tensor(np.stack(v))
                info[col] = (v.permute(0, 1, 3, 4, 2) if col == 'pixels' else v).numpy()
        return self._encode(prepare(info))[:, 0]

    @torch.no_grad()
    def _encode(self, p, pixels_key='pixels', prefix=None):
        p = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in p.items()}
        return self.model.encode(p, pixels_key=pixels_key, prefix=prefix, target='e_tmp')['e_tmp'].float()   # (B, T, P, d)

    def _vp(self):
        v = self.model.value_predictor
        return getattr(v, 'teacher', v)

    @torch.no_grad()
    def _phi(self, e):
        return torch.cat([self._vp().embed(e[i:i + 512].to(self.device).float())[:, 0].cpu() for i in range(0, len(e), 512)])

    def _steps(self, v):
        lo = -1 / (1 - self.gamma) + 1e-3
        s = torch.log(v.clamp(lo, -1e-3) * (1 - self.gamma) + 1) / np.log(self.gamma)
        return s.clamp_min(0).nan_to_num(1e4)

    @torch.no_grad()
    def dist(self, a, b):
        """expected steps from (N, P, d) a to (N, P, d) b, paired row by row (N, ...) on self.device."""
        if self.mode == 'hilp':
            pa, pb = self._vp().embed(a.float()), self._vp().embed(b.float())
            return self._steps(-(pa - pb).pow(2).sum(-1).add(1e-6).sqrt())[:, 0]
        return self._steps(self.model.predict_values(a[:, None], b[:, None]).squeeze(-1))[:, -1]

    def _dist_many(self, a, b):
        out = [self.dist(a[i:i + self.batch].to(self.device), b[i:i + self.batch].to(self.device)).cpu() for i in range(0, len(a), self.batch)]
        return torch.cat(out)

    @staticmethod
    def _cover(phi, r):
        keep, kept = [], torch.zeros(0, phi.shape[1])
        for i in range(len(phi)):
            if len(kept) == 0 or torch.cdist(phi[i:i + 1], kept).min() > r:
                keep.append(i); kept = torch.cat([kept, phi[i:i + 1]])
        return np.array(keep)

    def _build(self, cand):
        n = len(self.emb)
        f = self.emb.mean(1); f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        k = min(cand, n - 1)
        idx = torch.cat([torch.cdist(f[i:i + 512], f).topk(k + 1, largest=False).indices[:, 1:] for i in range(0, n, 512)])
        ai = torch.arange(n)[:, None].expand(-1, k).reshape(-1); bi = idx.reshape(-1)
        d = torch.cat([self._dist_many(self.emb[ai[i:i + 8192]], self.emb[bi[i:i + 8192]]) for i in range(0, len(ai), 8192)]).numpy()
        m = d < self.tau
        o = np.lexsort((self.step_all, self.ep_all)); x, y = o[:-1], o[1:]
        nx = self.ep_all[x] == self.ep_all[y]
        tx, ty = x[nx], y[nx]
        temporal = np.maximum(1.0, (self.step_all[ty] - self.step_all[tx]) / 5.0)
        self.G = csr_matrix((np.concatenate([np.maximum(d[m], 1.0), temporal]), (np.concatenate([ai.numpy()[m], tx]), np.concatenate([bi.numpy()[m], ty]))), shape=(n, n))
        print(f'[value_graph] {self.mode}: {n} nodes, {m.sum()} value edges of {len(d)} candidates (median {np.median(d):.1f}), '
              f'{len(tx)} temporal, tau {self.tau}', flush=True)

    def path(self, dist_from, dist_to):
        """shortest node path from the node nearest the start to the node nearest the goal; [] when they are not connected."""
        s, t = int(dist_from.argmin()), int(dist_to.argmin())
        D, pred = dijkstra(self.G, directed=True, indices=s, return_predecessors=True)
        if not np.isfinite(D[t]):
            return []
        p = [t]
        while p[-1] != s:
            p.append(int(pred[p[-1]]))
        return p[::-1]


class ValueGraphPolicy(GraphWaypointPolicy):
    def __init__(self, inner, vg, dataset, every=5, threshold=None, replan=0.0):
        self.vg_ = vg
        super().__init__(inner, None, vg, dataset, None, every=every)
        self.type = 'value_graph'
        self.thr = float(threshold if threshold is not None else 2 * vg.tau)
        self.replan = float(replan)
        self._paths = None

    def _obs_goal(self, info_dict):
        p = self.inner._prepare_info(info_dict)
        return self.vg_._encode(p)[:, -1], self.vg_._encode(p, 'goal', 'goal_')[:, -1]

    def _dists(self, e, nodes):  # one embedding -> list of nodes, both directions
        v = self.vg_
        a = e[None].expand(len(nodes), -1, -1).cpu()
        return v._dist_many(a, v.emb[nodes]).numpy(), v._dist_many(v.emb[nodes], a).numpy()

    def get_action(self, info_dict, **kwargs):
        if self._wp is None or self._t % self.every == 0:
            zs, zg = self._obs_goal(info_dict)
            n = len(zs)
            if self._paths is None:
                self._paths, self._wp = [None] * n, [-1] * n
            if self.vg_.G is None:          # +paa_noexclude: nothing asked for the exclusion, so nothing built the graph
                self.vg_.set_exclude([])
            allnodes = np.arange(len(self.vg_.emb))
            for b in range(n):
                dgo = float(self.vg_.dist(zs[b:b + 1], zg[b:b + 1]))
                if self._paths[b] is None:
                    fwd = self._dists(zs[b], allnodes)[0]
                    self._paths[b] = self.vg_.path(fwd, self._dists(zg[b], allnodes)[1])
                path = self._paths[b]
                if dgo <= self.thr or not path:
                    self._wp[b] = -1
                else:
                    d = self._dists(zs[b], np.array(path))[0]
                    ok = np.nonzero(d < self.thr)[0]
                    i = int(ok[-1]) if len(ok) else int(d.argmin())
                    if self.replan > 0 and d[i] > self.replan * self.thr:
                        self._paths[b] = None                                       # next refresh searches again
                    self._wp[b] = int(path[i])
                self.log.append((self._t, b, self._wp[b]))
        self._t += 1
        info, copied = dict(info_dict), set()
        for b, node in enumerate(self._wp):
            if node < 0:
                continue
            for k, v in self._row(node).items():
                if k not in info:
                    continue
                if k not in copied:
                    info[k] = np.array(info[k], copy=True); copied.add(k)
                info[k][b][...] = v
        return self.inner.get_action(info, **kwargs)
