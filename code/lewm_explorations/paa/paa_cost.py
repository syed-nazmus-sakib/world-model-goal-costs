"""Planner-Attainable Anchors (PAA): a planning objective that aims CEM at an intermediate latent anchor instead of the
far goal.  Anchors are graph nodes (paa_nodes.py) on the shortest latent-graph path from the current state to the goal.

Graph: directed temporal edges (frame -> frame 5 steps later, same episode) plus symmetric k-NN edges between latents
closer than the median one-block temporal distance; every edge counts as one block.  Episodes listed by set_exclude()
are removed (the evaluation pairs' own episodes), so an anchor never replays the test trajectory.

Anchor choice per replan and per env:
  mode='fixed' : the node `span` hops along the path (training-free, Anchored-Planning-style)
  mode='ap'    : Anchored Planning (arXiv 2609.30036) re-implemented on the same nodes for a matched comparison:
                 retrieve the node whose (z, z_{+remaining}, difference) key is nearest to (z_t, z_goal, z_goal - z_t),
                 remaining = max(5, goal offset - elapsed steps), and aim at that node's one-block successor
  mode='apg'   : AP retrieval of the top `ap_k` keys, then the successor with the smallest graph distance to the goal
                 (ties: retrieval order); stitches across episodes where a single recorded segment does not reach the goal
  mode='paa'   : the hop maximising P_attain(z_t, z_anchor, hops) * (graph progress), P from a learned ensemble
                 (paa_head.py) minus `pess` times its std; hops with P < tau are skipped
When the goal is within `span` hops (or no path exists) the anchor is the goal itself.
Cost: squared latent distance to the anchor, min over the predicted steps (agg='min') or the last step.
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch import nn


class PAAGoalCost(nn.Module):
    def __init__(self, nodes, mode='fixed', span=4, max_hops=12, head=None, tau=0.5, pess=1.0, knn=10, agg='min',
                 thr_q=0.5, ap_offset=None, ap_step=5, ap_k=8):
        super().__init__()
        d = np.load(nodes)
        self.emb_all = torch.from_numpy(d['emb'].astype(np.float32))
        self.ep_all, self.step_all = d['ep'], d['step']
        self.mode, self.span, self.max_hops, self.tau, self.pess, self.knn, self.agg = mode, int(span), int(max_hops), \
            float(tau), float(pess), int(knn), agg
        self.thr_q = float(thr_q)
        self.ap_offset, self.ap_step, self.ap_k = ap_offset, int(ap_step), int(ap_k)
        self.head = None
        if head:
            from paa_head import load_head
            self.head = load_head(head)
        self.set_exclude([])
        print(f'[paa] {len(self.emb_all)} nodes, mode={mode} span={span} max_hops={max_hops} head={head} tau={tau}',
              flush=True)

    # ----------------------------------------------------------------------------------------------------- graph
    def set_exclude(self, eps):
        keep = ~np.isin(self.ep_all, np.asarray(list(eps)))
        self.idx = np.nonzero(keep)[0]
        self.emb = self.emb_all[self.idx].cuda()
        ep, step = self.ep_all[self.idx], self.step_all[self.idx]
        n = len(self.idx)
        order = np.lexsort((step, ep))
        if self.mode in ('ap', 'apg'):  # retrieval banks are built per remaining horizon
            self._ap_order, self._ap_ep, self._ap_st = order, ep[order], step[order]
            self._ap_bank, self._ap_env = {}, {}
            print(f'[paa] ap memory: {n} nodes ({len(eps)} episodes excluded)', flush=True)
            if self.mode == 'ap':  # no graph
                return
        a, b = order[:-1], order[1:]
        nxt = (ep[a] == ep[b]) & (step[b] - step[a] == 5)
        ta, tb = a[nxt], b[nxt]
        one = (self.emb[torch.from_numpy(ta).cuda()] - self.emb[torch.from_numpy(tb).cuda()]).pow(2).sum(-1)
        self.thr = float(torch.quantile(one.float(), self.thr_q))
        rows, cols = [ta], [tb]
        for s in range(0, n, 1024):  # k-NN edges under the one-block threshold
            dd = torch.cdist(self.emb[s:s + 1024], self.emb).pow(2)
            dd[torch.arange(dd.shape[0]), torch.arange(s, s + dd.shape[0])] = float('inf')
            v, j = dd.topk(self.knn, largest=False)
            ok = (v < self.thr).cpu().numpy()
            r = np.repeat(np.arange(s, s + dd.shape[0])[:, None], self.knn, 1)
            rows += [r[ok], j.cpu().numpy()[ok]]
            cols += [j.cpu().numpy()[ok], r[ok]]
        r, c = np.concatenate(rows), np.concatenate(cols)
        G = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n))
        G.data[:] = 1.0
        self.GT = G.T.tocsr()  # distances TO the goal = dijkstra from the goal on the transposed graph
        del dd
        torch.cuda.empty_cache()  # several evaluation processes share one GPU
        self._goal_cache, self._anchor_cache = {}, {}
        print(f'[paa] graph: {n} nodes ({len(eps)} episodes excluded), {len(ta)} temporal + {G.nnz - len(ta)} k-NN edges, '
              f'thr={self.thr:.3f}', flush=True)

    def _nearest(self, z):
        return int((self.emb - z[None]).pow(2).sum(-1).argmin())

    def _goal_tree(self, g):
        key = hash(g.cpu().numpy().tobytes())
        if key not in self._goal_cache:
            gn = self._nearest(g)
            dist, pred = dijkstra(self.GT, directed=True, indices=gn, return_predecessors=True)
            self._goal_cache[key] = (gn, dist, pred)
        return self._goal_cache[key]

    def _path(self, start, gn, pred, hops):
        path, n = [], start
        for _ in range(hops):
            if n == gn or n < 0:
                break
            n = int(pred[n])  # predecessor on the transposed graph = next hop toward the goal
            if n < 0:
                break
            path.append(n)
        return path

    def anchor(self, z0, g):
        key = (hash(z0.cpu().numpy().tobytes()), hash(g.cpu().numpy().tobytes()))
        if key in self._anchor_cache:
            return self._anchor_cache[key]
        gn, dist, pred = self._goal_tree(g)
        s = self._nearest(z0)
        out = g
        if np.isfinite(dist[s]) and dist[s] > self.span:
            path = self._path(s, gn, pred, self.max_hops)
            if path:
                if self.mode == 'fixed' or self.head is None:
                    out = self.emb[path[min(self.span, len(path)) - 1]]
                else:
                    za = self.emb[torch.as_tensor(path, device=self.emb.device)]
                    hops = torch.arange(1, len(path) + 1, device=za.device, dtype=torch.float32)
                    mu, sd = self.head(z0[None].expand(len(path), -1), za, hops)
                    p = (mu - self.pess * sd).clamp(0, 1)
                    gain = torch.as_tensor(dist[s] - dist[np.asarray(path)], device=za.device, dtype=torch.float32)
                    score = torch.where(p >= self.tau, p * gain, torch.full_like(p, -1.0))
                    k = int(score.argmax()) if bool((score > 0).any()) else 0
                    out = za[k]
        self._anchor_cache[key] = out
        return out

    # ------------------------------------------------------------------------------------------- anchored planning
    def _ap_bank_for(self, k):
        if k not in self._ap_bank:
            self._ap_bank.clear()  # one remaining-horizon bank at a time (about 1 GB each)
            torch.cuda.empty_cache()
            o, ep, st = self._ap_order, self._ap_ep, self._ap_st
            pos = np.arange(len(o) - k)
            ok = (ep[pos + k] == ep[pos]) & (st[pos + k] - st[pos] == 5 * k) & (ep[pos + 1] == ep[pos]) & \
                 (st[pos + 1] - st[pos] == 5)
            src, far, nxt = (torch.from_numpy(o[pos[ok] + j]).cuda() for j in (0, k, 1))
            z, zg = self.emb[src], self.emb[far]
            f = torch.cat([z, zg, zg - z], -1)
            mu, sd = f.mean(0), f.std(0).clamp_min(1e-4)
            f = (f - mu) / sd
            self._ap_bank[k] = (f, f.pow(2).sum(-1), mu, sd, nxt)
            del z, zg
            torch.cuda.empty_cache()
        return self._ap_bank[k]

    def _ap_anchor(self, z0, g):
        out = []
        for b in range(len(z0)):  # elapsed steps per env: +ap_step every time its current latent changes (one decision)
            gk, zk = hash(g[b].cpu().numpy().tobytes()), hash(z0[b].cpu().numpy().tobytes())
            st = self._ap_env.setdefault(gk, {'z': zk, 't': 0, 'a': None})
            if st['z'] != zk:
                st['z'], st['t'], st['a'] = zk, st['t'] + self.ap_step, None
            if st['a'] is None:
                k = max(1, int(round(max(5, self.ap_offset - st['t']) / 5)))
                f, f2, mu, sd, nxt = self._ap_bank_for(k)
                q = (torch.cat([z0[b], g[b], g[b] - z0[b]]) - mu) / sd
                if self.mode == 'ap':
                    i = int((f2 - 2 * (f @ q)).argmin())
                else:
                    top = (f2 - 2 * (f @ q)).topk(self.ap_k, largest=False).indices
                    _, dist, _ = self._goal_tree(g[b])
                    i = int(top[int(np.argmin(dist[nxt[top].cpu().numpy()]))])  # argmin keeps retrieval order on ties
                st['a'] = self.emb[nxt[i]]
            out.append(st['a'])
        return torch.stack(out)

    # ----------------------------------------------------------------------------------------------------- cost
    def forward(self, info_dict):
        g = info_dict['goal_emb'][:, -1].float()  # (B, D)
        H = info_dict['pixels'].size(2)
        z0 = info_dict['predicted_emb'][:, 0, H - 1].float()  # current frame (last context step)
        if self.mode in ('ap', 'apg'):
            a = self._ap_anchor(z0, g).to(g.device)
        else:
            a = torch.stack([self.anchor(z0[b], g[b]) for b in range(g.shape[0])]).to(g.device)
        if self.agg == 'min':
            p = info_dict['predicted_emb'][:, :, H:].float()
            return (p - a[:, None, None]).pow(2).sum(-1).min(-1).values
        p = info_dict['predicted_emb'][:, :, -1].float()
        return (p - a[:, None]).pow(2).sum(-1)
