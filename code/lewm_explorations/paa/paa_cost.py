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
  mode='apk'   : AP with the successors of the top `ap_k` keys as targets; cost = distance to the nearest of them
  mode='apc'   : AP that commits to the retrieved record: while the current latent stays within one median block step
                 of the record's next frame, the target advances along that record instead of being re-retrieved
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
                 thr_q=0.5, ap_offset=None, ap_step=5, ap_k=8, gap_cal=False, frac=1.0, drop_box=None, switch=1.0,
                 handoff=False, handoff_hops=0):
        super().__init__()
        d = np.load(nodes)
        self.emb_all = torch.from_numpy(d['emb'].astype(np.float32))
        self.ep_all, self.step_all = d['ep'], d['step']
        self.mode, self.span, self.max_hops, self.tau, self.pess, self.knn, self.agg = mode, int(span), int(max_hops), \
            float(tau), float(pess), int(knn), agg
        self.thr_q = float(thr_q)
        self.ap_offset, self.ap_step, self.ap_k = ap_offset, int(ap_step), int(ap_k)
        self.gap_cal = bool(gap_cal)  # gap: convert graph hops to recorded blocks before choosing the retrieval bank
        self.frac = float(frac)  # ablation: build the graph from this fraction of the training episodes (seeded)
        # episode-switch cost: neighbour edges cost `switch` (temporal edges 1).  switch > 1 makes shortest paths follow
        # recorded trajectories for longer; the span K then counts temporal moves only (neighbour jumps are "same place")
        self.switch = float(switch)
        # goal handoff: a candidate whose plan comes within one recorded block of the goal (squared distance < thr, the graph's
        # own edge threshold) beats every candidate that does not; otherwise the anchor cost is unchanged
        self.handoff = bool(handoff)
        # gated handoff: only when the graph puts the goal at most `handoff_hops` hops from the current state (0 = no gate).
        # A goal can be close in latent space but far along the graph (behind a wall): there the handoff must not fire
        self.handoff_hops = float(handoff_hops)
        self._ho = {'n': 0, 'open': 0, 'fired': 0}  # decisions, decisions with the gate open, decisions where a candidate took it
        # coverage ablation: drop every node whose agent position (state[:, :2]) lies in the box (x0, x1, y0, y1)
        self.drop_box = None if drop_box in (None, 'none', '') else [float(v) for v in
                                                                     (drop_box.split(',') if isinstance(drop_box, str) else drop_box)]
        self.state_all = d['state'] if 'state' in d.files and d['state'].ndim == 2 else None
        self._t = {'build_s': 0.0, 'dijkstra_s': 0.0, 'dijkstra_n': 0, 'anchor_s': 0.0, 'anchor_n': 0}
        self.head = None
        if head:
            from paa_head import load_head
            self.head = load_head(head)
        self.set_exclude([])
        print(f'[paa] {len(self.emb_all)} nodes, mode={mode} span={span} max_hops={max_hops} head={head} tau={tau}',
              flush=True)

    # ----------------------------------------------------------------------------------------------------- graph
    def set_exclude(self, eps):
        import time as _time
        t_build = _time.perf_counter()
        keep = ~np.isin(self.ep_all, np.asarray(list(eps)))
        if self.drop_box is not None:
            assert self.state_all is not None, 'drop_box needs agent positions (state) in the nodes file'
            x0, x1, y0, y1 = self.drop_box
            xy = self.state_all[:, :2]
            inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
            keep &= ~inside
            print(f'[paa] drop_box {self.drop_box}: {int(inside.sum())} nodes removed', flush=True)
        if self.frac < 1.0:
            u = np.unique(self.ep_all)
            sub = np.random.default_rng(0).choice(u, size=max(1, int(round(self.frac * len(u)))), replace=False)
            keep &= np.isin(self.ep_all, sub)
        self.idx = np.nonzero(keep)[0]
        self.emb = self.emb_all[self.idx].cuda()
        ep, step = self.ep_all[self.idx], self.step_all[self.idx]
        n = len(self.idx)
        order = np.lexsort((step, ep))
        if self.mode in ('ap', 'apg', 'apk', 'apc', 'gap'):  # retrieval banks are built per remaining horizon
            self._ap_order, self._ap_ep, self._ap_st = order, ep[order], step[order]
            self._ap_bank, self._ap_env = {}, {}
            pos = np.arange(n - 1)
            cont = (self._ap_ep[pos + 1] == self._ap_ep[pos]) & (self._ap_st[pos + 1] - self._ap_st[pos] == 5)
            o = torch.from_numpy(order).cuda()
            self._ap_eps = float((self.emb[o[pos[cont]]] - self.emb[o[pos[cont] + 1]]).pow(2).sum(-1).median())
            self._ap_cont = np.append(cont, False)  # sorted position p continues to p + 1 in the same episode
            # longest span (blocks) any record can have: an offset beyond it is clamped instead of retrieving nothing
            starts = np.r_[0, np.nonzero(np.diff(self._ap_ep))[0] + 1]  # nodes are sorted by (episode, step)
            span = np.maximum.reduceat(self._ap_st, starts) - np.minimum.reduceat(self._ap_st, starts)
            self._ap_kmax = max(1, int(span.max()) // 5)
            del o
            print(f'[paa] ap memory: {n} nodes ({len(eps)} episodes excluded)', flush=True)
            if self.mode not in ('apg', 'gap'):  # no graph
                return
        a, b = order[:-1], order[1:]
        nxt = (ep[a] == ep[b]) & (step[b] - step[a] == 5)
        ta, tb = a[nxt], b[nxt]
        one = (self.emb[torch.from_numpy(ta).cuda()] - self.emb[torch.from_numpy(tb).cuda()]).pow(2).sum(-1)
        self.thr = float(torch.quantile(one.float(), self.thr_q))
        rows, cols, dd = [ta], [tb], None
        for s in range(0, n if self.knn > 0 else 0, 1024):  # k-NN edges under the one-block threshold (knn=0: temporal only)
            dd = torch.cdist(self.emb[s:s + 1024], self.emb).pow(2)
            dd[torch.arange(dd.shape[0]), torch.arange(s, s + dd.shape[0])] = float('inf')
            v, j = dd.topk(self.knn, largest=False)
            ok = (v < self.thr).cpu().numpy()
            r = np.repeat(np.arange(s, s + dd.shape[0])[:, None], self.knn, 1)
            rows += [r[ok], j.cpu().numpy()[ok]]
            cols += [j.cpu().numpy()[ok], r[ok]]
        r, c = np.concatenate(rows), np.concatenate(cols)
        self.succ = -np.ones(n, np.int64); self.succ[ta] = tb  # temporal successor of each node (-1: none)
        if self.switch == 1.0:
            G = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n))
            G.data[:] = 1.0
        else:  # weighted: temporal 1, neighbour `switch`; a pair that is both keeps the temporal weight
            w = np.full(len(r), self.switch, np.float32); w[:len(ta)] = 1.0
            key = r.astype(np.int64) * n + c
            o = np.lexsort((w, key)); first = np.r_[True, key[o][1:] != key[o][:-1]]
            o = o[first]
            G = csr_matrix((w[o], (r[o], c[o])), shape=(n, n))
        self.GT = G.T.tocsr()  # distances TO the goal = dijkstra from the goal on the transposed graph
        del dd
        torch.cuda.empty_cache()  # several evaluation processes share one GPU
        self._goal_cache, self._anchor_cache, self.anchor_log, self._anchor_node, self._anchor_hops = {}, {}, [], {}, {}
        self.hop_scale = 1.0
        self._t['build_s'] += _time.perf_counter() - t_build
        if self.mode == 'gap' and self.gap_cal:
            rng, r = np.random.default_rng(0), []
            starts = np.r_[0, np.nonzero(np.diff(ep[order]))[0] + 1, len(order)]
            for e in rng.choice(len(starts) - 1, size=min(40, len(starts) - 1), replace=False):
                seg = order[starts[e]:starts[e + 1]]
                if len(seg) < 8:
                    continue
                dsrc = dijkstra(G, directed=True, indices=int(seg[0]))
                for j in range(4, len(seg)):  # >= 4 blocks apart along the recorded episode
                    if np.isfinite(dsrc[seg[j]]) and dsrc[seg[j]] > 0:
                        r.append(j / dsrc[seg[j]])
            self.hop_scale = float(np.median(r)) if r else 1.0
            print(f'[paa] gap hop scale: {self.hop_scale:.2f} recorded blocks per graph hop ({len(r)} pairs)', flush=True)
        print(f'[paa] graph: {n} nodes ({len(eps)} episodes excluded), {len(ta)} temporal + {G.nnz - len(ta)} k-NN edges, '
              f'thr={self.thr:.3f}', flush=True)

    def _nearest(self, z):
        return int((self.emb - z[None]).pow(2).sum(-1).argmin())

    def _goal_tree(self, g):
        key = hash(g.cpu().numpy().tobytes())
        if key not in self._goal_cache:
            import time as _time
            t0 = _time.perf_counter()
            gn = self._nearest(g)
            dist, pred = dijkstra(self.GT, directed=True, indices=gn, return_predecessors=True)
            self._t['dijkstra_s'] += _time.perf_counter() - t0
            self._t['dijkstra_n'] += 1
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

    def _move_anchor(self, s, gn, pred):
        """switch > 1: walk the shortest path and stop after `span` temporal moves; None if the goal comes first"""
        u, moves = s, 0
        for _ in range(self.max_hops * 8):
            if u == gn:
                return None
            v = int(pred[u])
            if v < 0:
                return None
            moves += int(self.succ[u] == v)
            u = v
            if moves >= self.span:
                return u if u != gn else None
        return u

    def anchor(self, z0, g):
        key = (hash(z0.cpu().numpy().tobytes()), hash(g.cpu().numpy().tobytes()))
        if key in self._anchor_cache:
            return self._anchor_cache[key]
        gn, dist, pred = self._goal_tree(g)
        s = self._nearest(z0)
        out, a_node = g, -1
        if self.switch != 1.0 and self.mode == 'fixed':
            if np.isfinite(dist[s]):
                u = self._move_anchor(s, gn, pred)
                if u is not None:
                    a_node, out = u, self.emb[u]
        elif np.isfinite(dist[s]) and dist[s] > self.span:
            path = self._path(s, gn, pred, self.max_hops)
            if path:
                if self.mode == 'fixed' or self.head is None:
                    a_node = path[min(self.span, len(path)) - 1]
                    out = self.emb[a_node]
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
        self._anchor_hops[key] = float(dist[s])  # graph distance (hops, or weighted with switch) to the goal
        self._anchor_node[key] = int(self.idx[a_node]) if a_node >= 0 else -1
        # figure log, indices into nodes.npz: (goal node, current node, anchor node or -1 = the goal itself)
        self.anchor_log.append((int(self.idx[gn]), int(self.idx[s]), int(self.idx[a_node]) if a_node >= 0 else -1))
        return out

    def timing(self):
        """Graph build time (s, all set_exclude calls), mean shortest-path search per goal (ms), number of goals."""
        t = self._t
        return {'graph_build_s': t['build_s'], 'dijkstra_ms': 1e3 * t['dijkstra_s'] / max(1, t['dijkstra_n']),
                'goals': float(t['dijkstra_n']),
                **({'handoff_open': self._ho['open'] / max(1, self._ho['n']), 'handoff_fired': self._ho['fired'] / max(1, self._ho['n'])}
                   if self.handoff else {})}

    def anchor_node(self, z0, g):
        """Index (into the nodes file) of the anchor anchor() picks for (z0, g); -1 when it is the goal itself."""
        self.anchor(z0, g)
        return self._anchor_node[(hash(z0.cpu().numpy().tobytes()), hash(g.cpu().numpy().tobytes()))]

    # ------------------------------------------------------------------------------------------- anchored planning
    def _ap_bank_for(self, k):
        if k not in self._ap_bank:
            if self.mode != 'gap':  # gap keeps its few banks (k <= span); AP keeps one per remaining horizon
                self._ap_bank.clear()  # one remaining-horizon bank at a time (about 1 GB each)
            torch.cuda.empty_cache()
            o, ep, st = self._ap_order, self._ap_ep, self._ap_st
            pos = np.arange(len(o) - k)
            ok = (ep[pos + k] == ep[pos]) & (st[pos + k] - st[pos] == 5 * k) & (ep[pos + 1] == ep[pos]) & \
                 (st[pos + 1] - st[pos] == 5)
            src, far, nxt = (torch.from_numpy(o[pos[ok] + j]).cuda() for j in (0, k, 1))
            npos = pos[ok] + 1  # sorted position of each successor (for apc)
            z, zg = self.emb[src], self.emb[far]
            f = torch.cat([z, zg, zg - z], -1)
            mu, sd = f.mean(0), f.std(0).clamp_min(1e-4)
            f = (f - mu) / sd
            self._ap_bank[k] = (f, f.pow(2).sum(-1), mu, sd, nxt, npos)
            del z, zg
            torch.cuda.empty_cache()
        return self._ap_bank[k]

    def _ap_anchor(self, z0, g):
        out = []
        for b in range(len(z0)):  # elapsed steps per env: +ap_step every time its current latent changes (one decision)
            gk, zk = hash(g[b].cpu().numpy().tobytes()), hash(z0[b].cpu().numpy().tobytes())
            st = self._ap_env.setdefault(gk, {'z': zk, 't': 0, 'a': None, 'p': None})
            if st['z'] != zk:
                st['z'], st['t'], st['a'] = zk, st['t'] + self.ap_step, None
                p = st['p']
                if self.mode == 'apc' and p is not None and self._ap_cont[p] and \
                        float((z0[b] - self.emb[int(self._ap_order[p])]).pow(2).sum()) < self._ap_eps:
                    st['p'] = p + 1  # on track: aim at the record's next frame
                    st['a'] = self.emb[int(self._ap_order[p + 1])]
            if st['a'] is None:
                k = min(self._ap_kmax, max(1, int(round(max(5, self.ap_offset - st['t']) / 5))))
                f, f2, mu, sd, nxt, npos = self._ap_bank_for(k)
                q = (torch.cat([z0[b], g[b], g[b] - z0[b]]) - mu) / sd
                if self.mode in ('ap', 'apc'):
                    i = int((f2 - 2 * (f @ q)).argmin())
                    st['p'] = int(npos[i])
                elif self.mode == 'apk':
                    top = (f2 - 2 * (f @ q)).topk(self.ap_k, largest=False).indices
                    st['a'] = self.emb[nxt[top]]  # (k, D)
                    out.append(st['a'])
                    continue
                else:
                    top = (f2 - 2 * (f @ q)).topk(self.ap_k, largest=False).indices
                    _, dist, _ = self._goal_tree(g[b])
                    i = int(top[int(np.argmin(dist[nxt[top].cpu().numpy()]))])  # argmin keeps retrieval order on ties
                st['a'] = self.emb[nxt[i]]
            out.append(st['a'])
        return torch.stack(out)

    def _gap_anchor(self, z0, g):
        """Graph-guided retrieval: the graph picks a waypoint `span` hops toward the goal and says how far it is (hops);
        AP-style retrieval then returns the recorded successor of the record that best matches (z0, waypoint, hops).
        No goal offset is needed, and the waypoint can lie on a path stitched from several episodes."""
        out = []
        for b in range(len(z0)):
            key = (hash(z0[b].cpu().numpy().tobytes()), hash(g[b].cpu().numpy().tobytes()))
            if key not in self._anchor_cache:
                gn, dist, pred = self._goal_tree(g[b])
                s = self._nearest(z0[b])
                w, k = g[b], self.span
                if np.isfinite(dist[s]):
                    if dist[s] > self.span:
                        path = self._path(s, gn, pred, self.span)
                        if path:
                            w, k = self.emb[path[-1]], len(path)
                    else:
                        k = int(dist[s])
                k = min(self._ap_kmax, max(1, int(round(k * self.hop_scale))))
                f, f2, mu, sd, nxt, npos = self._ap_bank_for(k)
                q = (torch.cat([z0[b], w, w - z0[b]]) - mu) / sd
                self._anchor_cache[key] = self.emb[nxt[int((f2 - 2 * (f @ q)).argmin())]]
            out.append(self._anchor_cache[key])
        return torch.stack(out)

    # ----------------------------------------------------------------------------------------------------- cost
    def forward(self, info_dict):
        g = info_dict['goal_emb'][:, -1].float()  # (B, D)
        H = info_dict['pixels'].size(2)
        z0 = info_dict['predicted_emb'][:, 0, H - 1].float()  # current frame (last context step)
        if self.mode == 'apk':  # nearest of the k retrieved successors, last predicted step
            a = self._ap_anchor(z0, g).to(g.device)  # (B, k, D)
            p = info_dict['predicted_emb'][:, :, -1].float()
            return (p[:, :, None] - a[:, None]).pow(2).sum(-1).min(-1).values
        if self.mode in ('ap', 'apg', 'apc'):
            a = self._ap_anchor(z0, g).to(g.device)
        elif self.mode == 'gap':
            a = self._gap_anchor(z0, g).to(g.device)
        else:
            a = torch.stack([self.anchor(z0[b], g[b]) for b in range(g.shape[0])]).to(g.device)
        if self.agg == 'min':
            p = info_dict['predicted_emb'][:, :, H:].float()
            da = (p - a[:, None, None]).pow(2).sum(-1).min(-1).values
            if not self.handoff:
                return da
            dg = (p - g[:, None, None]).pow(2).sum(-1).min(-1).values
        else:
            p = info_dict['predicted_emb'][:, :, -1].float()
            da = (p - a[:, None]).pow(2).sum(-1)
            if not self.handoff:
                return da
            dg = (p - g[:, None]).pow(2).sum(-1)
        take = dg < self.thr
        if self.handoff_hops > 0 and self.mode == 'fixed':
            hops = torch.tensor([self._anchor_hops[(hash(z0[b].cpu().numpy().tobytes()), hash(g[b].cpu().numpy().tobytes()))]
                                 for b in range(g.shape[0])], device=g.device)
            gate = hops <= self.handoff_hops
            take = take & gate.view(-1, *([1] * (take.dim() - 1)))
            self._ho['open'] += int(gate.sum())
        else:
            self._ho['open'] += g.shape[0]
        self._ho['n'] += g.shape[0]
        self._ho['fired'] += int(take.flatten(1).any(1).sum())
        return torch.where(take, dg, da + self.thr)
