"""NGC baseline (concurrent work, NOT ours): Neighborhood Graph Cost of You et al., "Latent World Models Should Plan with
Distances Along the Latent Manifold", arXiv 2610.25013 (code: github.com/youzunzhi/ngc-latent-manifold, MIT license,
Copyright (c) the NGC authors).  `build_graph`, `node_distances` and `NGCCost` below are copied from their ngc/graph.py
and ngc/cost.py with only comments shortened, so that the comparison uses their cost exactly.

What we add for our protocol (NGCObjective):
  * nodes: N (default 5,000, their setting) frames, one random 5-step-grid frame from each of N random training episodes of
    our nodes file, after removing the evaluation episodes (set_exclude, called by eval_wm.py), as NGC excludes the pair
    episodes; if fewer than N episodes are eligible, every eligible episode contributes one node
  * target='goal' : NGC as published (terminal cost, LeWM search H=5)
    target='anchor': hybrid, our graph anchor (paa_cost.PAAGoalCost, mode fixed, span K) as the target and NGC's graph
                     distance to it as the cost; agg='min' takes the minimum over the predicted steps (our search settings)
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist
from torch import nn


# ----------------------------------------------------------------------------------------- NGC (copied, MIT license)
def neighbor_links(distance, neighbors, rule):
    nearest = np.argsort(distance, axis=1, kind='stable')[:, :neighbors]
    listed = np.zeros(distance.shape, dtype=bool)
    listed[np.arange(len(distance))[:, None], nearest] = True
    if rule == 'mutual':
        return listed & listed.T
    return listed | listed.T


def link_graph(distance, neighbors, rule):
    linked = neighbor_links(distance, neighbors, rule)
    bridges = []
    count, labels = connected_components(csr_matrix(linked), directed=False)
    while count > 1:
        between = np.where(labels[:, None] != labels[None, :], distance, np.inf)
        i, j = np.unravel_index(np.argmin(between), between.shape)
        linked[i, j] = linked[j, i] = True
        bridges.append((int(i), int(j)))
        count, labels = connected_components(csr_matrix(linked), directed=False)
    return linked, bridges


def build_graph(latents, neighbors, rule):
    nodes = np.asarray(latents, dtype=np.float64)
    assert nodes.ndim == 2 and len(nodes) > neighbors
    distance = cdist(nodes, nodes)
    np.fill_diagonal(distance, np.inf)
    linked, bridges = link_graph(distance, neighbors, rule)
    row, col = np.nonzero(linked)
    edges = csr_matrix((distance[row, col], (row, col)), shape=distance.shape)
    graph_distance = dijkstra(edges, directed=False)
    return nodes, graph_distance, bridges


def node_distances(latents, nodes):
    distance = torch.cdist(latents, nodes, compute_mode='use_mm_for_euclid_dist')
    return torch.sort(distance, dim=1, stable=True)


class NGCCost(nn.Module):
    def __init__(self, nodes, graph_distance, neighbors, anchors):
        super().__init__()
        nodes = torch.as_tensor(nodes, dtype=torch.float32)
        graph_distance = torch.as_tensor(graph_distance, dtype=torch.float32)
        assert nodes.ndim == 2 and len(nodes) > neighbors >= anchors >= 1
        assert torch.isfinite(graph_distance).all()
        self.register_buffer('nodes', nodes)
        self.register_buffer('graph_distance', graph_distance)
        self.neighbors, self.anchors = neighbors, anchors
        self.cached_goal = None
        self.cached_goal_attachment = None

    def attach(self, latents):
        nearest, ids = node_distances(latents, self.nodes)
        radius = nearest[:, self.neighbors - 1]
        return ids[:, : self.anchors], nearest[:, : self.anchors], radius

    def attach_goal(self, goal):
        if self.cached_goal is None or not (self.cached_goal.shape == goal.shape and torch.equal(self.cached_goal, goal)):
            self.cached_goal = goal.clone()
            self.cached_goal_attachment = self.attach(goal)
        return self.cached_goal_attachment

    def cost(self, q, g):
        """q: predicted latents (B, S, D), g: goal latents (B, D) -> (B, S); the body of their forward()"""
        batch, samples, dim = q.shape
        q_ids, q_distance, q_radius = self.attach(q.reshape(-1, dim))
        q_ids = q_ids.reshape(batch, samples, self.anchors)
        q_distance = q_distance.reshape(batch, samples, self.anchors)
        q_radius = q_radius.reshape(batch, samples)
        g_ids, g_distance, g_radius = self.attach_goal(g)
        route = (self.graph_distance[q_ids[..., :, None], g_ids[:, None, None, :]]
                 + q_distance[..., :, None] + g_distance[:, None, None, :]).amin(dim=(-1, -2))
        direct = torch.linalg.vector_norm(q - g[:, None], dim=-1)
        local = direct <= torch.minimum(q_radius, g_radius[:, None])
        return torch.where(local, direct, route)

    def forward(self, info_dict):
        q = info_dict['predicted_emb'][..., -1, :].detach().float()
        g = info_dict['goal_emb'][:, -1, :].detach().float()
        return self.cost(q, g).to(info_dict['predicted_emb'].dtype)


# ----------------------------------------------------------------------------------------- our wrapper
class NGCObjective(nn.Module):
    def __init__(self, nodes, n_nodes=5000, neighbors=15, anchors=4, rule='mutual', seed=0, target='goal', span=6,
                 agg='last', max_hops=12, knn=10):
        super().__init__()
        d = np.load(nodes)
        self.emb_all, self.ep_all = d['emb'].astype(np.float32), d['ep']
        self.n, self.k, self.a, self.rule, self.seed = int(n_nodes), int(neighbors), int(anchors), rule, int(seed)
        self.target, self.agg = target, agg
        self.anchor_cost = None
        if target == 'anchor':
            from paa_cost import PAAGoalCost
            self.anchor_cost = PAAGoalCost(nodes, mode='fixed', span=span, max_hops=max_hops, knn=knn)
        self.ngc = None
        self.set_exclude([])

    def set_exclude(self, eps):
        import time
        t0 = time.perf_counter()
        eligible = np.setdiff1d(np.unique(self.ep_all), np.asarray(list(eps)))
        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(eligible, size=min(self.n, len(eligible)), replace=False)
        order = np.argsort(self.ep_all, kind='stable')
        ep_sorted = self.ep_all[order]
        rows = []
        for e in np.sort(chosen):  # one random grid frame of each chosen episode
            lo, hi = np.searchsorted(ep_sorted, [e, e + 1])
            rows.append(order[rng.integers(lo, hi)])
        lat = self.emb_all[np.sort(rows)]
        nodes, gd, bridges = build_graph(lat, self.k, self.rule)
        self.ngc = NGCCost(nodes, gd, self.k, self.a).cuda()
        if self.anchor_cost is not None:
            self.anchor_cost.set_exclude(eps)
        self.build_s = time.perf_counter() - t0
        print(f'[ngc] {len(nodes)} nodes ({len(eps)} episodes excluded), mutual {self.k}-NN, {len(bridges)} bridges, '
              f'built in {self.build_s:.1f}s', flush=True)

    def timing(self):
        return {'graph_build_s': float(self.build_s)}

    def forward(self, info_dict):
        g = info_dict['goal_emb'][:, -1].detach().float()
        if self.anchor_cost is not None:  # hybrid: our anchor as the target
            H = info_dict['pixels'].size(2)
            z0 = info_dict['predicted_emb'][:, 0, H - 1].float()
            g = torch.stack([self.anchor_cost.anchor(z0[b], g[b]) for b in range(g.shape[0])]).to(g.device).float()
        P = info_dict['predicted_emb'].detach().float()  # (B, S, T, D)
        if self.agg == 'min':
            H = info_dict['pixels'].size(2)
            c = torch.stack([self.ngc.cost(P[:, :, t], g) for t in range(H, P.shape[2])], -1)
            return c.min(-1).values.to(info_dict['predicted_emb'].dtype)
        return self.ngc.cost(P[:, :, -1], g).to(info_dict['predicted_emb'].dtype)
