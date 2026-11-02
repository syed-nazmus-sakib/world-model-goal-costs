"""Graph waypoints for a goal-conditioned policy (the "SoRB-style" baseline: graph + policy, no world-model planning).

Every `every` environment steps, the current frame and the goal frame are embedded with the frozen LeWM encoder (the same
latent space the graph nodes were built in, paa_nodes.py), and PAAGoalCost picks the anchor node `span` hops along the
shortest graph path (mode fixed, the same rule as our planner). The wrapped feed-forward policy (e.g. GCBC) is then given
that node's recorded dataset frame and proprio as its goal instead of the real goal; when the goal is at most `span` hops
away (anchor = -1) the real goal is kept. Everything else (pairs, budget, success test) is unchanged, so the comparison
with graph anchors + CEM differs only in who follows the waypoints: a learned policy vs planning with the world model.

Used by eval_wm.py when +graph_waypoints=true (with +ff=true, +graph_nodes=<nodes.npz>, +paa_span, +graph_wm=<LeWM checkpoint>).
"""
import numpy as np
import torch

from stable_worldmodel.policy import BasePolicy


class GraphWaypointPolicy(BasePolicy):
    def __init__(self, inner, encoder, graph, dataset, transform, every=5, img_dtype=torch.float32):
        super().__init__()
        self.type = 'graph_waypoint'
        self.inner, self.wm, self.graph, self.dataset = inner, encoder, graph, dataset
        self.tf, self.every, self.img_dtype = transform, int(every), img_dtype
        self._t, self._wp, self._rows = 0, None, {}
        self.log = []  # (env step, env index, anchor node or -1), for analysis

    def set_env(self, env):
        self.env = env
        self.inner.set_env(env)

    @staticmethod
    def _last(frames):  # (B, T, H, W, C) or (B, H, W, C) -> (B, H, W, C)
        frames = np.asarray(frames)
        return frames[:, -1] if frames.ndim == 5 else frames

    @torch.no_grad()
    def _enc(self, frames):
        device = next(self.wm.parameters()).device
        im = torch.stack([self.tf(np.ascontiguousarray(x)) for x in frames]).to(device, self.img_dtype)
        return self.wm.encode({'pixels': im[:, None]})['emb'][:, 0].float()

    def _row(self, node):  # the node's recorded frame and per-step columns, in the env's 'goal*' format
        if node not in self._rows:
            ep, st = int(self.graph.ep_all[node]), int(self.graph.step_all[node])
            d = self.dataset.load_chunk(np.array([ep]), np.array([st]), np.array([st + 1]))[0]
            row = {}
            for col, val in d.items():
                if col.startswith('goal') or not isinstance(val, (torch.Tensor, np.ndarray)):
                    continue
                if col.startswith('pixels'):
                    val = val.permute(0, 2, 3, 1)
                arr = val.numpy() if isinstance(val, torch.Tensor) else val
                row['goal' if col == 'pixels' else f'goal_{col}'] = arr[-1]
            self._rows[node] = row
        return self._rows[node]

    def get_action(self, info_dict, **kwargs):
        if self._wp is None or self._t % self.every == 0:
            z0, g = self._enc(self._last(info_dict['pixels'])), self._enc(self._last(info_dict['goal']))
            self._wp = [self.graph.anchor_node(z0[b], g[b]) for b in range(len(z0))]
            self.log += [(self._t, b, n) for b, n in enumerate(self._wp)]
        self._t += 1
        info, copied = dict(info_dict), set()
        for b, node in enumerate(self._wp):
            if node < 0:
                continue
            for k, v in self._row(node).items():
                if k not in info:
                    continue
                if k not in copied:
                    info[k] = np.array(info[k], copy=True)
                    copied.add(k)
                info[k][b][...] = v  # broadcasts over a history dimension if there is one
        return self.inner.get_action(info, **kwargs)
