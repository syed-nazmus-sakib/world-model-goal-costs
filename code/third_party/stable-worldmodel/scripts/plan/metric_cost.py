"""Method C (2026-10-22): planning objective with a learned task-aligned goal cost (candC_train.py).

cost = || f(z_T) - f(g) ||^2 on the last predicted latent z_T and the goal latent g, f the trained head
(agg='last'), or the minimum over the predicted steps (agg='min': reach the goal at any step of the plan, so a long
look-ahead does not force the plan to keep moving when the goal is near); optionally mixed with the released latent MSE: cost = mix * released + (1 - mix) * learned (both z-scored per call).
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'lewm_explorations' / 'scripts'))


class MetricGoalCost(nn.Module):
    def __init__(self, head_path, mix=0.0, agg='last'):
        super().__init__()
        if head_path == 'released':  # control: the released latent MSE, with the same aggregation options
            self.head, ck = nn.Identity(), {'args': {'target': 'released'}}
        else:
            ck = torch.load(head_path, map_location='cpu')
            from candC_train import Head

            self.head = Head(d_in=ck['d_in'])
            self.head.load_state_dict(ck['state_dict'])
        self.head.eval().requires_grad_(False)
        self.mix = float(mix)
        assert agg in ('last', 'min')
        self.agg = agg
        print(f'[metric] learned goal cost {head_path} (target={ck["args"]["target"]}), mix={self.mix}, agg={agg}', flush=True)

    def forward(self, info_dict):
        g = info_dict['goal_emb'][:, -1]  # (B, D)
        self.head.to(g.device)
        fg = self.head(g.float())
        if self.agg == 'last':
            p = info_dict['predicted_emb'][:, :, -1]  # (B, S, D)
            cost = (self.head(p.float()) - fg[:, None]).pow(2).sum(-1)
        else:  # skip the encoded context frames, keep the predicted ones
            H = info_dict['pixels'].size(2)
            p = info_dict['predicted_emb'][:, :, H:]  # (B, S, T, D)
            cost = (self.head(p.float()) - fg[:, None, None]).pow(2).sum(-1).min(-1).values
            p = info_dict['predicted_emb'][:, :, -1]
        if self.mix > 0:
            rel = (p.float() - g.float()[:, None]).pow(2).sum(-1)
            z = lambda c: (c - c.mean(1, keepdim=True)) / (c.std(1, keepdim=True) + 1e-8)
            cost = self.mix * z(rel) + (1 - self.mix) * z(cost)
        return cost
