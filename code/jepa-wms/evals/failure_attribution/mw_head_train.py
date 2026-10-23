# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Method C on Metaworld (2026-10-22), step 2: a learned goal cost on frozen JEPA-WM encodings.

A small head f maps the patch tokens (and proprio embedding) of one frame to a 128-d vector; the planning cost is
||f(z) - f(g)||^2. It is trained so that ||f(z) - f(g)|| regresses log(1 + steps from z to g / 5) for frames of the
same expert episode (self-supervised: no simulator state, no task labels). Half the goals are the episode's last
frame, which is what the eval uses as its goal. One head is trained for all tasks in the data file.

Head: per-patch linear projection + learned position embedding, attention pooling with 4 learned queries (keeps
small, local entities that a patch average dilutes), concatenated with the proprio embedding, then an MLP.

Validation (10% held-out episodes): Spearman correlation with the true step gap, head vs the released cost
(mean squared patch difference).

Usage (from code/jepa-wms):
    python -m evals.failure_attribution.mw_head_train --data <npz from mw_head_data> --out <head.pt>
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn


class PatchHead(nn.Module):
    def __init__(self, n_patches, d_in, d_prop, width=128, n_queries=4, d_out=128):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, width))
        self.pos = nn.Parameter(torch.zeros(1, n_patches, width))
        self.queries = nn.Parameter(torch.randn(1, n_queries, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.prop = nn.Sequential(nn.LayerNorm(d_prop), nn.Linear(d_prop, width)) if d_prop else None
        d_cat = width * (n_queries + (1 if d_prop else 0))
        self.mlp = nn.Sequential(nn.Linear(d_cat, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, d_out))

    def forward(self, vis, prop=None):
        """vis (N, P, D), prop (N, Dp) -> (N, d_out)"""
        x = self.proj(vis) + self.pos
        q = self.queries.expand(len(x), -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        h = [pooled.flatten(1)]
        if self.prop is not None:
            h.append(self.prop(prop))
        return self.mlp(torch.cat(h, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    D = np.load(args.data)
    vis = torch.from_numpy(D["vis"]).cuda()  # (N, P, Dv) float16, kept on the GPU
    prop = torch.from_numpy(D["prop"]).cuda()
    ep, t = D["ep"], D["t"]
    N, Pn, Dv = vis.shape
    order = np.lexsort((t, ep))  # frames sorted by (episode, step)
    assert (order == np.arange(N)).all(), "mw_head_data writes frames in (episode, step) order"
    eps = np.unique(ep)
    first = np.searchsorted(ep, eps)
    last = np.r_[first[1:], N] - 1
    ep_last = dict(zip(eps, last))
    val_eps = set(rng.choice(eps, size=max(1, len(eps) // 10), replace=False).tolist())
    is_val = np.isin(ep, list(val_eps))
    train_idx, val_idx = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]

    def pairs(pool, n):
        a = rng.choice(pool, size=n)
        hi = np.array([ep_last[e] for e in ep[a]])
        g = np.where(rng.random(n) < 0.5, hi, rng.integers(a, hi + 1))  # goal at or after the anchor
        return a, g, np.log1p((t[g] - t[a]) / 5.0).astype(np.float32)

    head = PatchHead(Pn, Dv, prop.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    va, vg, vy = pairs(val_idx, 4000)

    @torch.no_grad()
    def validate():
        head.eval()
        ds, dr = [], []
        for s in range(0, len(va), 500):
            a, g = va[s : s + 500], vg[s : s + 500]
            fa = head(vis[a].float(), prop[a].float())
            fg = head(vis[g].float(), prop[g].float())
            ds.append((fa - fg).pow(2).sum(-1).cpu())
            dr.append((vis[a].float() - vis[g].float()).pow(2).mean((1, 2)).cpu())
        head.train()
        gap = t[vg] - t[va]
        return spearmanr(torch.cat(ds), gap)[0], spearmanr(torch.cat(dr), gap)[0]

    log = []
    for step in range(1, args.steps + 1):
        a, g, y = pairs(train_idx, args.batch)
        fa = head(vis[a].float(), prop[a].float())
        fg = head(vis[g].float(), prop[g].float())
        d = (fa - fg).pow(2).sum(-1).add(1e-8).sqrt()
        loss = F.smooth_l1_loss(d, torch.from_numpy(y).cuda())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == args.steps:
            sh, sr = validate()
            rec = {"step": step, "loss": float(loss), "val_spearman_head": float(sh), "val_spearman_released": float(sr)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    torch.save(
        {"state_dict": head.state_dict(), "n_patches": Pn, "d_in": Dv, "d_prop": prop.shape[1], "args": vars(args),
         "tasks": D["tasks"].tolist(), "log": log},
        args.out,
    )
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
