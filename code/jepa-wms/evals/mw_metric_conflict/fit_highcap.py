# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Archival check for fit.py: does the gap between one global metric and per-task metrics
survive when the metric has much higher capacity?

    full_diag           one free weight per (patch, channel) entry of the latent
    full_diag_lowrank   full_diag plus a rank-r Mahalanobis term ||L (z - z_goal)||^2

Same pairwise labels and the same train / test episode splits as fit.py.

Usage:
    python -m evals.mw_metric_conflict.fit_highcap --data_dir <data> --emb_dir <emb> --out <results.json>
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from evals.mw_metric_conflict.fit import TASKS, make_pairs


def load_task(data_dir, emb_dir, task, device):
    d = np.load(os.path.join(data_dir, f"{task}.npz"))
    e = np.load(os.path.join(emb_dir, f"{task}.npz"))
    ep = d["cand_episode"]
    z = torch.from_numpy(e["cand"].astype(np.float32)).flatten(1)
    g = torch.from_numpy(e["goal"].astype(np.float32)).flatten(1)[ep]
    return {"D": (z - g).to(device), "cost": d["cand_cost"].astype(np.float64), "episode": ep}


class HighCapMetric(torch.nn.Module):
    def __init__(self, dim, rank, device, seed):
        super().__init__()
        self.log_w = torch.nn.Parameter(torch.zeros(dim, device=device))
        self.log_scale = torch.nn.Parameter(torch.zeros((), device=device))
        self.rank = rank
        if rank > 0:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            self.L = torch.nn.Parameter((torch.randn(rank, dim, generator=gen) * 1e-2 / dim**0.5).to(device))

    def forward(self, D):
        w = torch.softmax(self.log_w, 0)  # sums to 1, so the Euclidean metric is log_w = 0
        d = D.pow(2) @ w
        if self.rank > 0:
            d = d + (D @ self.L.T).pow(2).sum(1)
        return d


def train(model, tasks, data, pairs, steps, lr, wd):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = wd * model.log_w.pow(2).mean()
        for t in tasks:
            d = model(data[t]["D"])
            better, worse = pairs[t]
            loss = loss + F.softplus(-(d[worse] - d[better]) * model.log_scale.exp() * 100).mean() / len(tasks)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def pair_acc(model, dat, pairs):
    d = model(dat["D"]).cpu().numpy()
    return float((d[pairs[0]] < d[pairs[1]]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--train_frac", type=float, default=2 / 3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = {t: load_task(args.data_dir, args.emb_dir, t, device) for t in TASKS}
    dim = data[TASKS[0]]["D"].shape[1]
    runs = []
    for split in range(args.n_splits):
        # same episode splits as fit.py
        rs = np.random.RandomState(split)
        tr, te = {}, {}
        for t in TASKS:
            eps = rs.permutation(np.unique(data[t]["episode"]))
            n_tr = int(round(args.train_frac * len(eps)))
            tr[t] = make_pairs(data[t]["cost"], data[t]["episode"], eps[:n_tr])
            te[t] = make_pairs(data[t]["cost"], data[t]["episode"], eps[n_tr:])
        run = {"split": split}
        for name, rank in (("full_diag", 0), (f"full_diag_lowrank{args.rank}", args.rank)):
            res = {"E1_global": {}, "E3_per_task": {}, "E1_global_train": {}, "E3_per_task_train": {}}
            g = train(HighCapMetric(dim, rank, device, split), TASKS, data, tr, args.steps, args.lr, args.wd)
            for t in TASKS:
                res["E1_global"][t] = pair_acc(g, data[t], te[t])
                res["E1_global_train"][t] = pair_acc(g, data[t], tr[t])
                m = train(HighCapMetric(dim, rank, device, split), [t], data, tr, args.steps, args.lr, args.wd)
                res["E3_per_task"][t] = pair_acc(m, data[t], te[t])
                res["E3_per_task_train"][t] = pair_acc(m, data[t], tr[t])
            run[name] = res
            print(
                f"split {split} {name}: E1 {[round(res['E1_global'][t], 3) for t in TASKS]} "
                f"E3 {[round(res['E3_per_task'][t], 3) for t in TASKS]}",
                flush=True,
            )
        runs.append(run)
    with open(args.out, "w") as f:
        json.dump({"tasks": TASKS, "runs": runs}, f, indent=1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
