# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Does one shared planning metric over a frozen latent suffer negative transfer across tasks?

All metrics are diagonal reweightings of the squared latent goal error S[n, p, c] = (z - z_goal)^2
(p: patch, c: channel) and are trained on the same pairwise labels "candidate i has lower simulator
task cost than candidate j" (same episode, same goal image):
    E0  Euclidean, no training
    E1  global: one set of weights for all tasks
    E2  task-conditioned: one part of the weights shared, the other part per task
    E3  per-task: independent weights for every task (upper bound)

Usage:
    python -m evals.mw_metric_conflict.fit --data_dir <data> --emb_dir <emb> --out <results.json>
"""

import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

TASKS = ["reach", "push", "drawer-open", "door-open"]
# pairs whose task costs differ by less than this are not labelled (m)
COST_MARGIN = 0.01
FAMILIES = ["chan", "patch", "patch_x_chan"]


def load_task(data_dir, emb_dir, task, device):
    d = np.load(os.path.join(data_dir, f"{task}.npz"))
    e = np.load(os.path.join(emb_dir, f"{task}.npz"))
    ep = d["cand_episode"]
    z = torch.from_numpy(e["cand"].astype(np.float32))
    g = torch.from_numpy(e["goal"].astype(np.float32))[ep]
    S = (z - g).pow(2).to(device)  # (N, P, C)
    return {
        "S": S,
        "S_chan": S.sum(1),
        "S_patch": S.sum(2),
        "cost": d["cand_cost"].astype(np.float64),
        "episode": ep,
        "tcp_to_target": np.linalg.norm(d["cand_tcp"] - d["cand_target"], axis=1),
        "obj_to_target": np.linalg.norm(d["cand_obj"] - d["cand_target"], axis=1),
    }


def make_pairs(cost, episode, episodes):
    """Index pairs (better, worse) inside each of the given episodes."""
    better, worse = [], []
    for e in episodes:
        idx = np.nonzero(episode == e)[0]
        i, j = np.triu_indices(len(idx), k=1)
        i, j = idx[i], idx[j]
        dc = cost[i] - cost[j]
        keep = np.abs(dc) > COST_MARGIN
        i, j, dc = i[keep], j[keep], dc[keep]
        better.append(np.where(dc < 0, i, j))
        worse.append(np.where(dc < 0, j, i))
    return np.concatenate(better), np.concatenate(worse)


class Metric(torch.nn.Module):
    """log-weights over patches and / or channels, one row per task slot."""

    def __init__(self, family, n_slots_patch, n_slots_chan, P, C, device):
        super().__init__()
        self.family = family
        self.log_u = torch.nn.Parameter(torch.zeros(n_slots_patch, P, device=device))
        self.log_v = torch.nn.Parameter(torch.zeros(n_slots_chan, C, device=device))
        self.log_scale = torch.nn.Parameter(torch.zeros((), device=device))
        self.log_u.requires_grad_(family in ("patch", "patch_x_chan"))
        self.log_v.requires_grad_(family in ("chan", "patch_x_chan"))

    def weights(self, slot_patch, slot_chan):
        u = torch.softmax(self.log_u[slot_patch], 0) * self.log_u.shape[1]  # mean 1
        v = torch.softmax(self.log_v[slot_chan], 0) * self.log_v.shape[1]
        return u, v

    def forward(self, dat, slot_patch=0, slot_chan=0):
        u, v = self.weights(slot_patch, slot_chan)
        if self.family == "chan":
            d = dat["S_chan"] @ v
        elif self.family == "patch":
            d = dat["S_patch"] @ u
        else:
            d = (dat["S"] @ v) @ u
        return d / (len(u) * len(v))


def rank_loss(model, d, better, worse):
    return F.softplus(-(d[worse] - d[better]) * model.log_scale.exp() * 100).mean()


def train(model, tasks, slots, data, pairs, steps, lr, wd):
    """tasks: list of task names trained jointly with equal weight; slots[task] = (slot_patch, slot_chan)"""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0
        for t in tasks:
            d = model(data[t], *slots[t])
            loss = loss + rank_loss(model, d, *pairs[t]) / len(tasks)
        loss = loss + wd * (model.log_u.pow(2).mean() + model.log_v.pow(2).mean())
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, task, slot, data, pairs, episodes):
    d = model(data[task], *slot).cpu().numpy()
    better, worse = pairs
    acc = float((d[better] < d[worse]).mean())
    rho = [
        stats.spearmanr(d[data[task]["episode"] == e], data[task]["cost"][data[task]["episode"] == e])[0]
        for e in episodes
    ]
    return {"pair_acc": acc, "spearman": float(np.mean(rho))}


def task_grad(model, task, slot, data, pairs):
    model.zero_grad()
    rank_loss(model, model(data[task], *slot), *pairs[task]).backward()
    return torch.cat([p.grad.flatten() for p in (model.log_u, model.log_v) if p.grad is not None]).clone()


def cosine_matrix(grads):
    g = torch.stack([grads[t] for t in TASKS])
    g = g / g.norm(dim=1, keepdim=True)
    return (g @ g.T).cpu().numpy().round(4).tolist()


def preference_conflict(data, tasks=("reach", "push")):
    """
    Reach and push share a scene (arm, puck, target marker). On the same pairs of states, how often do
    the reach cost ||tcp - target|| and the push cost ||obj - target|| order the pair differently?
    """
    out = {}
    for t in tasks:
        i, j = [], []
        for e in np.unique(data[t]["episode"]):
            idx = np.nonzero(data[t]["episode"] == e)[0]
            a, b = np.triu_indices(len(idx), k=1)
            i.append(idx[a])
            j.append(idx[b])
        i, j = np.concatenate(i), np.concatenate(j)
        dr = data[t]["tcp_to_target"][i] - data[t]["tcp_to_target"][j]
        dp = data[t]["obj_to_target"][i] - data[t]["obj_to_target"][j]
        both = (np.abs(dr) > COST_MARGIN) & (np.abs(dp) > COST_MARGIN)
        out[f"{t}_states"] = {
            "n_pairs_both_labelled": int(both.sum()),
            "frac_pairs_both_labelled": float(both.mean()),
            "conflict_rate": float((np.sign(dr[both]) != np.sign(dp[both])).mean()) if both.any() else None,
        }
    return out


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
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = {t: load_task(args.data_dir, args.emb_dir, t, device) for t in TASKS}
    P, C = data[TASKS[0]]["S"].shape[1:]
    T = len(TASKS)
    slot_id = {t: k for k, t in enumerate(TASKS)}
    results = {"tasks": TASKS, "cost_margin": COST_MARGIN, "preference_conflict_reach_vs_push": preference_conflict(data)}
    runs = []

    for split in range(args.n_splits):
        rs = np.random.RandomState(split)
        tr_pairs, te_pairs, te_eps = {}, {}, {}
        for t in TASKS:
            eps = rs.permutation(np.unique(data[t]["episode"]))
            n_tr = int(round(args.train_frac * len(eps)))
            tr_pairs[t] = make_pairs(data[t]["cost"], data[t]["episode"], eps[:n_tr])
            te_pairs[t] = make_pairs(data[t]["cost"], data[t]["episode"], eps[n_tr:])
            te_eps[t] = eps[n_tr:]
        ev = lambda model, slots: {t: evaluate(model, t, slots[t], data, te_pairs[t], te_eps[t]) for t in TASKS}
        fit = lambda model, tasks, slots: train(model, tasks, slots, data, tr_pairs, args.steps, args.lr, args.wd)
        zero = {t: (0, 0) for t in TASKS}
        own = {t: (slot_id[t], slot_id[t]) for t in TASKS}
        run = {"split": split, "n_train_pairs": {t: len(tr_pairs[t][0]) for t in TASKS}}

        euclid = Metric("patch_x_chan", 1, 1, P, C, device)
        run["E0_euclidean"] = ev(euclid, zero)
        # which way does each task pull a shared metric that starts Euclidean
        run["grad_cosine_at_euclidean"] = cosine_matrix(
            {t: task_grad(euclid, t, (0, 0), data, tr_pairs) for t in TASKS}
        )

        for fam in FAMILIES:
            res = {}
            # E1: one metric for all tasks
            g = fit(Metric(fam, 1, 1, P, C, device), TASKS, zero)
            res["E1_global"] = ev(g, zero)
            # E3: independent metric per task. Also score each on the other tasks (cross-task transfer)
            res["E3_per_task"], transfer = {}, {}
            for t in TASKS:
                m = fit(Metric(fam, 1, 1, P, C, device), [t], zero)
                scores = ev(m, zero)
                res["E3_per_task"][t] = scores[t]
                transfer[t] = {t2: scores[t2]["pair_acc"] for t2 in TASKS}
            res["transfer_trained_on_row_tested_on_col"] = transfer
            if fam == "patch_x_chan":
                # E2: task identity selects one factor, the other factor is shared by all tasks
                m = fit(Metric(fam, T, 1, P, C, device), TASKS, {t: (slot_id[t], 0) for t in TASKS})
                res["E2_task_patch_shared_chan"] = ev(m, {t: (slot_id[t], 0) for t in TASKS})
                m = fit(Metric(fam, 1, T, P, C, device), TASKS, {t: (0, slot_id[t]) for t in TASKS})
                res["E2_shared_patch_task_chan"] = ev(m, {t: (0, slot_id[t]) for t in TASKS})
                # same parameter count as E3 but trained jointly: checks that joint training itself costs nothing
                m = fit(Metric(fam, T, T, P, C, device), TASKS, own)
                res["E2_task_patch_task_chan_joint"] = ev(m, own)
                # leave-one-task-out global metrics: does removing a task help the others
                loo = {}
                for t_out in TASKS:
                    rest = [t for t in TASKS if t != t_out]
                    m = fit(Metric(fam, 1, 1, P, C, device), rest, zero)
                    scores = ev(m, zero)
                    loo[t_out] = {t2: scores[t2]["pair_acc"] for t2 in TASKS}
                res["global_leave_one_task_out_row_left_out"] = loo
                # pairwise-task global metrics
                pairwise = {}
                for a, b in itertools.combinations(TASKS, 2):
                    m = fit(Metric(fam, 1, 1, P, C, device), [a, b], zero)
                    scores = ev(m, zero)
                    pairwise[f"{a}+{b}"] = {a: scores[a]["pair_acc"], b: scores[b]["pair_acc"]}
                res["global_two_tasks"] = pairwise
            run[fam] = res
            print(f"split {split} {fam}: E1 {[round(res['E1_global'][t]['pair_acc'], 3) for t in TASKS]} "
                  f"E3 {[round(res['E3_per_task'][t]['pair_acc'], 3) for t in TASKS]}", flush=True)
        runs.append(run)

    results["runs"] = runs
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
