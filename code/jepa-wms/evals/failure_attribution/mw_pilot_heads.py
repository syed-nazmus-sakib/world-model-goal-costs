# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Method D pilot (2026-10-22), step 2: train goal-cost heads on mw_pilot_data encodings and score them offline.

All heads are mw_head_train.PatchHead on the frozen encodings, cost ||f(z) - f(g)||^2, g = an encoded goal image.
They differ only in what supervision they get (training episodes only):
  state      privileged: ||f(z) - f(g)|| regresses log(1 + obj_to_target / threshold) of z (distance to the success set,
             since every goal g is a success state). Upper reference, not a method.
  time       unlabeled: expert frames, ||f(z) - f(g)|| regresses log(1 + steps from z to g / 5) (Method C target).
  evid       weak task evidence: a budget of --n_pos success frames and --n_neg near-miss frames
             (1 < obj_to_target / threshold < 3); success frames pulled onto f(g), near misses pushed beyond a margin.
  evid_time  both losses (Method D: "what counts as done" from evidence, "how to get there" from time).
  rtc        approach-aware (2026-10-24, from the Stage 1 oracle result): ||f(z) - f(g)|| regresses
             log(1 + d_task(z)) once the object has moved (> 2 mm from its start), and
             log(1 + d_task(start) + hand-to-object distance / 2 cm) before: dense in the approach phase, never below the
             start's task distance there, so the structure "reach the object, then change it" is learned, not hand-coded.
             Tasks whose object never moves (reach-wall) keep the plain task-distance target.
             Privileged states at training time only. Trained across tasks (one head) when several --train_dirs are given.

Offline test on held-out episodes, per candidate set (one start point of one episode; sets where no candidate changes
the true cost are skipped):
  acc       ordering accuracy against the true cost on clearly different pairs (true-cost gap above the set's median gap)
  spearman  rank correlation with the true cost
  regret    true cost (success units) of the candidate the cost selects minus the best candidate's
  top1      the selected candidate is a success; any = at least one candidate in the set is a success
  decoy     over all frames of a test episode: fraction of (success frame, arm-at-goal failure) pairs in which the cost
            prefers the success frame (arm-at-goal failure: hand within 5 cm of the eval goal's hand, obj_to_target > 2x)
Costs: released (alpha 0 and the eval's 0.1), Method C time head (--c_head), and the heads above; goals: eval goal
(variant 0) and, where they exist, arm-moved variants.

    python -m evals.failure_attribution.mw_pilot_heads --train_dir <dir> --test_dir <dir> --task mw-drawer-open --out_dir <dir>
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from evals.failure_attribution.mw_head_train import PatchHead
from evals.failure_attribution.mw_sim import SUCCESS_THRESHOLD


def load(dirs, tasks=None):
    """One or several pilot-data dirs (aligned with task names) -> dict; ep is unique across dirs, thr is per frame."""
    dirs = [dirs] if isinstance(dirs, str) else list(dirs)
    tasks = [tasks] if isinstance(tasks, str) else (list(tasks) if tasks else [None] * len(dirs))
    # two passes: count frames from the small 'kind' arrays, then fill preallocated arrays (np.concatenate on the patch
    # features doubled the peak memory: 28 GB of features + copy exceeded the pod's 62 GB container limit, 2026-10-24)
    files = []
    n_ep = 0
    for di, (d, task) in enumerate(zip(dirs, tasks)):
        fs = sorted(glob.glob(os.path.join(d, "ep_*.npz")))
        files += [(f, n_ep + i, SUCCESS_THRESHOLD.get(task, 0.05), di) for i, f in enumerate(fs)]
        n_ep += len(fs)
    counts = [len(np.load(f)["kind"]) for f, _, _, _ in files]
    N = int(sum(counts))
    z0 = np.load(files[0][0])
    out = {"vis": np.empty((N,) + z0["vis"].shape[1:], z0["vis"].dtype), "prop": np.empty((N,) + z0["prop"].shape[1:], z0["prop"].dtype),
           "state": np.empty((N,) + z0["state"].shape[1:], z0["state"].dtype), "kind": np.empty(N, z0["kind"].dtype),
           "t": np.empty(N, z0["t"].dtype), "var": np.empty(N, z0["var"].dtype), "start": np.empty(N, z0["start"].dtype),
           "ep": np.empty(N, np.int64), "thr": np.empty(N, np.float32), "task": np.empty(N, np.int64)}
    s = 0
    for (f, e, thr, di), n in zip(files, counts):
        z = np.load(f)
        sl = slice(s, s + n)
        for k in ("vis", "prop", "state", "kind", "t", "var", "start"):
            out[k][sl] = z[k]
        out["ep"][sl], out["thr"][sl], out["task"][sl] = e, thr, di
        s += n
    return out


def train_head(D, mode, thr, steps, n_pos, n_neg, seed, dev, n_ens=1):
    """Returns an nn.ModuleList of n_ens PatchHeads (different inits, per-member bootstrap masks on shared batches) and info.
    With n_ens > 1 the planner can use mean + beta * std of the member costs (costs.LearnedHeadMPCObjective), so that
    candidates outside the training distribution, where members disagree, are not exploited."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # features stay on the CPU (8 tasks x 24 episodes of patch features are about 19 GB); each batch moves to the GPU
    vis = torch.from_numpy(D["vis"])
    prop = torch.from_numpy(D["prop"])
    ep, kind, t, st = D["ep"], D["kind"], D["t"], D["state"]
    thr = D["thr"] if "thr" in D else np.full(len(st), thr, dtype=np.float32)
    y_succ = np.log1p(st[:, 40] / thr).astype(np.float32)
    # approach-aware target: before the object moves, start task distance + hand-to-object distance (2 cm units)
    first = {e: np.nonzero((ep == e) & (kind == 0))[0] for e in np.unique(ep)}
    first = {e: ix[np.argmin(t[ix])] for e, ix in first.items() if len(ix)}
    obj0 = np.stack([st[first[e], 4:7] for e in ep])
    d_start = np.array([st[first[e], 40] for e in ep]) / thr
    moved = np.linalg.norm(st[:, 4:7] - obj0, axis=1) > 0.002
    hand_obj = np.linalg.norm(st[:, :3] - st[:, 4:7], axis=1)
    # tasks without an object (reach-wall: obs[4:7] is a dummy that never moves) keep the plain task-distance target;
    # with the approach term they would learn to approach the dummy (offline ranking inverted, 2026-10-24)
    travel = {e: float(np.linalg.norm(st[(ep == e) & (kind == 0), 4:7] - obj0[(ep == e) & (kind == 0)], axis=1).max(initial=0.0)) for e in np.unique(ep)}
    has_object = np.array([travel[e] > 0.01 for e in ep])
    y_rtc = np.where(moved | ~has_object, np.log1p(st[:, 40] / thr), np.log1p(d_start + hand_obj / 0.02)).astype(np.float32)
    goals_of = {e: np.nonzero((ep == e) & (kind == 1))[0] for e in np.unique(ep)}
    expert_of = {e: np.nonzero((ep == e) & (kind == 0))[0] for e in np.unique(ep)}
    labelled = np.nonzero(kind != 1)[0]
    ratio = st[:, 40] / thr
    pos_pool = labelled[st[labelled, 39] > 0.5]
    neg_pool = labelled[(ratio[labelled] > 1) & (ratio[labelled] < 3) & (st[labelled, 39] < 0.5)]
    pos = rng.choice(pos_pool, size=min(n_pos, len(pos_pool)), replace=False) if len(pos_pool) else np.zeros(0, int)
    neg = rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False) if len(neg_pool) else np.zeros(0, int)
    head = torch.nn.ModuleList([PatchHead(vis.shape[1], vis.shape[2], prop.shape[1]) for _ in range(n_ens)]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    B = 256

    def f(i):  # (E, N, d)
        i = torch.as_tensor(np.asarray(i))
        v, p = vis[i].to(dev).float(), prop[i].to(dev).float()
        return torch.stack([h(v, p) for h in head])

    def dist(a, g):  # (E, N)
        return (f(a) - f(g)).pow(2).sum(-1).add(1e-8).sqrt()

    def goal_for(a):
        return np.array([rng.choice(goals_of[e]) for e in ep[a]])

    def regress(d, y):  # per-member bootstrap mask (Bernoulli 0.63) on the shared batch; one member sees everything
        y = torch.from_numpy(y).to(dev).expand(n_ens, -1)
        m = torch.from_numpy(rng.random((n_ens, d.shape[1])) < 0.63).to(dev)
        m[0] = True
        return F.smooth_l1_loss(d[m], y[m])

    for step in range(steps):
        loss = 0.0
        if mode in ("state", "rtc"):
            a = rng.choice(labelled, size=B)  # expert, candidate and walk frames (kind 0, 2, 3)
            y = y_succ if mode == "state" else y_rtc
            loss = regress(dist(a, goal_for(a)), y[a])
        if mode in ("time", "evid_time"):
            e = rng.choice(list(expert_of), size=B)
            a, g, gap = [], [], []
            for ee in e:
                ix = expert_of[ee]
                i = rng.integers(len(ix))
                if rng.random() < 0.5:  # the goal image itself (one step after the last expert frame)
                    a.append(ix[i]), g.append(rng.choice(goals_of[ee])), gap.append(t[goals_of[ee][0]] - t[ix[i]])
                else:
                    j = rng.integers(i, len(ix))
                    a.append(ix[i]), g.append(ix[j]), gap.append(t[ix[j]] - t[ix[i]])
            a, g = np.array(a), np.array(g)
            loss = loss + regress(dist(a, g), np.log1p(np.maximum(np.array(gap), 0) / 5.0).astype(np.float32))
        if mode in ("evid", "evid_time") and len(pos) and len(neg):
            p = rng.choice(pos, size=B // 2)
            n = rng.choice(neg, size=B // 2)
            loss = loss + dist(p, goal_for(p)).pow(2).mean() + F.relu(1.0 - dist(n, goal_for(n))).pow(2).mean()
        if not torch.is_tensor(loss):  # evid without any success or near-miss frame: nothing to learn
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    head.eval()
    return head, {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "pos_pool": int(len(pos_pool)), "neg_pool": int(len(neg_pool)),
                  "final_loss": float(loss)}


def ordering_accuracy(c, y):
    i, j = np.triu_indices(len(y), 1)
    gap = np.abs(y[i] - y[j])
    keep = gap > np.median(gap)
    if keep.sum() == 0:
        return np.nan
    return float((np.sign(c[i] - c[j]) == np.sign(y[i] - y[j]))[keep].mean())


@torch.no_grad()
def embed(head, vis, prop, dev, bs=512):
    """PatchHead -> (N, d); ModuleList of E heads -> (E, N, d)"""
    members = list(head) if isinstance(head, torch.nn.ModuleList) else [head]
    out = []
    for s in range(0, len(vis), bs):
        v, p = torch.from_numpy(vis[s : s + bs]).to(dev).float(), torch.from_numpy(prop[s : s + bs]).to(dev).float()
        out.append(torch.stack([h(v, p) for h in members]))
    E = torch.cat(out, 1)
    return E if isinstance(head, torch.nn.ModuleList) else E[0]


def ensemble_cost(E, gi, beta=0.0):
    """member embeddings (E, N, d), goal index -> mean_k ||E_k - E_k[gi]||^2 + beta * std_k, as the planner uses it"""
    d = (E - E[:, gi : gi + 1]).pow(2).sum(-1)
    c = d.mean(0)
    if beta and E.shape[0] > 1:
        c = c + beta * d.std(0, unbiased=False)
    return c.cpu().numpy()


@torch.no_grad()
def released(vis, prop, gi, alpha, dev, bs=512):
    gv = torch.from_numpy(vis[gi]).to(dev).float()
    gp = torch.from_numpy(prop[gi]).to(dev).float()
    out = []
    for s in range(0, len(vis), bs):
        v = torch.from_numpy(vis[s : s + bs]).to(dev).float()
        p = torch.from_numpy(prop[s : s + bs]).to(dev).float()
        out.append((v - gv).pow(2).mean((1, 2)) + alpha * (p - gp).pow(2).mean(1))
    return torch.cat(out).cpu().numpy()


def evaluate(T, cost_fns, thr):
    """cost_fns: name -> fn(goal index) -> cost of every test frame. Returns per (cost, goal kind) metrics."""
    st, ep, kind, var, start = T["state"], T["ep"], T["kind"], T["var"], T["start"]
    y = st[:, 40] / thr
    res = {}
    for name, fn in cost_fns.items():
        rows = {"eval_goal": [], "arm_moved": []}
        decoy = []
        for e in np.unique(ep):
            goals = np.nonzero((ep == e) & (kind == 1))[0]
            for gi in goals:
                gk = "eval_goal" if var[gi] == 0 else ("arm_moved" if var[gi] >= 2 else None)
                if gk is None:
                    continue
                c = fn(gi)
                for s in np.unique(start[(ep == e) & (kind == 2)]):
                    ix = np.nonzero((ep == e) & (kind == 2) & (start == s))[0]
                    ys, cs = y[ix], c[ix]
                    if np.nanstd(ys) < 1e-3 or np.isnan(ys).any():
                        continue
                    b = int(np.argmin(cs))
                    rows[gk].append([ordering_accuracy(cs, ys), spearmanr(cs, ys)[0], ys[b] - ys.min(),
                                     float(st[ix[b], 39] > 0.5), float((st[ix, 39] > 0.5).any())])
                if var[gi] == 0:  # decoy test against the eval goal
                    fr = np.nonzero((ep == e) & (kind != 1))[0]
                    hand_d = np.linalg.norm(st[fr, :3] - st[gi, :3], axis=1)
                    succ = fr[st[fr, 39] > 0.5]
                    dec = fr[(hand_d < 0.05) & (y[fr] > 2)]
                    if len(succ) and len(dec):
                        decoy.append(float((c[succ][:, None] < c[dec][None, :]).mean()))
        res[name] = {}
        for gk, r in rows.items():
            if r:
                a = np.array(r, dtype=float)
                res[name][gk] = {"n_sets": len(r), "acc": float(np.nanmean(a[:, 0])), "spearman": float(np.nanmean(a[:, 1])),
                                 "regret": float(np.nanmean(a[:, 2])), "top1": float(a[:, 3].mean()), "any": float(a[:, 4].mean())}
        res[name]["decoy"] = {"n_eps": len(decoy), "pref_success": float(np.mean(decoy)) if decoy else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", nargs="+", required=True, help="one or several pilot-data dirs (one head for all)")
    ap.add_argument("--test_dir", nargs="+", required=True, help="aligned with --task; offline tables per task")
    ap.add_argument("--task", nargs="+", required=True, help="dataset task names (mw-...), aligned with --train_dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--modes", default="rtc,state")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n_pos", type=int, default=64)
    ap.add_argument("--n_neg", type=int, default=64)
    ap.add_argument("--c_head", default="", help="Method C time head (mw_head_train) to score as a baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_ens", type=int, default=1, help="ensemble members per head (planner cost: mean + beta * std)")
    ap.add_argument("--beta", type=float, default=1.0, help="offline: also score the pessimistic ensemble cost with this beta")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda"
    thr = SUCCESS_THRESHOLD.get(args.task[0], 0.05)
    D = load(args.train_dir, args.task)
    kinds = {int(k): int(n) for k, n in zip(*np.unique(D["kind"], return_counts=True))}
    print(f"train {len(D['kind'])} frames / {len(np.unique(D['ep']))} eps from {len(args.train_dir)} task(s); "
          f"kinds {kinds}; train success frames {int((D['state'][:, 39] > 0.5).sum())}", flush=True)
    heads, info = {}, {}
    if args.c_head:
        ck = torch.load(args.c_head, map_location="cpu")
        h = PatchHead(ck["n_patches"], ck["d_in"], ck["d_prop"])
        h.load_state_dict(ck["state_dict"])
        heads["methodC_time"] = h.to(dev).eval()
    for mode in args.modes.split(","):
        h, info[mode] = train_head(D, mode, thr, args.steps, args.n_pos, args.n_neg, args.seed, dev, args.n_ens)
        heads[mode] = h
        torch.save({"state_dict": h[0].state_dict(), "state_dicts": [m.state_dict() for m in h],
                    "n_patches": D["vis"].shape[1], "d_in": D["vis"].shape[2],
                    "d_prop": D["prop"].shape[1], "mode": mode, "task": args.task, "info": info[mode]},
                   os.path.join(args.out_dir, f"head_{mode}.pt"))
        print(mode, json.dumps(info[mode]), flush=True)
    all_res = {}
    for test_dir, task in zip(args.test_dir, args.task):
        T = load(test_dir, task)
        thr = SUCCESS_THRESHOLD.get(task, 0.05)
        cost_fns = {"released_a0": lambda gi, T=T: released(T["vis"], T["prop"], gi, 0.0, dev),
                    "released_a0.1": lambda gi, T=T: released(T["vis"], T["prop"], gi, 0.1, dev)}
        for name, h in heads.items():
            E = embed(h, T["vis"], T["prop"], dev)
            if E.dim() == 2:
                cost_fns[name] = (lambda E: lambda gi: (E - E[gi]).pow(2).sum(-1).cpu().numpy())(E)
            else:  # ensemble: mean cost, and the pessimistic cost the planner would use
                cost_fns[name] = (lambda E: lambda gi: ensemble_cost(E, gi))(E)
                if E.shape[0] > 1:
                    cost_fns[f"{name}+{args.beta:g}std"] = (lambda E: lambda gi: ensemble_cost(E, gi, args.beta))(E)
        res = evaluate(T, cost_fns, thr)
        all_res[task] = res
        print(f"== {task}: test {len(T['kind'])} frames / {len(np.unique(T['ep']))} eps")
        print(f"{'cost':<14}{'goal':<10}{'sets':>5}{'acc':>7}{'rho':>7}{'regret':>8}{'top1':>6}{'any':>6}{'decoy':>7}")
        for name, r in res.items():
            for gk in ("eval_goal", "arm_moved"):
                if gk in r:
                    m = r[gk]
                    dec = r["decoy"]["pref_success"] if gk == "eval_goal" else None
                    print(f"{name:<14}{gk:<10}{m['n_sets']:>5}{m['acc']:>7.3f}{m['spearman']:>7.3f}{m['regret']:>8.2f}{m['top1']:>6.2f}"
                          f"{m['any']:>6.2f}{(f'{dec:.2f}' if dec is not None else '-'):>7}")
    json.dump({"tasks": args.task, "args": vars(args), "train_info": info, "results": all_res},
              open(os.path.join(args.out_dir, "offline.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
