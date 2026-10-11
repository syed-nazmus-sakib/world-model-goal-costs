# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Attributes the failures of the released Push-T planning eval to dynamics, cost landscape or search,
from the per-episode files written by run.py.

    wm fails, sim_latent succeeds        -> dynamics: the same CEM and cost succeed with true dynamics
    sim_latent fails, sim_true succeeds  -> cost landscape: the goal is the global minimum of the latent
                                            cost under true dynamics (cost 0), but CEM does not reach it
    both simulator planners fail         -> search: CEM fails even with true dynamics and true cost

Usage:
    python -m evals.pusht_failure_attribution.summarize --run_dir <dir> [--out <summary.json>]
"""

import argparse
import glob
import json
import os

import numpy as np

from evals.failure_attribution.analysis import failure_masks
from evals.failure_attribution.io import save_json

PLANNERS = ["wm", "sim_latent", "sim_true"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (float(mid - half), float(mid + half))


def angle_diff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    eps = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(args.run_dir, "ep_*.npz")))]
    n = len(eps)
    if not eps:
        parser.error("No completed ep_*.npz files found in --run_dir")
    has_oracles = "sim_true_success" in eps[0].files
    planners = PLANNERS if has_oracles else ["wm"]
    alpha = 0.1
    meta_file = os.path.join(args.run_dir, "run_meta.json")
    if os.path.exists(meta_file):
        alpha = json.load(open(meta_file))["alpha"]
    get = lambda k: np.array([e[k] for e in eps])
    succ = {p: get(f"{p}_success").astype(bool) for p in planners}
    summary = {"n_episodes": n, "max_fast_rollout_diff": float(get("fast_rollout_max_abs_diff").max())}

    print(f"{n} episodes | max |fast rollout - env rollout| = {summary['max_fast_rollout_diff']}")
    print("\n== success ladder (same CEM budget, same start and goal)")
    summary["success"] = {}
    for p in planners:
        k = int(succ[p].sum())
        lo, hi = wilson(k, n)
        summary["success"][p] = {"k": k, "rate": k / n, "wilson95": [lo, hi]}
        print(f"  {p:11s} {k:3d}/{n} = {100 * k / n:5.1f}%   [{100 * lo:.1f}, {100 * hi:.1f}]")
    print(f"  {'expert':11s} {n:3d}/{n} = 100.0%   (the goal is the end of the expert segment)")

    if has_oracles:
        fail = ~succ["wm"]
        masks = failure_masks(succ["wm"], succ["sim_latent"], succ["sim_true"])
        buckets = {
            "dynamics (sim_latent succeeds)": masks["dynamics"],
            "cost landscape (sim_latent fails, sim_true succeeds)": masks["cost"],
            "search (both simulator planners fail)": masks["search"],
        }
        print(f"\n== attribution of the {int(fail.sum())} world-model failures")
        summary["attribution"] = {}
        for name, m in buckets.items():
            summary["attribution"][name] = {"k": int(m.sum()), "episodes": np.nonzero(m)[0].tolist()}
            print(f"  {name:55s} {int(m.sum()):3d}  ({100 * m.sum() / max(fail.sum(), 1):.0f}% of failures)")
        print("\n== cross-tab (rows wm, cols sim_latent / sim_true; S = success)")
        for a in (True, False):
            row = []
            for b in (True, False):
                for c in (True, False):
                    row.append(int(((succ["wm"] == a) & (succ["sim_latent"] == b) & (succ["sim_true"] == c)).sum()))
            print(f"  wm={'S' if a else 'F'}:  latent=S,true=S {row[0]:3d} | latent=S,true=F {row[1]:3d} | "
                  f"latent=F,true=S {row[2]:3d} | latent=F,true=F {row[3]:3d}")
        summary["wm_success_but_sim_latent_fails"] = int((succ["wm"] & ~succ["sim_latent"]).sum())

    # ---- does the world model itself prefer the expert plan to the plan CEM returned
    wm_pred, exp_pred = get("wm_pred_cost"), get("expert_pred_cost")
    wm_real = get("wm_real_cost_visual")[:, -1] + alpha * get("wm_real_cost_proprio")[:, -1]
    prefers_expert = exp_pred < wm_pred
    print("\n== world-model cost of the returned plan vs of the expert plan")
    summary["model_prefers_expert"] = {}
    for label, m in (("all", np.ones(n, bool)), ("wm success", succ["wm"]), ("wm failure", ~succ["wm"])):
        if m.sum() == 0:
            continue
        summary["model_prefers_expert"][label] = {
            "n": int(m.sum()),
            "frac_expert_scored_better": float(prefers_expert[m].mean()),
            "median_plan_pred_cost": float(np.median(wm_pred[m])),
            "median_expert_pred_cost": float(np.median(exp_pred[m])),
            "median_plan_real_cost": float(np.median(wm_real[m])),
        }
        print(f"  {label:11s} n={int(m.sum()):3d}: model scores expert better in {100 * prefers_expert[m].mean():5.1f}% | "
              f"median predicted cost plan {np.median(wm_pred[m]):.4f}, expert {np.median(exp_pred[m]):.4f} | "
              f"real latent cost of plan {np.median(wm_real[m]):.4f}")

    # ---- dynamics error of the executed plan, per model step
    gap = get("pred_vs_real_visual")
    print("\n== predicted vs realized visual latent of the executed plan, per model step (median)")
    summary["pred_vs_real_visual_by_step"] = {}
    for label, m in (("wm success", succ["wm"]), ("wm failure", ~succ["wm"])):
        if m.sum():
            summary["pred_vs_real_visual_by_step"][label] = np.median(gap[m], 0).tolist()
            print(f"  {label:11s}", np.round(np.median(gap[m], 0), 4))
    optimism = wm_real - wm_pred
    summary["real_minus_pred_cost_median"] = {
        "wm success": float(np.median(optimism[succ["wm"]])) if succ["wm"].any() else None,
        "wm failure": float(np.median(optimism[~succ["wm"]])) if (~succ["wm"]).any() else None,
    }
    print("  real - predicted final cost (median):", summary["real_minus_pred_cost_median"])

    # ---- how the task difficulty relates to failures
    init, goal = get("init_state"), get("goal_state")
    rot = np.degrees(angle_diff(init[:, 4], goal[:, 4]))
    trans = np.linalg.norm(init[:, 2:4] - goal[:, 2:4], axis=1)
    print("\n== success by block rotation required between init and goal")
    summary["success_by_required_rotation_deg"] = {}
    for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 181)):
        m = (rot >= lo) & (rot < hi)
        if m.sum() == 0:
            continue
        row = {p: float(succ[p][m].mean()) for p in planners}
        summary["success_by_required_rotation_deg"][f"{lo}-{hi}"] = {"n": int(m.sum()), **row}
        print(f"  {lo:3d}-{hi:3d} deg n={int(m.sum()):3d} (median translation {np.median(trans[m]):5.1f} px): "
              + "  ".join(f"{p} {100 * row[p]:5.1f}%" for p in planners))

    out = args.out or os.path.join(args.run_dir, "summary.json")
    save_json(out, summary, indent=1)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
