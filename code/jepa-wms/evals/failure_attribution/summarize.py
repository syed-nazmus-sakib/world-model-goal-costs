# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Attributes the failures of a released single-plan planning eval to dynamics, cost landscape or search,
from the per-episode files written by run.py.

    wm fails, sim_latent succeeds        -> dynamics: the same CEM and cost succeed with true dynamics
    sim_latent fails, sim_true succeeds  -> cost landscape: with true dynamics the goal has latent cost 0,
                                            but CEM on the latent cost does not reach it
    both simulator planners fail         -> search: CEM fails even with true dynamics and true cost

Usage:
    python -m evals.failure_attribution.summarize --run_dir <dir> [--out <summary.json>]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    eps = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(args.run_dir, "ep_*.npz")))]
    n = len(eps)
    if not eps:
        parser.error("No completed ep_*.npz files found in --run_dir")
    keys = set(eps[0].files)
    planners = [p for p in PLANNERS if f"{p}_success" in keys]
    meta_file = os.path.join(args.run_dir, "run_meta.json")
    meta = json.load(open(meta_file)) if os.path.exists(meta_file) else {}
    alpha = meta.get("alpha", 0.1)
    kind = meta.get("kind", "wall" if "wall_x" in keys else "pusht")
    get = lambda k: np.array([e[k] for e in eps])
    succ = {p: get(f"{p}_success").astype(bool) for p in planners}
    summary = {"n_episodes": n, "kind": kind, "max_fast_rollout_diff": float(get("fast_rollout_max_abs_diff").max())}

    print(f"{kind}: {n} episodes | max |fast rollout - env rollout| = {summary['max_fast_rollout_diff']}")
    print("\n== success ladder (same CEM budget, same start and goal)")
    summary["success"] = {}
    for p in planners:
        k = int(succ[p].sum())
        lo, hi = wilson(k, n)
        summary["success"][p] = {"k": k, "rate": k / n, "wilson95": [lo, hi]}
        print(f"  {p:11s} {k:3d}/{n} = {100 * k / n:5.1f}%   [{100 * lo:.1f}, {100 * hi:.1f}]")

    if len(planners) == 3:
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
            row = [
                int(((succ["wm"] == a) & (succ["sim_latent"] == b) & (succ["sim_true"] == c)).sum())
                for b in (True, False)
                for c in (True, False)
            ]
            print(f"  wm={'S' if a else 'F'}:  latent=S,true=S {row[0]:3d} | latent=S,true=F {row[1]:3d} | "
                  f"latent=F,true=S {row[2]:3d} | latent=F,true=F {row[3]:3d}")
        summary["wm_success_but_sim_latent_fails"] = int((succ["wm"] & ~succ["sim_latent"]).sum())

    # ---- what the failures look like, per planner
    err_keys = [k[len("wm_") :] for k in sorted(keys) if k.startswith("wm_") and k.endswith("_err")]
    print("\n== final errors (median) of failed / successful episodes")
    summary["final_errors"] = {}
    for p in planners:
        row = {}
        for label, m in (("fail", ~succ[p]), ("success", succ[p])):
            if m.sum():
                row[label] = {k: float(np.median(get(f"{p}_{k}")[m])) for k in err_keys}
        summary["final_errors"][p] = row
        print(f"  {p:11s} " + " | ".join(
            f"{label} (n={int(m.sum())}): " + ", ".join(f"{k} {row[label][k]:.2f}" for k in err_keys)
            for label, m in (("fail", ~succ[p]), ("success", succ[p])) if label in row
        ))

    if kind == "pusht":
        print("\n== Push-T: block-only success (block within 20 px and 20 deg) and pusher within 20 px")
        summary["pusht"] = {}
        for p in planners:
            block_ok = (get(f"{p}_object_err") < 20) & (get(f"{p}_object_ang_err") < np.pi / 9)
            agent_ok = get(f"{p}_agent_err") < 20
            summary["pusht"][p] = {
                "block_only_success": float(block_ok.mean()),
                "pusher_within_20px": float(agent_ok.mean()),
                "block_ok_pusher_not": int((block_ok & ~agent_ok).sum()),
            }
            print(f"  {p:11s} block-only {100 * block_ok.mean():5.1f}% | pusher within 20 px {100 * agent_ok.mean():5.1f}% | "
                  f"block right but pusher not: {int((block_ok & ~agent_ok).sum())}")
    if kind == "wall":
        wall_x, goal, init = get("wall_x"), get("goal_state"), get("init_state")
        print("\n== Wall: which side of the wall the dot ends on")
        summary["wall"] = {}
        for p in planners:
            fs = get(f"{p}_final_state")
            crossed = np.sign(fs[:, 0] - wall_x) == np.sign(goal[:, 0] - wall_x)
            summary["wall"][p] = {
                "ends_on_goal_side": float(crossed.mean()),
                "failures_on_start_side": int((~succ[p] & ~crossed).sum()),
                "failures_on_goal_side": int((~succ[p] & crossed).sum()),
            }
            print(f"  {p:11s} ends on the goal side {100 * crossed.mean():5.1f}% | failures stuck on the start side "
                  f"{int((~succ[p] & ~crossed).sum())}, on the goal side but outside tolerance {int((~succ[p] & crossed).sum())}")
        summary["wall"]["median_travel_px"] = float(np.median(np.linalg.norm(init[:, :2] - goal[:, :2], axis=1)))

    # ---- does the world model itself prefer the expert plan to the plan CEM returned
    wm_pred = get("wm_pred_cost")
    wm_real = get("wm_real_cost_visual")[:, -1] + alpha * get("wm_real_cost_proprio")[:, -1]
    if "expert_pred_cost" in keys:
        exp_pred = get("expert_pred_cost")
        print("\n== world-model cost of the returned plan vs of the expert plan")
        summary["model_prefers_expert"] = {}
        for label, m in (("all", np.ones(n, bool)), ("wm success", succ["wm"]), ("wm failure", ~succ["wm"])):
            if m.sum() == 0:
                continue
            summary["model_prefers_expert"][label] = {
                "n": int(m.sum()),
                "frac_expert_scored_better": float((exp_pred < wm_pred)[m].mean()),
                "median_plan_pred_cost": float(np.median(wm_pred[m])),
                "median_expert_pred_cost": float(np.median(exp_pred[m])),
            }
            print(f"  {label:11s} n={int(m.sum()):3d}: model scores expert better in {100 * (exp_pred < wm_pred)[m].mean():5.1f}% | "
                  f"median predicted cost plan {np.median(wm_pred[m]):.4f}, expert {np.median(exp_pred[m]):.4f}")

    # ---- dynamics error of the executed plan
    gap = get("pred_vs_real_visual")
    print("\n== predicted vs realized visual latent of the executed plan, per model step (median)")
    summary["pred_vs_real_visual_by_step"] = {}
    summary["plan_cost_predicted_vs_real_median"] = {}
    for label, m in (("wm success", succ["wm"]), ("wm failure", ~succ["wm"])):
        if m.sum():
            summary["pred_vs_real_visual_by_step"][label] = np.median(gap[m], 0).tolist()
            summary["plan_cost_predicted_vs_real_median"][label] = [float(np.median(wm_pred[m])), float(np.median(wm_real[m]))]
            print(f"  {label:11s}", np.round(np.median(gap[m], 0), 4),
                  f"| final cost of the plan: predicted {np.median(wm_pred[m]):.4f}, real {np.median(wm_real[m]):.4f}")

    out = args.out or os.path.join(args.run_dir, "summary.json")
    save_json(out, summary, indent=1)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
