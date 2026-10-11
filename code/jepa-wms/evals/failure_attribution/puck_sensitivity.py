# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Detection threshold of the released latent goal cost for small-object progress (Metaworld push).

For each episode: the puck resting at its goal x-y (as in goal_variants), arm at its initial pose. The puck is
shifted by d in 4 directions; each scene is rendered with and without the robot and scored against the
unshifted scene (visual part of the released cost). We report the cost as a function of d, its monotonicity,
and, for comparison, how far the CEM-initial random candidates actually move the puck (from goal_equiv_mw).

    python -m evals.failure_attribution.puck_sensitivity --fname <mw push config> --episodes 12
"""

import argparse
import json
import os

import mujoco
import numpy as np
import torch
from scipy.stats import spearmanr

from evals.failure_attribution.goal_variants import _obj_qpos_adr, find_mw_wrapper, goal_td, hook_expert_capture
from evals.failure_attribution.mw_sim import capture, render_hidden, restore, robot_geoms
from evals.failure_attribution.run import build, load_args

SHIFTS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    from evals.utils import prepare_obs

    cfg, agent, env, evaluator = build(load_args(args.fname))
    hook_expert_capture(evaluator)
    model, device = agent.model, agent.device
    obs_type = cfg.task_specification.obs
    res = {"robot": [], "robot_free": []}
    for ep in range(args.episodes):
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        env.reset(seed=ep_seed, task_idx=-1)
        u = env.proprio_env.unwrapped
        u._freeze_rand_vec, u.seeded_rand_vec = False, True
        env.seed(ep_seed)
        evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=-1)
        w = find_mw_wrapper(evaluator.expert_env)
        mw = w.unwrapped
        qa, va = _obj_qpos_adr(mw)
        restore(mw, evaluator.expert_final_snap)
        goal_xy = mw.data.qpos[qa:qa + 2].copy()
        restore(mw, evaluator.expert_init_snap)
        rest = mw.data.qpos[qa:qa + 7].copy()
        rest[:2] = goal_xy
        rg = robot_geoms(mw.model)
        frames = {"robot": [], "robot_free": []}
        obs = []
        for d in SHIFTS:
            for ang in (0, 90, 180, 270) if d > 0 else (0,):
                restore(mw, evaluator.expert_init_snap)
                q = rest.copy()
                q[:2] += d * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
                mw.data.qpos[qa:qa + 7] = q
                mw.data.qvel[va:va + 6] = 0
                mujoco.mj_forward(mw.model, mw.data)
                frames["robot"].append(w.render().copy())
                frames["robot_free"].append(render_hidden(w, rg))
                obs.append(np.asarray(mw._get_obs(), dtype=np.float32))
        dist = np.array([0.0] + [d for d in SHIFTS[1:] for _ in range(4)])
        for kind in res:
            with torch.no_grad():
                encs = [model.encode(prepare_obs(obs_type, goal_td(f, o)).unsqueeze(0).to(device), act=False)
                        for f, o in zip(frames[kind], obs)]
            vis = torch.cat([e["visual"] for e in encs], 1)  # (1, N, ...)
            c = (vis - vis[:, :1]).pow(2).flatten(2).mean(-1)[0].cpu().numpy()
            res[kind].append(c)
        try:
            evaluator.expert_env.close()
        except Exception:
            pass
        print(f"ep {ep}: robot-free cost at shifts {[round(float(np.mean(res['robot_free'][-1][dist == d])), 4) for d in SHIFTS]}", flush=True)
    out = {}
    for kind, C in res.items():
        C = np.array(C)  # (E, N)
        per_d = {f"{d * 100:g}cm": float(np.median(C[:, dist == d])) for d in SHIFTS}
        rho = float(np.nanmean([spearmanr(dist, c).correlation for c in C]))
        out[kind] = {"median_cost_by_shift": per_d, "spearman_cost_vs_shift": rho}
    # how far do CEM-initial random candidates move the puck? (from the goal_equiv_mw run, if present)
    ge = os.path.join(os.environ["JEPAWM_LOGS"], "failure_attribution", "goal_equiv_mw_push")
    if os.path.isdir(ge):
        moves = []
        for f in sorted(os.listdir(ge)):
            if f.startswith("ep_"):
                t = np.load(os.path.join(ge, f))["true_cost"]
                M2 = len(t) // 2
                moves.append(np.abs(t[:M2] - t[:M2].max()) * 0.05)  # cost = obj_to_target / 0.05
        moves = np.concatenate(moves)
        out["random_candidates_puck_progress"] = {
            "frac_any_(>0.5mm)": float(np.mean(moves > 0.0005)), "frac_>2mm": float(np.mean(moves > 0.002)),
            "frac_>1cm": float(np.mean(moves > 0.01)), "median_m": float(np.median(moves))}
    print(json.dumps(out, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
