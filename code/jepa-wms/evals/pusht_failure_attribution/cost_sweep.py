# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Sweeps the AGGREGATION of the latent goal cost over patches (evals/failure_attribution/costs.py), with the
simulator as dynamics. Same episodes, start, goal, CEM budget and CEM samples as alpha_sweep / run.py.
cost "l2" with alpha 0.1 must reproduce the sim_latent planner of run.py.

The attribution run shows that most failures on Push-T leave the block at the goal and the pusher far
from its goal position, with or without the world model. This re-plans the episodes saved by run.py
(same start, goal, CEM budget and CEM samples as its sim_latent planner) for several values of alpha,
to see whether the cost weighting alone explains those failures. alpha = 0.1 must reproduce sim_latent.

Usage:
    python -m evals.pusht_failure_attribution.alpha_sweep --fname <eval_config.yaml> --run_dir <run.py out_dir> \
        --alphas 0 0.1 1 3 10 30 [--episodes 96]
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from evals.failure_attribution.io import save_episode, save_json
import torch
from tensordict import TensorDict

from evals.pusht_failure_attribution.sim_worker import SimPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--costs", nargs="+", default=["l2", "topk:8", "pmean:4"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()
    pool = SimPool(args.num_workers)

    from evals.pusht_failure_attribution.run import SimUnroll, build, true_cost
    from evals.failure_attribution.costs import PatchAggMPCObjective
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner

    cfg, agent, env, _ = build(args.fname)
    model, device = agent.model, agent.device
    files = sorted(glob.glob(os.path.join(args.run_dir, "ep_*.npz")))[: args.episodes]
    sim = SimUnroll(agent, pool, "latent")
    to_env = SimUnroll(agent, pool, "state").env_actions
    planner = CEMPlanner(
        unroll=sim,
        action_dim=model.action_dim,
        local_generator=torch.Generator(device="cuda:0"),
        **{**cfg.planner, "decode_each_iteration": False},
    )
    out_dir = os.path.join(args.run_dir, "cost_sweep")
    os.makedirs(out_dir, exist_ok=True)
    tag = lambda c: f"{c.replace(':', '')}_alpha{args.alpha:g}"
    done = {c: os.path.join(out_dir, f"{tag(c)}.npz") for c in args.costs}

    for a in args.costs:
        if os.path.exists(done[a]) and len(np.load(done[a])["success"]) == len(files):
            print(f"cost {a}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        rows = {k: [] for k in ("episode", "final_state", "success", "pusher_err", "block_err", "ang_err", "cem_cost")}
        mismatch = 0.0
        for f in files:
            ep = np.load(f, allow_pickle=True)
            episode = (int(ep["ep_seed"]), ep["init_state"], str(ep["shape"]))
            sim.episode = episode
            goal_state = ep["goal_state"]
            # the goal observation is the last frame of the expert segment, re-rendered by the exact simulator path
            a_exp = torch.from_numpy(ep["expert_plan"]).to(device)
            states, frames = pool.rollout(*episode, to_env(a_exp.unsqueeze(1)), True)
            assert np.abs(states[0] - goal_state).max() < 1e-4, "expert replay does not reach the saved goal state"
            goal_td = TensorDict(
                {
                    "visual": torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(1),
                    "proprio": torch.from_numpy(states[:, [0, 1, 5, 6]]).unsqueeze(1),
                },
                batch_size=[1],
            )
            goal_enc = model.encode(goal_td.to(device), act=False).detach()
            planner.set_objective(PatchAggMPCObjective(cfg, target_enc=goal_enc, cost=a, alpha=args.alpha))
            # same CEM samples as the sim_latent planner of run.py
            planner.local_generator.manual_seed(int(ep["ep_seed"]) + 1000)
            res = planner.plan(None, steps_left=cfg.planner.horizon)
            st, _ = pool.rollout(*episode, to_env(res.actions.unsqueeze(1)), False)
            st = st[0].astype(np.float64)
            success = bool(env.eval_state(goal_state, st)["success"])
            _, _, ang = true_cost(st[None], goal_state)
            rows["episode"].append(int(ep["episode"]))
            rows["final_state"].append(st)
            rows["success"].append(success)
            rows["pusher_err"].append(np.linalg.norm(st[:2] - goal_state[:2]))
            rows["block_err"].append(np.linalg.norm(st[2:4] - goal_state[2:4]))
            rows["ang_err"].append(ang[0])
            rows["cem_cost"].append(float(res.losses[-1]))
            if a == "l2" and abs(args.alpha - 0.1) < 1e-9 and "sim_latent_final_state" in ep.files:
                mismatch = max(mismatch, float(np.abs(st - ep["sim_latent_final_state"]).max()))
            n = len(rows["success"])
            print(
                f"ep {int(ep['episode']):03d} cost {a}: {'S' if success else 'F'} | pusher err {rows['pusher_err'][-1]:6.1f} px, "
                f"block err {rows['block_err'][-1]:5.1f} px | success so far {sum(rows['success'])}/{n} | "
                f"{(time.time() - t0) / n:.0f}s per episode",
                flush=True,
            )
        save_episode(done[a], **{k: np.array(v) for k, v in rows.items()})
        if a == "l2" and abs(args.alpha - 0.1) < 1e-9:
            print(f"l2 alpha 0.1: max |final state - sim_latent of run.py| = {mismatch}", flush=True)

    print("\n== simulator dynamics + latent cost, by patch aggregation")
    for c in args.costs:
        r = np.load(done[c])
        print(f"{c:10s} official {100 * r['success'].mean():5.1f}% | pusher<20px {100 * (r['pusher_err'] < 20).mean():5.1f}% "
              f"| block<20px {100 * (r['block_err'] < 20).mean():5.1f}% | n={len(r['success'])}")


if __name__ == "__main__":
    main()
