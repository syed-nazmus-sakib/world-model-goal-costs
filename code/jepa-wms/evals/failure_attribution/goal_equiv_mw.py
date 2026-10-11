# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Task-equivalence test on Metaworld (levels 1-3, exact simulator dynamics, released encoder and cost).

For each episode of the released eval, goal variants with the SAME object goal and different arm poses
(goal_variants.make_arm_variants; variant 0 = the eval's own goal). Then, from the episode's start state:
  L1 cost sensitivity  M random candidate futures (30 env steps, exact simulator) scored against each goal
                       variant with the released latent cost: relative change of the same candidates' cost
  L2 ranking flips     fraction of candidate pairs ordered differently under two equivalent goals
                       (true task cost: identical object goal -> 0 by construction; we also report the latent
                       cost's rank correlation with the true cost under each variant)
  L3 selection         which candidate the cost selects: top-1 agreement and top-10 overlap between two equivalent
                       goals (deterministic: same candidates, no search noise). Candidates: half CEM-initial
                       random sequences, half the expert's own first actions + noise (so the object does move).
  L3b first action     CEM (simulator dynamics, latent cost, identical seed) per goal variant vs noise floor

    python -m evals.failure_attribution.goal_equiv_mw --fname <mw config> --out_dir <dir> --episodes 24
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from evals.failure_attribution.io import save_episode, save_json
import torch
from scipy.stats import spearmanr

from evals.failure_attribution.adapters import get_adapter
from evals.failure_attribution.goal_variants import goal_td, hook_expert_capture, make_arm_variants, robot_free_frame
from evals.failure_attribution.mw_sim import capture
from evals.failure_attribution.run import SimUnroll, build, load_args
from evals.failure_attribution.sim_worker import SimPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--robot_free", action="store_true", help="goal and candidates rendered without the robot")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    args_eval = load_args(args.fname)
    task = args_eval["task_specification"]["task"]
    adapter = get_adapter(task)
    pool = SimPool(adapter.kind, args.num_workers)

    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    from evals.utils import prepare_obs

    cfg, agent, env, evaluator = build(args_eval)
    hook_expert_capture(evaluator)
    model, device = agent.model, agent.device
    obs_type = cfg.task_specification.obs
    alpha = cfg.planner.planning_objective.alpha
    unroll = SimUnroll(agent, pool, adapter, "latent")
    planner = CEMPlanner(unroll=unroll, action_dim=model.action_dim, local_generator=torch.Generator(device="cuda:0"),
                         **{**cfg.planner, "decode_each_iteration": False})
    rng = np.random.default_rng(0)
    for ep in range(args.episodes):
        path = os.path.join(args.out_dir, f"ep_{ep:03d}.npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        env.reset(seed=ep_seed, task_idx=-1)
        u = env.proprio_env.unwrapped
        u._freeze_rand_vec, u.seeded_rand_vec = False, True
        env.seed(ep_seed)
        init_obs, goal_obs, _, _ = evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=-1)
        var = make_arm_variants(evaluator.expert_env, evaluator.expert_init_snap, evaluator.expert_final_snap,
                                args.variants, seed=ep)
        if args.robot_free:  # all arm variants then render identically: keep the eval goal only
            var = [(var[0][0], robot_free_frame(evaluator.expert_env, var[0][0]), var[0][2], var[0][3])]
        V = len(var)
        try:  # each episode's expert env holds its own EGL context: free it, or rendering fails after ~9 episodes
            evaluator.expert_env.close()
        except Exception:
            pass
        evaluator.expert_env = None
        goal_encs = [model.encode(prepare_obs(obs_type, goal_td(f, o)).unsqueeze(0).to(device), act=False).detach()
                     for _, f, o, _ in var]
        spec = (task, capture(env.unwrapped), args.robot_free)
        unroll.spec = spec
        z = model.encode(prepare_obs(obs_type, init_obs).to(device).unsqueeze(0), act=True)
        steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)

        # L1 / L2: fixed candidate futures from the start state, scored against every goal variant
        H = min(cfg.planner.horizon, steps_left)
        M2 = args.candidates // 2
        rand = torch.randn(H, M2, model.action_dim, device=device)  # CEM's initial distribution
        exp = evaluator.expert_actions[0][: H * cfg.frameskip]  # (H*f, 4) raw env actions
        exp = agent.preprocessor.normalize_actions(exp).reshape(H, 1, -1).to(device)
        noise = torch.tensor(np.r_[0.0, np.linspace(0.05, 1.0, args.candidates - M2 - 1)], device=device, dtype=torch.float32)
        expn = exp + noise.view(1, -1, 1) * torch.randn(H, len(noise), model.action_dim, device=device)
        cand = torch.cat([rand, expn], 1)
        with torch.no_grad():
            enc = unroll(z, act_suffix=cand)
            costs = np.stack([ReprTargetDistMPCObjective(cfg, target_enc=g, alpha=alpha)(enc, cand).cpu().numpy()
                              for g in goal_encs])  # (V, M)
            states, _ = pool.rollout(spec, unroll.env_actions(cand), False)
        true_cost = adapter.true_cost(states.astype(np.float64), None)
        # L3: CEM first action per goal variant, identical seed; noise floor = variant 1 with a second seed
        first, plans = [], []
        for vi, g in enumerate(goal_encs):
            planner.set_objective(ReprTargetDistMPCObjective(cfg, target_enc=g, alpha=alpha))
            planner.local_generator.manual_seed(int(ep_seed))
            res = planner.plan(z, steps_left=steps_left)
            plans.append(res.actions.cpu().numpy())
        planner.set_objective(ReprTargetDistMPCObjective(cfg, target_enc=goal_encs[min(1, V - 1)], alpha=alpha))
        planner.local_generator.manual_seed(int(ep_seed) + 7)
        noise_plan = planner.plan(z, steps_left=steps_left).actions.cpu().numpy()
        moved = np.linalg.norm(states[:, 4:7] - np.asarray(env.unwrapped._get_obs())[4:7], axis=1)
        save_episode(path, costs=costs, true_cost=true_cost, puck_moved=moved, plans=np.array(plans), noise_plan=noise_plan,
                 hand=np.array([v[2][:3] for v in var]), obj=np.array([v[3] for v in var]),
                 goal_frames=np.array([v[1] for v in var]))
        print(f"ep {ep:03d}: {V} goal variants, {time.time() - t0:.0f}s", flush=True)
    pool.close()
    summarize(args.out_dir)


def summarize(out_dir):
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("ep_"))
    sens, flips, rho, act, act_noise, flips_v0, top1, top10, moved = [], [], [], [], [], [], [], [], []
    for f in files:
        d = np.load(os.path.join(out_dir, f))
        c, t = d["costs"], d["true_cost"]
        V = len(c)
        for a, b in itertools.combinations(range(1, V), 2):  # variants 1..V-1 share the exact object pose
            sens.append(np.median(np.abs(c[a] - c[b]) / np.minimum(c[a], c[b])))
            i, j = np.triu_indices(c.shape[1], 1)
            flips.append(np.mean(np.sign(c[a, i] - c[a, j]) != np.sign(c[b, i] - c[b, j])))
            act.append(np.linalg.norm(d["plans"][a][0] - d["plans"][b][0]))
            top1.append(float(np.argmin(c[a]) == np.argmin(c[b])))
            top10.append(len(set(np.argsort(c[a])[:10]) & set(np.argsort(c[b])[:10])) / 10)
        for a in range(1, V):
            i, j = np.triu_indices(c.shape[1], 1)
            flips_v0.append(np.mean(np.sign(c[0, i] - c[0, j]) != np.sign(c[a, i] - c[a, j])))
        rho.append([spearmanr(c[v], t).correlation for v in range(V)])
        moved.append(float(np.mean(t < t.max() - 1e-3)))
        act_noise.append(np.linalg.norm(d["plans"][min(1, V - 1)][0] - d["noise_plan"][0]))
    rho = np.array(rho)
    bins = {"moved<0.5mm": (0, 5e-4), "0.5-5mm": (5e-4, 5e-3), "5mm-2cm": (5e-3, 2e-2), ">2cm": (2e-2, 9)}
    by_move = {}
    for name, (lo, hi) in bins.items():
        r = []
        for f in files:
            d = np.load(os.path.join(out_dir, f))
            if "puck_moved" not in d.files:
                continue
            m = (d["puck_moved"] >= lo) & (d["puck_moved"] < hi)
            if m.sum() >= 5 and np.std(d["true_cost"][m]) > 0:
                r.append(spearmanr(d["costs"][0][m], d["true_cost"][m]).correlation)
        by_move[name] = [round(float(np.nanmean(r)), 3) if r else None, len(r)]
    out = {"spearman_latent_vs_true_by_puck_move (mean, n_episodes)": by_move,
        "episodes": len(files),
        "L1_median_rel_cost_change_between_equivalent_goals": float(np.median(sens)) if sens else None,
        "L2_ranking_flip_rate_between_equivalent_goals": float(np.mean(flips)) if flips else None,
        "L2_ranking_flip_rate_vs_eval_goal": float(np.mean(flips_v0)) if flips_v0 else None,
        "L2_true_cost_flip_rate": 0.0,
        "spearman_latent_vs_true_per_variant(mean over episodes)": np.nanmean(rho, 0).round(3).tolist(),
        "L3_top1_candidate_agreement_between_equivalent_goals": float(np.mean(top1)) if top1 else None,
        "L3_top10_overlap_between_equivalent_goals": float(np.mean(top10)) if top10 else None,
        "fraction_of_candidates_that_move_the_object": float(np.mean(moved)),
        "L3b_first_action_change_between_equivalent_goals": float(np.median(act)) if act else None,
        "L3_first_action_noise_floor_same_goal_other_seed": float(np.median(act_noise)),
    }
    print(json.dumps(out, indent=1))
    save_json(os.path.join(out_dir, "summary.json"), out, indent=1)


if __name__ == "__main__":
    main()
