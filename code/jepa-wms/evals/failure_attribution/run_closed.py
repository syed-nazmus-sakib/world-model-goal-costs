# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Closed-loop failure attribution, for evals that replan (Metaworld: horizon 6, 3 model steps executed,
100-step episodes).

Each episode of the released eval is played three times from the identical start state and goal, each time
driven end to end by a different planner, replanning at the same points:

    wm            world-model dynamics + latent goal cost + CEM        (the released eval, same RNG)
    sim_latent    simulator dynamics   + latent goal cost + CEM        (removes dynamics error)
    sim_true      simulator dynamics   + true state cost  + CEM        (removes dynamics and cost error)

At every replan of an oracle, the simulator copy used inside CEM starts from an exact snapshot of the real
env (evals/failure_attribution/mw_sim.py; tests/test_fast_rollout_mw.py checks equality). Success is the
eval's own: the env's success flag at the last step of the episode.

Usage:
    python -m evals.failure_attribution.run_closed --fname <mw eval config> --out_dir <dir> [--episodes 48]
"""

import argparse
import json
import os
import time

import numpy as np

from evals.failure_attribution.io import save_episode, save_json
import torch
from einops import rearrange

from evals.failure_attribution.adapters import get_adapter
from evals.failure_attribution.mw_sim import capture, capture_env, restore_env
from evals.failure_attribution.run import CEMProgress, SimUnroll, TrueCostObjective, build, load_args
from evals.failure_attribution.sim_worker import SimPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--episodes", type=int, default=None, help="defaults to meta.eval_episodes of the config")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=None, help="overrides the proprio weight of the planning cost")
    parser.add_argument("--cost", default="l2", help="patch aggregation of the latent cost: l2 (released), topk:K, pmean:P")
    parser.add_argument("--planners", nargs="+", default=["wm", "sim_latent", "sim_true"],
                        help="also sim_latent_reach / sim_true_reach: the same cost inside the reach-then-change oracle (costs.py)")
    parser.add_argument("--reach_eps", type=float, default=0.002, help="object displacement (m) that counts as 'moved' in the reach oracle")
    parser.add_argument("--reach_r0", type=float, default=0.02, help="hand-to-object distance (m) below which the reach term saturates")
    parser.add_argument("--reach_r0_img", type=float, default=1.0, help="--cost reach: grid distance (patches) below which the reach term saturates")
    parser.add_argument("--goal_variant", type=int, default=0,
                        help="0: the eval's goal; k>0: same object goal, arm moved to pose k (goal_variants.py)")
    parser.add_argument("--orbit", type=str, default="",
                        help="comma list of arm-pose goal variants whose encodings are averaged into one goal latent "
                             "(oracle goal-set upper bound, e.g. 0,1,2,3,4)")
    parser.add_argument("--wm_set", type=int, default=0,
                        help="K: goal latent = mean over the goal and K world-model free-motion rollouts (goalset.py)")
    parser.add_argument("--wm_h", type=int, default=8)
    parser.add_argument("--wm_mag", type=float, default=1.0)
    parser.add_argument("--wm_gate", type=float, default=0.0, help="fraction of rollouts kept by the reversibility gate (0 = no gate)")
    parser.add_argument("--robot_free", action="store_true",
                        help="oracle: goal and simulator candidate frames rendered without the robot (sim planners only)")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    args_eval = load_args(args.fname)
    task = args_eval["task_specification"]["task"]
    adapter = get_adapter(task)
    pool = SimPool(adapter.kind, args.num_workers)  # spawned before CUDA is initialized

    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    from evals.simu_env_planning.planning.utils import make_td
    from evals.utils import prepare_obs

    cfg, agent, env, evaluator = build(args_eval, alpha=args.alpha)
    if args.goal_variant or args.robot_free or args.orbit:
        from evals.failure_attribution.goal_variants import goal_td, hook_expert_capture, make_arm_variants, robot_free_frame

        hook_expert_capture(evaluator)
        assert not (args.robot_free and "wm" in args.planners), "the robot-free oracle needs simulator rendering"
    assert cfg.planner.planner_name == "cem"
    model, device = agent.model, agent.device
    obs_type = cfg.task_specification.obs
    n_episodes = args.episodes or cfg.meta.eval_episodes

    sim_unrolls = {m: SimUnroll(agent, pool, adapter, m) for m in ("latent", "state")}
    oracles = {}
    for name, mode in (("sim_latent", "latent"), ("sim_true", "state"), ("sim_latent_reach", "latent"), ("sim_true_reach", "state")):
        if name in args.planners:
            oracles[name] = (
                CEMPlanner(
                    unroll=sim_unrolls[mode],
                    action_dim=model.action_dim,
                    local_generator=torch.Generator(device="cuda:0"),
                    **{**cfg.planner, "decode_each_iteration": False},
                ),
                sim_unrolls[mode],
            )
    progress = {"wm": CEMProgress(agent.planner)}
    progress.update({name: CEMProgress(p) for name, (p, _) in oracles.items()})

    start = 0
    done = sorted(f for f in os.listdir(args.out_dir) if f.startswith("ep_") and f.endswith(".npz"))
    if done:
        last = np.load(os.path.join(args.out_dir, done[-1]))
        agent.local_generator.set_state(torch.from_numpy(last["cpu_generator_state"]))
        agent.local_gpu_generator.set_state(torch.from_numpy(last["gpu_generator_state"]))
        start = int(last["episode"]) + 1
        print(f"resuming at episode {start}", flush=True)

    def denorm(a):
        a = rearrange(a.cpu(), "t (f d) -> (t f) d", d=env.action_dim)
        return agent.preprocessor.denormalize_actions(a)

    times = []
    for ep in range(start, n_episodes):
        t0 = time.time()
        # ---- episode setup, exactly as PlanEvaluator.eval
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        evaluator.expert_actions = None
        env.reset(seed=ep_seed, task_idx=-1)
        unwrapped = env.proprio_env.unwrapped
        unwrapped._freeze_rand_vec = False
        unwrapped.seeded_rand_vec = True
        env.seed(ep_seed)
        init_obs, goal_obs, _, expert_success = evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=-1)
        if goal_obs["visual"].shape[0] > model.tubelet_size_enc:
            goal_obs["visual"] = goal_obs["visual"][-model.tubelet_size_enc :]
        goal_info = {}
        orbit_ids = [int(x) for x in args.orbit.split(",")] if args.orbit else []
        if args.goal_variant or args.robot_free or orbit_ids:
            var = make_arm_variants(evaluator.expert_env, evaluator.expert_init_snap, evaluator.expert_final_snap,
                                    max([args.goal_variant, 0] + orbit_ids), seed=ep)
            if args.goal_variant >= len(var):
                raise RuntimeError(f"Episode {ep}: requested goal variant {args.goal_variant}, but only {len(var) - 1} were generated")
            k = args.goal_variant
            g_snap, g_frame, g_obs, g_obj = var[k]
            if args.robot_free:
                g_frame = robot_free_frame(evaluator.expert_env, g_snap)
            goal_obs = goal_td(g_frame, g_obs)
            if orbit_ids:  # mean encoding over the goal set (squared distance to the mean latent = mean squared distance)
                orbit_encs = []
                for kk in orbit_ids:
                    _, fr, ob, _ = var[min(kk, len(var) - 1)]
                    orbit_encs.append(agent.model.encode(prepare_obs(obs_type, goal_td(fr, ob)).unsqueeze(0).to(agent.device),
                                                         act=False).detach())
            goal_info = {"goal_variant_used": k, "goal_hand": g_obs[:3], "goal_obj": g_obj, "goal_frame": g_frame}
            try:
                evaluator.expert_env.close()
            except Exception:
                pass
            evaluator.expert_env = None
        agent.set_goal(prepare_obs(obs_type, goal_obs))
        if args.wm_set:
            from evals.failure_attribution.goalset import build_goalset_target
            from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

            tgt, gerr = build_goalset_target(agent, agent.goal_state_enc, K=args.wm_set, H=args.wm_h, mag=args.wm_mag,
                                             gate_q=args.wm_gate, seed=ep)
            agent.goal_state_enc = tgt
            agent.objective = ReprTargetDistMPCObjective(cfg, target_enc=tgt, **cfg.planner.planning_objective)
            agent.planner.set_objective(agent.objective)
            goal_info["wm_set_cycle_err"] = gerr.numpy()
        if orbit_ids:
            from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

            mean_enc = orbit_encs[0].clone()
            for key in ("visual", "proprio"):
                mean_enc[key] = torch.stack([e[key] for e in orbit_encs]).mean(0)
            agent.goal_state_enc = mean_enc
            agent.objective = ReprTargetDistMPCObjective(cfg, target_enc=mean_enc, **cfg.planner.planning_objective)
            agent.planner.set_objective(agent.objective)
        if args.cost.split(":")[0] in ("reach", "reachrf", "reachrf2"):  # reach-then-change from images (costs.ReachThenChangeImage)
            from evals.failure_attribution.costs import ReachThenChangeImage

            kind = args.cost.split(":")[0]
            agent.objective = ReachThenChangeImage(cfg, target_enc=agent.goal_state_enc, head_path=args.cost.split(":", 1)[1],
                                                   unroll=oracles["sim_latent"][1] if "sim_latent" in oracles else None,
                                                   r0=args.reach_r0_img, rf=kind != "reach",
                                                   radius_gate=0.35 if kind == "reachrf2" else 0.0, verify_replans=2 if kind == "reachrf2" else 0)
            agent.planner.set_objective(agent.objective)
        elif args.cost.startswith("tempo:"):  # TEMPO baseline: tempo:<map.pt>[:w] (costs.TempoMixObjective)
            from evals.failure_attribution.costs import TempoMixObjective

            parts = args.cost.split(":")
            agent.objective = TempoMixObjective(cfg, target_enc=agent.goal_state_enc, map_path=parts[1], w=float(parts[2]) if len(parts) > 2 else 0.5)
            agent.planner.set_objective(agent.objective)
        elif args.cost.startswith("head:"):  # method C: learned goal cost (evals/failure_attribution/mw_head_train.py)
            from evals.failure_attribution.costs import LearnedHeadMPCObjective

            agent.objective = LearnedHeadMPCObjective(cfg, target_enc=agent.goal_state_enc, head_path=args.cost[5:])
            agent.planner.set_objective(agent.objective)
        elif args.cost != "l2":  # area-invariant aggregation (evals/failure_attribution/costs.py), same alpha
            from evals.failure_attribution.costs import PatchAggMPCObjective

            agent.objective = PatchAggMPCObjective(
                cfg, target_enc=agent.goal_state_enc, cost=args.cost, alpha=cfg.planner.planning_objective.alpha
            )
            agent.planner.set_objective(agent.objective)
        goal_state = np.asarray(evaluator.state_g, dtype=np.float64)
        snap0 = capture_env(env)
        init_state = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
        out = {
            "episode": ep,
            "ep_seed": ep_seed,
            "goal_state": goal_state,
            "expert_success": bool(expert_success),
            "rand_vec": snap0["mw"]["rand_vec"],
            **goal_info,
        }
        print(f"ep {ep:03d} start ({ep + 1}/{n_episodes}) {task}: expert success {bool(expert_success)}", flush=True)
        # the world model runs first, with the agent's generators untouched, as in the released eval
        gen_state = None
        for k, name in enumerate(args.planners):
            restore_env(env, snap0)
            td = init_obs
            replans, chunk_success, cem_final, hand, obj = 0, [], [], [], []
            if name != "wm":
                planner, unroll = oracles[name]
                objective = agent.objective if name.startswith("sim_latent") else TrueCostObjective(adapter, goal_state, device)
                if name.endswith("_reach"):
                    from evals.failure_attribution.costs import ReachThenChangeOracle

                    objective = ReachThenChangeOracle(objective, unroll, init_state[4:7], goal_state[4:7], eps=args.reach_eps, r0=args.reach_r0)
                    out[f"{name}_active"] = objective.active
                planner.set_objective(objective)
                planner.local_generator.manual_seed(int(ep_seed) + 1000 * (k + 1))
            while True:
                steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
                progress[name].start(f"ep {ep:03d} [{name}] replan {replans}")
                if name == "wm":
                    a = agent.act(prepare_obs(obs_type, td), steps_left=steps_left)
                    cem_final.append(float(agent._prev_losses.squeeze(-1)[-1]))
                else:
                    z = model.encode(prepare_obs(obs_type, td).to(device, non_blocking=True).unsqueeze(0), act=True)
                    unroll.spec = (task, capture(env.unwrapped), args.robot_free)
                    if replans > 0 and hasattr(planner.objective, "update"):  # v2: verify the detected target in closed loop
                        planner.objective.update(z, np.asarray(env.unwrapped._get_obs(), dtype=np.float64)[:3])
                    if replans == 0 and hasattr(planner.objective, "set_start"):  # change map from the episode's start and goal
                        import mujoco

                        mw0 = env.unwrapped
                        cid = mujoco.mj_name2id(mw0.model, mujoco.mjtObj.mjOBJ_CAMERA, "corner2")
                        planner.objective.set_camera(mw0.data.cam_xpos[cid], mw0.data.cam_xmat[cid], float(mw0.model.cam_fovy[cid]),
                                                     int(cfg.task_specification.img_size))
                        hand0 = torch.from_numpy(np.asarray(mw0._get_obs(), dtype=np.float32)[:3])
                        gh = np.asarray(goal_obs["proprio"], dtype=np.float64).reshape(-1, 4)[-1, :3]
                        planner.objective.set_start(z, hand0, goal_hand_z=float(gh[2]), goal_hand=gh)
                        out[f"{name}_reach_info"] = json.dumps(planner.objective.info)
                        print(f"  [{name}] reach-then-change: {planner.objective.info}", flush=True)
                    res = planner.plan(z, steps_left=steps_left)
                    a = res.actions
                    cem_final.append(float(res.losses.squeeze(-1)[-1]))
                progress[name].stop()
                obses, _, dones, infos = env.step_multiple(denorm(a))
                td = make_td(obses[-1], infos[-1])
                chunk_success.append(bool(infos[-1]["success"]))
                st = np.asarray(infos[-1]["state"], dtype=np.float64)
                hand.append(st[:3]); obj.append(st[4:7])
                replans += 1
                if dones[-1]:
                    break
            if name == "wm":
                gen_state = (agent.local_generator.get_state(), agent.local_gpu_generator.get_state())
            out.update(
                {
                    f"{name}_success": chunk_success[-1],
                    f"{name}_success_any": any(chunk_success),
                    f"{name}_chunk_success": np.array(chunk_success),
                    f"{name}_cem_final_cost": np.array(cem_final),
                    f"{name}_hand": np.array(hand),
                    f"{name}_obj": np.array(obj),
                    f"{name}_final_state": st,
                    f"{name}_obj_to_target": float(infos[-1].get("obj_to_target", np.nan)),
                    **({f"{name}_reach_info_final": json.dumps(planner.objective.info)} if name != "wm" and hasattr(planner.objective, "info") else {}),
                }
            )
        if gen_state is None:  # world model not run: the agent's generators were not used
            gen_state = (agent.local_generator.get_state(), agent.local_gpu_generator.get_state())
        out["cpu_generator_state"] = gen_state[0].numpy()
        out["gpu_generator_state"] = gen_state[1].cpu().numpy()
        save_episode(os.path.join(args.out_dir, f"ep_{ep:03d}.npz"), **out)

        times.append(time.time() - t0)
        msg = " ".join(f"{n}={'S' if out[f'{n}_success'] else 'F'}" for n in args.planners)
        n_succ = {n: 0 for n in args.planners}
        for f in os.listdir(args.out_dir):
            if f.startswith("ep_") and f.endswith(".npz"):
                d = np.load(os.path.join(args.out_dir, f))
                for n in args.planners:
                    n_succ[n] += bool(d[f"{n}_success"])
        tally = " ".join(f"{n} {v}/{ep + 1}" for n, v in n_succ.items())
        left = (n_episodes - ep - 1) * float(np.mean(times))
        print(f"ep {ep:03d} done: {msg} | so far {tally} | {times[-1]:.0f}s, about {left / 60:.0f} min left", flush=True)

    pool.close()
    save_json(os.path.join(args.out_dir, 'run_meta.json'), {"fname": os.path.abspath(args.fname), "task": task, "episodes": n_episodes,
                   "planners": args.planners, "alpha": float(cfg.planner.planning_objective.alpha), "cost": args.cost,
                   "goal_variant": args.goal_variant, "robot_free": args.robot_free, "reach_eps": args.reach_eps, "reach_r0": args.reach_r0,
                   "wm_set": args.wm_set, "wm_h": args.wm_h, "wm_mag": args.wm_mag, "wm_gate": args.wm_gate})
    print("done", flush=True)


if __name__ == "__main__":
    main()
