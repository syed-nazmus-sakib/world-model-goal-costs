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
from evals.failure_attribution.costs import HeadChangeObjective
from evals.failure_attribution.mw_sim import capture, capture_env, restore_env
from evals.failure_attribution.run import CEMProgress, SimUnroll, TrueCostObjective, build, load_args
from evals.failure_attribution.sim_worker import SimPool



def frame_of(td):
    """Last RGB frame of an observation TensorDict as (H, W, 3) uint8."""
    return np.asarray(td["visual"])[-1].transpose(1, 2, 0).copy()

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
    parser.add_argument("--ep_start", type=int, default=0, help="first episode index (training episodes start at 100 in the pilot data)")
    parser.add_argument("--dump_dir", default="",
                        help="save the sim_latent planner's visited frames (encoding + 41-d state, kind 5) per episode: training data "
                             "for the next head (mw_merge_planner_frames.py merges them with the expert frames of the pilot data)")
    parser.add_argument("--dump_cands", type=int, default=32,
                        help="with --dump_dir: also save this many of the planner's final-population candidates per replan (kind 6)")
    parser.add_argument("--save_frames", action="store_true", help="store the start, goal and per-replan RGB frames in the episode file (figures)")
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
    start = max(start, args.ep_start)
    n_episodes += args.ep_start
    if args.dump_dir:
        from evals.failure_attribution.mw_pilot_data import flat, full_state

        os.makedirs(args.dump_dir, exist_ok=True)

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
        if args.cost.split(":")[0] in ("reach", "reachrf", "reachrf2", "reachrf3"):  # reach-then-change from images (costs.ReachThenChangeImage)
            from evals.failure_attribution.costs import ReachThenChangeImage

            kind = args.cost.split(":")[0]
            v23 = kind in ("reachrf2", "reachrf3")
            agent.objective = ReachThenChangeImage(cfg, target_enc=agent.goal_state_enc, head_path=args.cost.split(":", 1)[1],
                                                   unroll=oracles["sim_latent"][1] if "sim_latent" in oracles else None,
                                                   r0=args.reach_r0_img, rf=kind != "reach",
                                                   radius_gate=0.35 if v23 else 0.0, verify_replans=2 if v23 else 0, v3b=kind == "reachrf3")
            agent.planner.set_objective(agent.objective)
        elif args.cost.startswith("tempo:"):  # TEMPO baseline: tempo:<map.pt>[:w] (costs.TempoMixObjective)
            from evals.failure_attribution.costs import TempoMixObjective

            parts = args.cost.split(":")
            agent.objective = TempoMixObjective(cfg, target_enc=agent.goal_state_enc, map_path=parts[1], w=float(parts[2]) if len(parts) > 2 else 0.5)
            agent.planner.set_objective(agent.objective)
        elif args.cost.startswith("headrf:"):  # head + robot-free change term: headrf:<head.pt>:<robot.pt>[:beta[:lam]] (costs.HeadChangeObjective)
            # (imported at module level: a local import here made the isinstance check below fail with UnboundLocalError for every other --cost)

            parts = args.cost.split(":")
            agent.objective = HeadChangeObjective(cfg, target_enc=agent.goal_state_enc, head_path=parts[1], robot_path=parts[2],
                                                  beta=float(parts[3]) if len(parts) > 3 else 1.0, lam=float(parts[4]) if len(parts) > 4 else 0.5)
            agent.planner.set_objective(agent.objective)
        elif args.cost.startswith("head:"):  # learned goal cost head:<pt>[:beta] (mw_head_train.py / mw_pilot_heads.py)
            from evals.failure_attribution.costs import LearnedHeadMPCObjective

            parts = args.cost.split(":")
            agent.objective = LearnedHeadMPCObjective(cfg, target_enc=agent.goal_state_enc, head_path=parts[1],
                                                      beta=float(parts[2]) if len(parts) > 2 else 0.0)
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
        if args.save_frames:
            out["start_frame"], out["goal_frame"] = frame_of(init_obs), frame_of(goal_obs)
        print(f"ep {ep:03d} start ({ep + 1}/{n_episodes}) {task}: expert success {bool(expert_success)}", flush=True)
        # the world model runs first, with the agent's generators untouched, as in the released eval
        gen_state = None
        for k, name in enumerate(args.planners):
            restore_env(env, snap0)
            td = init_obs
            replans, chunk_success, cem_final, hand, obj, frames = 0, [], [], [], [], []
            if name != "wm":
                planner, unroll = oracles[name]
                objective = agent.objective if name.startswith("sim_latent") else TrueCostObjective(adapter, goal_state, device)
                if name.endswith("_reach"):
                    from evals.failure_attribution.costs import ReachThenChangeOracle

                    objective = ReachThenChangeOracle(objective, unroll, init_state[4:7], goal_state[4:7], eps=args.reach_eps, r0=args.reach_r0)
                    out[f"{name}_active"] = objective.active
                planner.set_objective(objective)
                planner.local_generator.manual_seed(int(ep_seed) + 1000 * (k + 1))
            dump, dump_steps = {"vis": [], "prop": [], "state": [], "t": [], "kind": []}, 0
            dump_rng = np.random.default_rng(ep)
            while True:
                steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
                if args.save_frames:
                    frames.append(frame_of(td))
                progress[name].start(f"ep {ep:03d} [{name}] replan {replans}")
                if name == "wm":
                    if replans == 0 and isinstance(agent.objective, HeadChangeObjective):  # the change term needs the start frame
                        agent.objective.set_start(model.encode(prepare_obs(obs_type, td).to(device, non_blocking=True).unsqueeze(0), act=True))
                    a = agent.act(prepare_obs(obs_type, td), steps_left=steps_left)
                    cem_final.append(float(agent._prev_losses.squeeze(-1)[-1]))
                    if args.dump_dir and args.dump_cands > 0 and getattr(agent.objective, "last_pop", None) is not None:
                        # kind 7: the PREDICTED final latents of the last CEM population (what the cost prices when it plans with the
                        # learned predictor), each labelled by executing its action sequence in the simulator from the current state
                        enc_p, acts = agent.objective.last_pop
                        agent.objective.last_pop = None
                        su = sim_unrolls["latent"]
                        su.spec = (task, capture(env.unwrapped), args.robot_free)
                        B_p = int(enc_p["visual"].shape[1])
                        sel = np.sort(dump_rng.choice(B_p, min(args.dump_cands, B_p), replace=False))
                        ls, _ = pool.rollout(su.spec, su.env_actions(acts[:, sel]), False)
                        from tensordict import TensorDict as _TD

                        fv, fp = flat(_TD({"visual": enc_p["visual"][-1:, sel], "proprio": enc_p["proprio"][-1:, sel]}, batch_size=[]))
                        t_c = dump_steps + len(denorm(a))
                        for j in range(len(sel)):
                            dump["vis"].append(fv[j])
                            dump["prop"].append(fp[j])
                            s_j = np.asarray(ls[j], dtype=np.float64)
                            dump["state"].append(s_j.astype(np.float32) if s_j.shape[-1] == 41 else full_state(env.unwrapped, s_j[:39]))
                            dump["t"].append(t_c)
                            dump["kind"].append(7)
                else:
                    z = model.encode(prepare_obs(obs_type, td).to(device, non_blocking=True).unsqueeze(0), act=True)
                    if args.dump_dir and name == "sim_latent":  # the planner's own visited frame (last of the history), with its labels
                        fv, fp = flat(z)
                        dump["vis"].append(fv[-1])
                        dump["prop"].append(fp[-1])
                        dump["state"].append(full_state(env.unwrapped, np.asarray(env.unwrapped._get_obs(), dtype=np.float64)))
                        dump["t"].append(dump_steps)
                        dump["kind"].append(5)
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
                        gp = np.asarray(goal_obs["proprio"], dtype=np.float64).reshape(-1, 4)[-1]
                        gh = gp[:3]
                        planner.objective.set_start(z, hand0, goal_hand_z=float(gh[2]), goal_hand=gh, goal_prop=gp)
                        out[f"{name}_reach_info"] = json.dumps(planner.objective.info)
                        print(f"  [{name}] reach-then-change: {planner.objective.info}", flush=True)
                    unroll.last_pop = None
                    res = planner.plan(z, steps_left=steps_left)
                    a = res.actions
                    if args.dump_dir and name == "sim_latent" and args.dump_cands > 0 and getattr(unroll, "last_pop", None) is not None:
                        # a sample of the last population the planner compared: the states it prices, right or wrong (kind 6)
                        ls, enc = unroll.last_pop
                        ls = np.asarray(ls)
                        fv, fp = flat(enc)  # (B, P, D), (B, Dp): one frame per candidate
                        t_c = dump_steps + len(denorm(a))
                        for i in np.sort(dump_rng.choice(min(len(ls), len(fv)), min(args.dump_cands, len(ls), len(fv)), replace=False)):
                            dump["vis"].append(fv[i])
                            dump["prop"].append(fp[i])
                            s_i = np.asarray(ls[i], dtype=np.float64)
                            dump["state"].append(s_i.astype(np.float32) if s_i.shape[-1] == 41 else full_state(env.unwrapped, s_i[:39]))
                            dump["t"].append(t_c)
                            dump["kind"].append(6)
                    cem_final.append(float(res.losses.squeeze(-1)[-1]))
                progress[name].stop()
                obses, _, dones, infos = env.step_multiple(denorm(a))
                dump_steps += len(obses)
                td = make_td(obses[-1], infos[-1])
                chunk_success.append(bool(infos[-1]["success"]))
                st = np.asarray(infos[-1]["state"], dtype=np.float64)
                hand.append(st[:3]); obj.append(st[4:7])
                replans += 1
                if dones[-1]:
                    break
            if args.save_frames:
                frames.append(frame_of(td))
                out[f"{name}_frames"] = np.stack(frames)
            if args.dump_dir and name in ("sim_latent", "wm") and dump["vis"]:  # the final frame too (the exploited end state), then save
                if name == "sim_latent":  # (the wm dump holds predicted latents only, kind 7)
                    zf = model.encode(prepare_obs(obs_type, td).to(device, non_blocking=True).unsqueeze(0), act=True)
                    fv, fp = flat(zf)
                    dump["vis"].append(fv[-1])
                    dump["prop"].append(fp[-1])
                    dump["state"].append(full_state(env.unwrapped, st))
                    dump["t"].append(dump_steps)
                    dump["kind"].append(5)
                shapes = {x.shape for x in dump["vis"]} | {x.shape for x in dump["prop"]}
                if len({x.shape for x in dump["vis"]}) == 1 and len({x.shape for x in dump["prop"]}) == 1:
                    n_d = len(dump["vis"])
                    np.savez(os.path.join(args.dump_dir, f"ep_{ep:03d}.npz"), vis=np.stack(dump["vis"]), prop=np.stack(dump["prop"]),
                             state=np.stack(dump["state"]).astype(np.float32), kind=np.array(dump["kind"]), t=np.array(dump["t"]),
                             var=np.full(n_d, -1), start=np.full(n_d, -1), episode=np.array(ep), success=np.array(chunk_success[-1]))
                    print(f"  [dump] ep {ep:03d}: {n_d} frames ({dump['kind'].count(5)} executed, {dump['kind'].count(6)} candidates)", flush=True)
                else:
                    print(f"  [dump] ep {ep:03d}: NOT saved, frame shapes differ: {shapes}", flush=True)
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
