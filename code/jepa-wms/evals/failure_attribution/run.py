# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Where do the failures of latent planning come from: dynamics, cost or search?

Each episode of a released planning eval with a single open-loop plan (Push-T, Wall: same seeds, same
init / goal, same CEM) is run as usual, and then re-planned with parts of the system replaced by an oracle:

    wm            world-model dynamics + latent goal cost + CEM        (the released eval)
    sim_latent    simulator dynamics   + latent goal cost + CEM        (removes dynamics error)
    sim_true      simulator dynamics   + true state cost  + CEM        (removes dynamics and cost error)

We also log, for the world-model plan, the predicted vs realized latents and, when the goal comes from a
dataset segment, the cost the world model assigns to the expert actions of that segment.

Usage:
    python -m evals.failure_attribution.run --fname <eval_config.yaml> --out_dir <dir> [--episodes 96]
"""

import argparse
import json
import os
import time

import numpy as np

from evals.failure_attribution.io import save_episode, save_json
import torch
import yaml
from einops import rearrange
from omegaconf import OmegaConf
from tensordict import TensorDict

from evals.failure_attribution.adapters import get_adapter
from evals.failure_attribution.sim_worker import SimPool


def load_args(fname):
    from src.utils.yaml_utils import expand_env_vars

    with open(fname, "r") as f:
        return expand_env_vars(yaml.load(f, Loader=yaml.FullLoader))


def build(args_eval, alpha=None):
    """Same construction as evals.simu_env_planning.eval.main, for a single process."""
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.eval import init_module
    from evals.simu_env_planning.planning.common.parser import parse_cfg
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from evals.simu_env_planning.planning.utils import set_seed
    from evals.utils import make_datasets

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_kwargs = args_eval["model_kwargs"]
    cfgs_data = model_kwargs.get("data", {})
    dset, preprocessor = make_datasets(cfgs_data, model_kwargs.get("data_aug", {}), 1, 0)
    args_eval["frameskip"] = cfgs_data["custom"]["frameskip"]
    args_eval["work_dir"] = os.path.join(args_eval["folder"], "failure_attribution")
    if alpha is not None:
        args_eval["planner"]["planning_objective"]["alpha"] = alpha
    model = init_module(
        folder=args_eval.get("checkpoint_folder", args_eval["folder"]),
        checkpoint=model_kwargs.get("checkpoint"),
        module_name=model_kwargs.get("module_name"),
        model_kwargs=model_kwargs.get("pretrain_kwargs", {}),
        wrapper_kwargs=model_kwargs.get("wrapper_kwargs", {}),
        cfgs_data=cfgs_data,
        device=device,
        action_dim=dset.action_dim,
        proprio_dim=dset.proprio_dim,
        preprocessor=preprocessor,
    )
    # the part of main_distributed_episodes_eval that prepares cfg, for rank 0 of a world of size 1
    cfg = parse_cfg(OmegaConf.create(args_eval))
    set_seed(cfg.meta.seed)
    cfg.rank, cfg.world_size, cfg.device = 0, 1, "cuda:0"
    cfg.num_active_gpus, cfg.active_ranks = 1, [0]
    cfg.task_specification.goal_source = cfg.task_specification.get("goal_source", "expert")
    cfg.action_ratio = 1 if cfg.planner.repeat_actskip else cfg.frameskip // model.action_skip
    make_env(cfg)  # defines cfg.action_dim, as in the eval
    cfg.planner.distribute_planner = False
    cfg.local_seed = cfg.meta.seed
    agent = GC_Agent(cfg, model, dset=dset, preprocessor=preprocessor)
    env = make_env(cfg)
    evaluator = PlanEvaluator(cfg, agent)
    return cfg, agent, env, evaluator


class SimUnroll:
    """Drop-in replacement of EncPredWM.unroll that executes the actions in the simulator."""

    def __init__(self, agent, pool, adapter, mode, batch_size=150):
        self.agent, self.pool, self.adapter, self.mode, self.batch_size = agent, pool, adapter, mode, batch_size
        self.spec = None

    def env_actions(self, act_suffix):
        """(T, B, frameskip * action_dim) normalized -> (B, T * frameskip, action_dim) float32 env actions"""
        a = rearrange(act_suffix.detach().cpu(), "t b (f d) -> b (t f) d", d=self.adapter.action_dim)
        return self.agent.preprocessor.denormalize_actions(a).numpy()

    @torch.no_grad()
    def encode_final(self, states, frames):
        proprio = self.adapter.proprio(states)
        encs = []
        for s in range(0, len(states), self.batch_size):
            td = TensorDict(
                {
                    "visual": torch.from_numpy(frames[s : s + self.batch_size]).permute(0, 3, 1, 2).unsqueeze(1),
                    "proprio": torch.from_numpy(proprio[s : s + self.batch_size]).unsqueeze(1),
                },
                batch_size=[len(states[s : s + self.batch_size])],
            )
            encs.append(self.agent.model.encode(td.to(self.agent.device)))
        vis = torch.cat([e["visual"] for e in encs]).transpose(0, 1)  # (b, 1, ...) -> (1, b, ...)
        prop = torch.cat([e["proprio"] for e in encs]).transpose(0, 1)
        return TensorDict({"visual": vis, "proprio": prop}, batch_size=vis.shape[:2])

    def __call__(self, z_ctxt, act_suffix=None, **kwargs):
        states, frames = self.pool.rollout(self.spec, self.env_actions(act_suffix), self.mode == "latent")
        self.last_states = states  # (B, state_dim) of the candidates just rolled out, for oracle costs that need states
        if self.mode == "latent":
            return self.encode_final(states, frames)
        return torch.from_numpy(states).unsqueeze(0)  # (1, B, state_dim)


class TrueCostObjective:
    def __init__(self, adapter, goal_state, device):
        self.adapter, self.goal_state, self.device = adapter, np.asarray(goal_state, dtype=np.float64), device

    def __call__(self, states, actions, keepdims=False):
        cost = self.adapter.true_cost(states[-1].numpy().astype(np.float64), self.goal_state)
        return torch.from_numpy(cost).to(self.device, dtype=torch.float32)


class CEMProgress:
    """Prints the best cost every few CEM iterations of a planner, without changing what it computes."""

    def __init__(self, planner, every=5):
        self.planner, self.every = planner, every
        self.cost_function = planner.cost_function
        self.label, self.count, self.t0 = None, 0, 0.0
        planner.cost_function = self

    def start(self, label):
        self.label, self.count, self.t0 = label, 0, time.time()

    def stop(self):
        self.label = None

    def __call__(self, actions, z_init):
        cost = self.cost_function(actions, z_init)
        if self.label is not None:
            self.count += 1
            if self.count % self.every == 0 or self.count == self.planner.iterations:
                print(
                    f"{self.label} CEM iteration {self.count:2d}/{self.planner.iterations}"
                    f"  best cost {cost.min().item():.4f}  ({time.time() - self.t0:.0f}s)",
                    flush=True,
                )
        return cost


def last_steps(encs, n_ctx):
    """Drops the context steps of an unrolled (tau + T, B, ...) prediction."""
    return {k: encs[k][n_ctx:] for k in ("visual", "proprio")}


@torch.no_grad()
def latent_terms(agent, encs):
    """Visual and proprio terms of the planning cost of encodings laid out as (T, B, ...), at every step."""
    tgt = agent.objective.target_enc
    d_vis = (tgt["visual"] - encs["visual"]).pow(2).flatten(2).mean(2)
    d_prop = (tgt["proprio"] - encs["proprio"]).pow(2).flatten(2).mean(2)
    return d_vis, d_prop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--episodes", type=int, default=None, help="defaults to meta.eval_episodes of the config")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=None, help="overrides the proprio weight of the planning cost")
    parser.add_argument("--cost", default="l2", help="patch aggregation of the latent cost: l2 (released), topk:K, pmean:P")
    parser.add_argument("--no_oracles", action="store_true", help="only run and instrument the released eval")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    args_eval = load_args(args.fname)
    adapter = get_adapter(args_eval["task_specification"]["task"])
    # workers are spawned before CUDA is initialized
    pool = SimPool(adapter.kind, args.num_workers)

    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    from evals.simu_env_planning.planning.utils import make_td
    from evals.utils import prepare_obs

    cfg, agent, env, evaluator = build(args_eval, alpha=args.alpha)
    assert cfg.planner.planner_name == "cem"
    assert cfg.planner.num_act_stepped == cfg.planner.horizon == cfg.task_specification.goal_H, "single-plan evals only"
    model, device = agent.model, agent.device
    obs_type = cfg.task_specification.obs
    alpha = cfg.planner.planning_objective.alpha
    n_episodes = args.episodes or cfg.meta.eval_episodes

    # remember how the env was prepared, so that the oracles start from exactly the same state
    prepared = {}
    env_prepare = env.prepare

    def recording_prepare(seed, init_state, env_info=None):
        prepared.update(seed=seed, init_state=np.array(init_state), env_info=env_info)
        return env_prepare(seed, init_state, env_info=env_info)

    env.prepare = recording_prepare

    sim_unrolls = {m: SimUnroll(agent, pool, adapter, m) for m in ("latent", "state")}
    to_env = sim_unrolls["state"].env_actions
    oracles = {}
    if not args.no_oracles:
        for name, mode in (("sim_latent", "latent"), ("sim_true", "state")):
            oracles[name] = CEMPlanner(
                unroll=sim_unrolls[mode],
                action_dim=model.action_dim,
                local_generator=torch.Generator(device="cuda:0"),
                **{**cfg.planner, "decode_each_iteration": False},
            )
    progress = {"wm": CEMProgress(agent.planner)}
    progress.update({name: CEMProgress(planner) for name, planner in oracles.items()})
    n_stages = 1 + len(oracles)

    # resume: the two generators of the agent decide the episodes and the CEM samples
    start = 0
    done = sorted(f for f in os.listdir(args.out_dir) if f.startswith("ep_") and f.endswith(".npz"))
    if done:
        last = np.load(os.path.join(args.out_dir, done[-1]))
        agent.local_generator.set_state(torch.from_numpy(last["cpu_generator_state"]))
        agent.local_gpu_generator.set_state(torch.from_numpy(last["gpu_generator_state"]))
        start = int(last["episode"]) + 1
        print(f"resuming at episode {start}", flush=True)

    times = []
    for ep in range(start, n_episodes):
        t0 = time.time()
        # ---- episode setup, as in PlanEvaluator.eval
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        evaluator.expert_actions = None
        env.reset(seed=ep_seed, task_idx=-1)
        unwrapped = env.proprio_env.unwrapped
        unwrapped._freeze_rand_vec = False
        unwrapped.seeded_rand_vec = True
        env.seed(ep_seed)
        init_obs, goal_obs, expert_obses, _ = evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=-1)
        if goal_obs["visual"].shape[0] > model.tubelet_size_enc:
            goal_obs["visual"] = goal_obs["visual"][-model.tubelet_size_enc :]
        agent.set_goal(prepare_obs(obs_type, goal_obs))
        if args.cost != "l2":  # area-invariant aggregation (evals/failure_attribution/costs.py), same alpha
            from evals.failure_attribution.costs import PatchAggMPCObjective

            agent.objective = PatchAggMPCObjective(
                cfg, target_enc=agent.goal_state_enc, cost=args.cost, alpha=cfg.planner.planning_objective.alpha
            )
            agent.planner.set_objective(agent.objective)
        goal_state = np.asarray(evaluator.state_g, dtype=np.float64)
        init_state = np.asarray(prepared["init_state"], dtype=np.float64)
        spec = adapter.spec(prepared, env)
        for u in sim_unrolls.values():
            u.spec = spec
        print(f"ep {ep:03d} start ({ep + 1}/{n_episodes}): {adapter.describe(init_state, goal_state, spec)}", flush=True)

        # ---- released eval: world-model CEM, as in PlanEvaluator.unroll_agent + GC_Agent.act
        steps_left = max((env.steps_left() + 1) * model.action_skip // cfg.frameskip, 1)
        z = model.encode(prepare_obs(obs_type, init_obs).to(device, non_blocking=True).unsqueeze(0), act=True)
        progress["wm"].start(f"ep {ep:03d} [1/{n_stages} world model]")
        a_plan = agent.plan(z, steps_left=steps_left)  # (T, frameskip * action_dim), normalized
        progress["wm"].stop()
        wm_losses = agent._prev_losses.squeeze(-1).numpy()
        env_action = rearrange(a_plan.cpu(), "t (f d) -> (t f) d", d=env.action_dim)
        env_action = agent.preprocessor.denormalize_actions(env_action)
        obses, _, _, infos = env.step_multiple(env_action)
        final_state = np.asarray(infos[-1]["state"], dtype=np.float64)
        official = env.eval_state(goal_state, final_state)

        out = {
            "episode": ep,
            "ep_seed": ep_seed,
            "init_state": prepared["init_state"],
            "goal_state": goal_state,
            **adapter.spec_to_save(spec),
            "cpu_generator_state": agent.local_generator.get_state().numpy(),
            "gpu_generator_state": agent.local_gpu_generator.get_state().cpu().numpy(),
            "wm_plan": a_plan.cpu().numpy(),
            "wm_cem_losses": wm_losses,
            "wm_final_state": final_state,
            "wm_success": bool(official["success"]),
            "wm_state_dist": float(official["state_dist"]),
            "wm_true_cost": float(adapter.true_cost(final_state[None], goal_state)[0]),
        }
        out.update({f"wm_{k}": v for k, v in adapter.errors(final_state, goal_state).items()})

        with torch.no_grad():
            # the same check as the released success, on the fast simulator path
            sim_state, _ = pool.rollout(spec, to_env(a_plan.unsqueeze(1)), False)
            out["fast_rollout_max_abs_diff"] = float(np.abs(sim_state[0] - np.asarray(infos[-1]["state"])).max())

            # predicted latents of the plan at every model step vs the encoded real observations
            pred = model.unroll(z, act_suffix=a_plan.unsqueeze(1))  # (tau + T, 1, ...)
            idx = np.arange(cfg.frameskip - 1, len(obses), cfg.frameskip)
            real = torch.stack([make_td(obses[i], infos[i]) for i in idx]).to(device)  # (T, ...) with t = 1 inside
            real = model.encode(real)  # batch = model steps, (T, 1, ...)
            n_ctx = pred["visual"].shape[0] - len(idx)
            pred_T = last_steps(pred, n_ctx)
            out["pred_vs_real_visual"] = (pred_T["visual"] - real["visual"]).pow(2).flatten(1).mean(1).cpu().numpy()
            out["pred_vs_real_proprio"] = (pred_T["proprio"] - real["proprio"]).pow(2).flatten(1).mean(1).cpu().numpy()
            dv, dp = latent_terms(agent, pred_T)
            out["wm_pred_cost_visual"], out["wm_pred_cost_proprio"] = dv[:, 0].cpu().numpy(), dp[:, 0].cpu().numpy()
            dv, dp = latent_terms(agent, real)
            out["wm_real_cost_visual"], out["wm_real_cost_proprio"] = dv[:, 0].cpu().numpy(), dp[:, 0].cpu().numpy()
            out["wm_pred_cost"] = float(agent.planner.cost_function(a_plan.unsqueeze(1), z)[0])

            if adapter.has_expert:
                # the cost the world model assigns to the expert actions, which reach the goal by construction
                a_exp = rearrange(evaluator.expert_actions[0], "(t f) d -> t (f d)", f=cfg.frameskip).to(device)
                out["expert_plan"] = a_exp.cpu().numpy()
                out["expert_pred_cost"] = float(agent.planner.cost_function(a_exp.unsqueeze(1), z)[0])
                dv, dp = latent_terms(agent, last_steps(model.unroll(z, act_suffix=a_exp.unsqueeze(1)), n_ctx))
                out["expert_pred_cost_visual"] = dv[:, 0].cpu().numpy()
                out["expert_pred_cost_proprio"] = dp[:, 0].cpu().numpy()
                exp_state, _ = pool.rollout(spec, to_env(a_exp.unsqueeze(1)), False)
                out["expert_true_cost"] = float(adapter.true_cost(exp_state.astype(np.float64), goal_state)[0])

        # ---- oracle planners: same CEM, same budget, same start
        for k, (name, planner) in enumerate(oracles.items()):
            planner.set_objective(
                agent.objective if name == "sim_latent" else TrueCostObjective(adapter, goal_state, device)
            )
            planner.local_generator.manual_seed(int(ep_seed) + 1000 * (k + 1))
            progress[name].start(f"ep {ep:03d} [{k + 2}/{n_stages} {name}]")
            res = planner.plan(z, steps_left=steps_left)
            progress[name].stop()
            st, _ = pool.rollout(spec, to_env(res.actions.unsqueeze(1)), False)
            st = st[0].astype(np.float64)
            off = env.eval_state(goal_state, st)
            out.update(
                {
                    f"{name}_plan": res.actions.cpu().numpy(),
                    f"{name}_cem_losses": res.losses.squeeze(-1).numpy(),
                    f"{name}_final_state": st,
                    f"{name}_success": bool(off["success"]),
                    f"{name}_state_dist": float(off["state_dist"]),
                    f"{name}_true_cost": float(adapter.true_cost(st[None], goal_state)[0]),
                }
            )
            out.update({f"{name}_{k2}": v for k2, v in adapter.errors(st, goal_state).items()})
        save_episode(os.path.join(args.out_dir, f"ep_{ep:03d}.npz"), **out)

        msg = f"ep {ep:03d} wm={'S' if out['wm_success'] else 'F'}"
        msg += "".join(f" {n}={'S' if out[f'{n}_success'] else 'F'}" for n in oracles)
        msg += f" | wm pred cost {out['wm_pred_cost']:.4f} real {out['wm_real_cost_visual'][-1] + alpha * out['wm_real_cost_proprio'][-1]:.4f}"
        if adapter.has_expert:
            msg += f" expert pred {out['expert_pred_cost']:.4f}"
        print(msg + f" | {time.time() - t0:.0f}s", flush=True)
        times.append(time.time() - t0)
        n_succ = sum(
            bool(np.load(os.path.join(args.out_dir, f))["wm_success"])
            for f in os.listdir(args.out_dir)
            if f.startswith("ep_") and f.endswith(".npz")
        )
        left = (n_episodes - ep - 1) * float(np.mean(times))
        print(
            f"ep {ep:03d} done: {ep + 1}/{n_episodes} episodes, world-model success so far {n_succ}/{ep + 1}, "
            f"about {left / 60:.0f} min left",
            flush=True,
        )

    pool.close()
    save_json(os.path.join(args.out_dir, 'run_meta.json'), {'fname': os.path.abspath(args.fname), 'task': cfg.task_specification.task, 'kind': adapter.kind, 'episodes': n_episodes, 'alpha': float(alpha), 'cost': args.cost})
    print("done", flush=True)


if __name__ == "__main__":
    main()
