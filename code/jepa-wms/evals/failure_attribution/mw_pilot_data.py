# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Method D pilot (2026-10-22), step 1: simulator-labelled frozen JEPA-WM encodings for one Metaworld task.

Per episode (the eval's seeding, episodes [--ep_start, --ep_start + --episodes)):
  expert  the scripted expert's own trajectory (the eval's goal generator), replayed from its initial snapshot,
          every --stride steps: final frames encoded, states kept
  goals   the eval goal (variant 0) and arm-moved goal variants (goal_variants.make_arm_variants), encoded exactly
          as the planner encodes its goal
  cands   candidate futures (exact simulator, the planner's horizon) from up to three start points on the expert
          trajectory: step 0, and 12 and 4 steps before the object first moves (so that some candidates move it);
          half CEM-initial random sequences, half the expert's next actions + noise; final frame encoded as the
          sim_latent planner encodes candidates
Every frame keeps its 41-d state [39-d obs, success, obj_to_target]. The heads decide which labels they may use.

Saves ep_XXX.npz: vis (N, P, D) fp16, prop (N, Dp) fp16, state (N, 41), kind (N,) 0 expert / 1 goal / 2 candidate,
t (N,) expert step (start step for candidates), var (N,) goal variant index (-1 otherwise), start (N,) start index.

    python -m evals.failure_attribution.mw_pilot_data --fname <mw config> --out_dir <dir> --ep_start 100 --episodes 32
"""

import argparse
import os
import time

import numpy as np
import torch

from evals.failure_attribution.adapters import get_adapter
from evals.failure_attribution.goal_variants import find_mw_wrapper, goal_td, hook_expert_capture, make_arm_variants
from evals.failure_attribution.mw_sim import capture, object_qpos, restore
from evals.failure_attribution.run import SimUnroll, build, load_args
from evals.failure_attribution.sim_worker import SimPool


def full_state(mw, obs):
    """[39-d obs, success, obj_to_target] of the env's current state (Metaworld's own evaluate_state)."""
    try:
        _, info = mw.evaluate_state(np.asarray(obs, dtype=np.float64), np.zeros(4))
        return np.r_[obs, float(info["success"]), float(info.get("obj_to_target", np.nan))].astype(np.float32)
    except Exception:
        return np.r_[obs, np.nan, np.nan].astype(np.float32)


def flat(enc):
    """encode_final / model.encode output -> vis (B, P, D), prop (B, Dp) on the CPU in fp16"""
    vis, prop = enc["visual"], enc["proprio"]
    B = vis.shape[1] if vis.shape[0] == 1 else vis.shape[0]
    return (vis.reshape(B, -1, vis.shape[-1]).half().cpu().numpy(), prop.reshape(B, -1).half().cpu().numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ep_start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=150, help="per start point")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    args_eval = load_args(args.fname)
    task = args_eval["task_specification"]["task"]
    adapter = get_adapter(task)
    pool = SimPool(adapter.kind, args.num_workers)

    from evals.utils import prepare_obs

    cfg, agent, env, evaluator = build(args_eval)
    hook_expert_capture(evaluator)
    model, device = agent.model, agent.device
    obs_type, fs = cfg.task_specification.obs, cfg.frameskip
    unroll = SimUnroll(agent, pool, adapter, "latent")
    for ep in range(args.ep_start, args.ep_start + args.episodes):
        path = os.path.join(args.out_dir, f"ep_{ep:03d}.npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        torch.manual_seed(ep)
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        evaluator.expert_actions = None
        env.reset(seed=ep_seed, task_idx=-1)
        u = env.proprio_env.unwrapped
        u._freeze_rand_vec, u.seeded_rand_vec = False, True
        env.seed(ep_seed)
        evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=-1)
        var = make_arm_variants(evaluator.expert_env, evaluator.expert_init_snap, evaluator.expert_final_snap,
                                args.variants, seed=ep)
        w = find_mw_wrapper(evaluator.expert_env)
        mw = w.unwrapped
        goal_states = []
        for snap, _, obs, _ in var:
            restore(mw, snap)
            goal_states.append(full_state(mw, obs))

        # expert replay from the expert's initial snapshot: snapshots at every step, frames every `stride` steps
        acts = evaluator.expert_actions[0]
        acts = (acts.detach().cpu().numpy() if torch.is_tensor(acts) else np.asarray(acts)).astype(np.float32)
        restore(mw, evaluator.expert_init_snap)
        q0 = object_qpos(mw).copy()
        snaps, ex_frames, ex_states, ex_t, contact = [evaluator.expert_init_snap], [], [], [], None
        for k, a in enumerate(acts):
            obs, _, _, _, info = w.step(a)
            snaps.append(capture(mw))
            if contact is None and np.abs(object_qpos(mw) - q0).max() > 2e-3:
                contact = k
            if (k + 1) % args.stride == 0 or k == len(acts) - 1:
                ex_frames.append(w.render().copy())
                ex_states.append(np.r_[obs, float(info["success"]), float(info.get("obj_to_target", np.nan))].astype(np.float32))
                ex_t.append(k + 1)
        replay_err = float(np.abs(ex_states[-1][:39] - goal_states[0][:39]).max())
        try:  # each expert env holds its own EGL context
            evaluator.expert_env.close()
        except Exception:
            pass
        evaluator.expert_env = None

        with torch.no_grad():
            V, P, S, K, T, VAR, ST = [], [], [], [], [], [], []

            def add(vp, states, kind, t, varidx, start):
                V.append(vp[0]), P.append(vp[1]), S.append(np.asarray(states, dtype=np.float32))
                n = len(states)
                K.append(np.full(n, kind)), T.append(np.broadcast_to(t, n).copy()), VAR.append(np.broadcast_to(varidx, n).copy())
                ST.append(np.full(n, start))

            ex_states = np.stack(ex_states)
            add(flat(unroll.encode_final(ex_states, np.stack(ex_frames))), ex_states, 0, np.array(ex_t), -1, -1)
            for vi, ((_, f, o, _), gs) in enumerate(zip(var, goal_states)):  # the planner's own goal encoding path
                g = model.encode(prepare_obs(obs_type, goal_td(f, o)).unsqueeze(0).to(device), act=False)
                add(flat(g), gs[None], 1, len(acts), vi, -1)
            g_check = flat(unroll.encode_final(np.stack(goal_states[:1]), var[0][1][None]))[0].astype(np.float32)
            goal_path_diff = float(np.abs(g_check - V[1][0].astype(np.float32)).mean())

            H = cfg.planner.horizon
            starts = [0] if contact is None else sorted({0, max(0, contact - 12), max(0, contact - 4)})
            M2 = args.candidates // 2
            noise = torch.tensor(np.r_[0.0, np.linspace(0.05, 1.0, args.candidates - M2 - 1)], device=device, dtype=torch.float32)
            for si, s in enumerate(starts):
                exp = acts[s : s + H * fs]
                if len(exp) < H * fs:
                    exp = np.concatenate([exp, np.zeros((H * fs - len(exp), 4), np.float32)])
                exp = agent.preprocessor.normalize_actions(torch.from_numpy(exp)).reshape(H, 1, -1).to(device).float()
                cand = torch.cat([torch.randn(H, M2, model.action_dim, device=device),
                                  exp + noise.view(1, -1, 1) * torch.randn(H, len(noise), model.action_dim, device=device)], 1)
                states, frames = pool.rollout((task, snaps[s], False), unroll.env_actions(cand), True)
                add(flat(unroll.encode_final(states, frames)), states, 2, s, -1, si)

        np.savez(path, vis=np.concatenate(V), prop=np.concatenate(P), state=np.concatenate(S), kind=np.concatenate(K),
                 t=np.concatenate(T), var=np.concatenate(VAR), start=np.concatenate(ST), starts=np.array(starts),
                 contact=np.array(-1 if contact is None else contact), replay_err=np.array(replay_err),
                 goal_path_diff=np.array(goal_path_diff))
        st = np.concatenate(S)
        kd = np.concatenate(K)
        print(f"ep {ep:03d}: {len(st)} frames, {len(var)} goals, contact {contact}, starts {starts}, "
              f"cand success {np.nanmean(st[kd == 2, 39]):.2f}, goal success {[float(g[39]) for g in goal_states]}, "
              f"replay err {replay_err:.1e}, goal path diff {goal_path_diff:.1e}, {time.time() - t0:.0f}s", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
