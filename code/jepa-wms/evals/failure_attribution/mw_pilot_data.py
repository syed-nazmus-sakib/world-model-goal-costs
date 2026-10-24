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
  walks   (--walks K, 2026-10-24) arm-coverage frames: from the expert start and 4 steps before contact, K random action
          sequences of --walk_chunks planner chunks (half the CEM prior, half with a constant random drift so that the
          arm travels), a frame every --walk_every chunks. The rtc head trained without them was exploited in closed
          loop (button 0/12): the planner visits arm poses that expert-path candidates never cover.
Every frame keeps its 41-d state [39-d obs, success, obj_to_target]. The heads decide which labels they may use.

  jitter  (--jitter K, 2026-10-24) near-goal coverage: K short random action sequences (1-2 chunks) from the expert's own states
          between 2 steps before contact and the end of the path (kind 4): gripper, height and object state vary at the goal pose.
Saves ep_XXX.npz: vis (N, P, D) fp16, prop (N, Dp) fp16, state (N, 41), kind (N,) 0 expert / 1 goal / 2 candidate / 3 walk / 4 jitter,
t (N,) expert step (start step for candidates), var (N,) goal variant index (-1 otherwise), start (N,) start index.
With --robot_mask also rmask (N, P) uint8: 1 where the 14x14 patch shows the robot (the frame differs from the same state
rendered with the robot hidden), 0 otherwise, 255 unknown (candidates). Training labels for the robot map (mw_robot_map.py).

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
    parser.add_argument("--robot_mask", action="store_true", help="also store the robot patch mask of expert and goal frames")
    parser.add_argument("--no_cands", action="store_true", help="skip the candidate futures (robot-map training data only)")
    parser.add_argument("--walks", type=int, default=0, help="arm-coverage random walks per start point (0: none)")
    parser.add_argument("--walk_chunks", type=int, default=14, help="walk length in planner chunks (14 x frameskip 5 = 70 steps)")
    parser.add_argument("--walk_every", type=int, default=2, help="a walk frame every this many chunks")
    parser.add_argument("--jitter", type=int, default=0,
                        help="near-goal coverage (kind 4): random action sequences per start point along the expert path after contact (0: none)")
    parser.add_argument("--jitter_chunks", type=int, default=14, help="longest aimed sequence in planner chunks (shorter when the goal is near)")
    parser.add_argument("--jitter_starts", type=int, default=8,
                        help="start points: the episode start, 24/12/6/2 steps before contact, the rest spread over the post-contact path")
    parser.add_argument("--jitter_hold", type=int, default=2, help="chunks of gripper command with the hand hovering before the drift")
    parser.add_argument("--jitter_drift", type=float, default=0.6, help="constant per-sequence drift (normalized units) on the xyz entries")
    parser.add_argument("--jitter_grip", type=float, default=1.5, help="constant per-sequence open-or-close command on the gripper entries")
    args = parser.parse_args()
    if args.robot_mask:
        from evals.failure_attribution.cost_share_mw import patch_mask
        from evals.failure_attribution.goal_variants import robot_free_frame
        from evals.failure_attribution.mw_sim import render_hidden, robot_geoms
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
        snaps, ex_frames, ex_states, ex_t, contact, masks = [evaluator.expert_init_snap], [], [], [], None, []
        geoms = robot_geoms(mw.model) if args.robot_mask else None
        for k, a in enumerate(acts):
            obs, _, _, _, info = w.step(a)
            snaps.append(capture(mw))
            if contact is None and np.abs(object_qpos(mw) - q0).max() > 2e-3:
                contact = k
            if (k + 1) % args.stride == 0 or k == len(acts) - 1:
                fr = w.render().copy()
                ex_frames.append(fr)
                if args.robot_mask:  # same state, robot hidden: the patches that differ are the robot
                    masks.append(patch_mask(fr, render_hidden(w, geoms)).reshape(-1).astype(np.uint8))
                ex_states.append(np.r_[obs, float(info["success"]), float(info.get("obj_to_target", np.nan))].astype(np.float32))
                ex_t.append(k + 1)
        replay_err = float(np.abs(ex_states[-1][:39] - goal_states[0][:39]).max())
        goal_masks = []
        if args.robot_mask:
            for snap, f, _, _ in var:
                goal_masks.append(patch_mask(f, robot_free_frame(evaluator.expert_env, snap)).reshape(-1).astype(np.uint8))
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
            if args.no_cands:
                starts = []
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

            # arm-coverage walks (kind 3): the planner's own sampling distribution, continued for a whole episode
            if args.walks > 0:
                walk_starts = [0] if contact is None else sorted({0, max(0, contact - 4)})
                Kw, Hw, d = args.walks, args.walk_chunks, model.action_dim
                for si, s in enumerate(walk_starts):
                    plain = torch.randn(Hw, Kw // 2, d, device=device)
                    drift = torch.randn(1, Kw - Kw // 2, d, device=device)
                    seq = torch.cat([plain, 0.4 * torch.randn(Hw, Kw - Kw // 2, d, device=device) + drift], 1)
                    for n in range(args.walk_every, Hw + 1, args.walk_every):
                        states, frames = pool.rollout((task, snaps[s], False), unroll.env_actions(seq[:n]), True)
                        add(flat(unroll.encode_final(states, frames)), states, 3, s + n * fs, -1, si)

            # near-goal coverage (kind 4, 2026-10-24): short random action sequences of the planner's prior from the expert's own
            # states between 2 steps before contact and the end of the path. The hand stays near the goal pose while the gripper,
            # the height and the object state vary: the terminal configurations the planner exploited with the walks-only head
            # (button: hand at the goal pose, gripper open around the button, button unpressed, priced as success)
            # Per-step N(0, 1) actions from post-contact states left the gripper where the expert had it (button: closed, pressed
            # in 83% of frames, never the open gripper around an unpressed button), so each sequence carries a constant drift: small
            # in xyz (jitter_drift), and on the gripper entries a fixed open-or-close command (±jitter_grip), from 3 pre-contact
            # starts (12, 8, 4 steps before contact) and the rest spread over the post-contact path.
            # Two phases per sequence (the one-phase drift still pressed the button whenever the hand reached the goal height: the
            # gripper had not opened by then): `jitter_hold` chunks of gripper command with the hand hovering, then `jitter_chunks`
            # chunks of xyz drift with the same gripper command, so an open gripper can descend around the button.
            # Aimed sequences (2026-10-24, three designs discarded before training): from the episode start and from states along
            # the expert path, a constant env-space velocity straight to the goal hand position (plus N(0, 1 cm) lateral offset),
            # i.e. the planner's own straight-line path, with each gripper command held constant (open or close). Random drifts
            # and top-down descents from the expert's pre-contact states never produced the exploited configuration (button:
            # the planner arrives at the goal height *laterally*, so the unpressed plunger sits between the open fingers; a
            # descent from above presses it even with the fingers open). Frames: one mid-way and the last four chunks (arrival,
            # one chunk beyond it), so the near-goal configurations get their labels.
            if args.jitter > 0:
                T_end = len(snaps) - 1
                if contact is None:
                    j_starts = sorted({int(round(x)) for x in np.linspace(0, T_end, args.jitter_starts)})
                else:
                    pre = [0] + [max(0, contact - k) for k in (24, 12, 6, 2)]
                    post = [int(round(x)) for x in np.linspace(contact, T_end, max(1, args.jitter_starts - 5))]
                    j_starts = sorted(set(pre + post))
                Kj, Ha, d = args.jitter, args.jitter_hold, model.action_dim
                gh_env = np.asarray(goal_states[0][:3], dtype=np.float64)  # the expert's final hand position (goal proprio)
                rng_j = np.random.default_rng(1000 + ep)
                for si, s in enumerate(j_starts):
                    hand_s = ex_states[int(np.argmin(np.abs(np.asarray(ex_t) - s))), :3].astype(np.float64)
                    offs = rng_j.normal(0.0, 0.01, (Kj, 3))  # 1 cm lateral spread; a third of the sequences exactly centred
                    offs[:, 2] = 0.0
                    offs[: max(1, Kj // 3)] = 0.0
                    dist = float(np.linalg.norm(gh_env - hand_s))
                    # the mocap arm realises about half of a 1 cm/step command on long moves (0.37-0.5 cm/step measured), so the
                    # sequence is planned at 0.5 cm/step and the command is scaled to full speed on its largest component
                    Hj = int(np.clip(np.ceil(dist / (fs * 0.005)), 2, args.jitter_chunks))
                    delta = gh_env[None] + offs - hand_s[None]
                    vel = delta / np.maximum(np.abs(delta).max(1, keepdims=True), 1e-6)  # env units, largest component = 1 cm per step
                    vel = vel * np.minimum(1.0, np.abs(delta).max(1, keepdims=True) / (Hj * fs * 0.005))
                    grip_env = rng_j.integers(0, 2, Kj) * 2.0 - 1.0  # open (-1) or close (+1), constant per sequence
                    Hd = Hj + 1  # one chunk beyond the arrival: keeps pushing toward the target
                    env_seq = np.zeros((Ha * fs + Hd * fs, Kj, 4), np.float32)
                    env_seq[..., 3] = grip_env[None]
                    env_seq[Ha * fs :, :, :3] = vel[None]
                    norm = agent.preprocessor.normalize_actions(torch.from_numpy(env_seq.reshape(-1, 4))).reshape(Ha + Hd, fs, Kj, 4)
                    seq = norm.permute(0, 2, 1, 3).reshape(Ha + Hd, Kj, d).to(device).float() + 0.3 * torch.randn(Ha + Hd, Kj, d, device=device)
                    frames_at = sorted({Ha + Hd // 2} | {Ha + k for k in range(max(1, Hd - 3), Hd + 1)})
                    for n in frames_at:
                        states, frames = pool.rollout((task, snaps[s], False), unroll.env_actions(seq[:n]), True)
                        add(flat(unroll.encode_final(states, frames)), states, 4, s + n * fs, -1, si)

        extra = {}
        if args.robot_mask:  # expert frames, then goals, then candidates (unknown), in the order of V
            n_total = sum(len(v) for v in V)
            rm = np.full((n_total, V[0].shape[1]), 255, dtype=np.uint8)
            known = np.concatenate([np.stack(masks), np.stack(goal_masks)])
            rm[: len(known)] = known
            extra["rmask"] = rm
        np.savez(path, vis=np.concatenate(V), prop=np.concatenate(P), state=np.concatenate(S), kind=np.concatenate(K),
                 t=np.concatenate(T), var=np.concatenate(VAR), start=np.concatenate(ST), starts=np.array(starts),
                 contact=np.array(-1 if contact is None else contact), replay_err=np.array(replay_err),
                 goal_path_diff=np.array(goal_path_diff), **extra)
        st = np.concatenate(S)
        kd = np.concatenate(K)
        print(f"ep {ep:03d}: {len(st)} frames, {len(var)} goals, contact {contact}, starts {starts}, "
              f"cand success {np.nanmean(st[kd == 2, 39]):.2f}, goal success {[float(g[39]) for g in goal_states]}, "
              f"replay err {replay_err:.1e}, goal path diff {goal_path_diff:.1e}, {time.time() - t0:.0f}s", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
