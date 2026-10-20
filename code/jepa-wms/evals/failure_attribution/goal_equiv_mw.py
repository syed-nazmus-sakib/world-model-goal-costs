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
from tensordict import TensorDict

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
    parser.add_argument("--alpha", type=float, default=None, help="overrides the proprio weight of the cost")
    parser.add_argument("--wm_orbit", type=int, default=0, help="K random free-motion action sequences used to build a world-model goal set")
    parser.add_argument("--wm_scales", type=str, default="0.5,1,2", help="std of the xyz nuisance actions (normalized units)")
    parser.add_argument("--wm_h", type=int, default=3, help="planner steps of the nuisance rollouts")
    parser.add_argument("--wm_mode", choices=["noise", "dir"], default="noise", help="iid noise vs one sustained random direction per sequence")
    parser.add_argument("--wm_gate", type=float, default=0.0, help="keep this fraction of rollouts with the smallest reverse-cycle error (0 = no gate)")
    parser.add_argument("--encoder", default="dinov2", help="visual backbone for the cost (alt_encoders.py); not dinov2 implies --no_cem")
    parser.add_argument("--explain", action="store_true", help="arm-explainability ratio of each goal given the start (goalset.explain_ratio)")
    parser.add_argument("--no_cem", action="store_true", help="skip the L3b CEM plans (offline L1-L3 only, much faster)")
    parser.add_argument("--proj", type=int, default=0, help="method candidate 1: K free-motion rollouts from each goal define a nuisance subspace "
                        "that is projected out of the cost after removing the start-to-goal direction (goalset.nuisance_projection)")
    parser.add_argument("--proj_h", type=int, default=16, help="planner steps of the --proj rollouts")
    parser.add_argument("--proj_mag", type=float, default=0.7, help="xyz magnitude of the --proj rollouts (raw action units)")
    args = parser.parse_args()
    assert not (args.proj and args.encoder != "dinov2"), "--proj needs the released predictor"
    os.makedirs(args.out_dir, exist_ok=True)
    args_eval = load_args(args.fname)
    task = args_eval["task_specification"]["task"]
    adapter = get_adapter(task)
    pool = SimPool(adapter.kind, args.num_workers)

    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner
    from evals.utils import prepare_obs

    cfg, agent, env, evaluator = build(args_eval)
    if args.encoder != "dinov2":
        from evals.failure_attribution.alt_encoders import install_alt_encoder

        assert args.no_cem and not args.wm_orbit, "alternative encoders have no predictor: offline levels only"
        install_alt_encoder(agent, args.encoder)
        print(f"visual encoder replaced by {args.encoder}", flush=True)
    hook_expert_capture(evaluator)
    model, device = agent.model, agent.device
    obs_type = cfg.task_specification.obs
    alpha = cfg.planner.planning_objective.alpha if args.alpha is None else args.alpha
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
        wm_costs, orb_cost, gate_dump = None, None, None
        if args.wm_orbit:  # goal set from the world model's own free-motion rollouts, averaged with the goal itself
            scales = [float(x) for x in args.wm_scales.split(",")]
            wm_targets, gate_dump = [], []
            gen = torch.Generator(device=device).manual_seed(1000 + ep)
            cgen = torch.Generator().manual_seed(2000 + ep)
            fs = cfg.frameskip
            K, Hn = args.wm_orbit, args.wm_h

            def norm_seq(raw):  # raw (K, Hn*fs, 4) -> normalized (Hn, K, fs*4)
                n = torch.stack([agent.preprocessor.normalize_actions(raw[k]) for k in range(raw.shape[0])])
                return n.reshape(raw.shape[0], Hn, -1).permute(1, 0, 2).to(device).float()

            with torch.no_grad():
                for sc in scales:
                    row = []
                    for vi, g in enumerate(goal_encs):
                        if args.wm_mode == "dir":  # one sustained random direction per sequence (reaches far poses)
                            d = torch.randn(K, 3, generator=cgen)
                            d = d / d.norm(dim=1, keepdim=True)
                            raw = torch.zeros(K, Hn * fs, 4)
                            raw[..., :3] = sc * d[:, None, :]
                            A = norm_seq(raw)
                        else:
                            raw = None
                            A = torch.zeros(Hn, K, model.action_dim, device=device)
                            A4 = A.view(Hn, K, -1, 4)
                            A4[..., :3] = sc * torch.randn(A4[..., :3].shape, device=device, generator=gen)
                        out = model.unroll(g, act_suffix=A)
                        fin = {key: out[key][-1].unsqueeze(1) for key in ("visual", "proprio")}  # (K, 1, ...)
                        keep = torch.arange(K)
                        err = torch.zeros(K)
                        if raw is not None:  # reverse the motion from the predicted end state and compare with the goal
                            z2 = TensorDict({"visual": fin["visual"], "proprio": fin["proprio"]}, batch_size=[K])
                            out2 = model.unroll(z2, act_suffix=norm_seq(-raw.flip(1)))
                            cyc = out2["visual"][-1]
                            err = (cyc - g["visual"][0]).pow(2).flatten(1).mean(1).cpu()
                            if args.wm_gate > 0:
                                keep = torch.argsort(err)[: max(1, int(round(args.wm_gate * K)))]
                        tgt = g.clone()
                        for key in ("visual", "proprio"):
                            tgt[key] = torch.cat([g[key], fin[key][keep.to(device)]], 0).mean(0, keepdim=True)
                        row.append(tgt)
                        if raw is not None and vi == 0:
                            gate_dump.append((sc, raw.numpy(), err.numpy()))
                    wm_targets.append(row)
        spec = (task, capture(env.unwrapped), args.robot_free)
        unroll.spec = spec
        z = model.encode(prepare_obs(obs_type, init_obs).to(device).unsqueeze(0), act=True)
        explain_arr = np.zeros((1, 2))
        proj = None
        if args.proj:
            from evals.failure_attribution.goalset import projection_bases

            proj = [projection_bases(agent, z, g, K=args.proj, H=args.proj_h, mag=args.proj_mag, seed=3000 + ep) for g in goal_encs]
        if args.explain:
            from evals.failure_attribution.goalset import explain_ratio

            explain_arr = np.array([explain_ratio(agent, z, g, K=32, H=16, mag=0.7, seed=ep) for g in goal_encs])  # (V, 2)
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
            if args.wm_orbit:
                wm_costs = np.stack([[ReprTargetDistMPCObjective(cfg, target_enc=t, alpha=alpha)(enc, cand).cpu().numpy()
                                      for t in row] for row in wm_targets])  # (S, V, M)
            mean_t = goal_encs[0].clone()
            for key in ("visual", "proprio"):
                mean_t[key] = torch.stack([g[key] for g in goal_encs]).mean(0)
            orb_cost = ReprTargetDistMPCObjective(cfg, target_enc=mean_t, alpha=alpha)(enc, cand).cpu().numpy()  # (M,)
            proj_costs, proj_rho = np.zeros((1, 1, 1)), np.zeros(1)
            if proj is not None:  # (3, V, M): [d-perp projection, full projection, rho-gated full projection]
                from evals.failure_attribution.goalset import projected_costs

                X = enc["visual"][0].flatten(1)  # (M, P) final latents of the candidates
                rows = []
                for g, (Qf, Qd, rho) in zip(goal_encs, proj):
                    gv = g["visual"].flatten()
                    full = (X - gv[None]).pow(2).sum(1) / X.shape[1]
                    pf = projected_costs(X, gv, Qf)
                    rows.append(torch.stack([projected_costs(X, gv, Qd), pf, full - (1.0 - rho) * (full - pf)]).cpu().numpy())
                proj_costs = np.stack(rows, 1)  # (3, V, M)
                proj_rho = np.array([p[2] for p in proj])
        gate_labels = None
        if args.wm_orbit and gate_dump:
            gspec = (task, var[0][0], False)
            gate_labels = []
            for sc, raw, err in gate_dump:
                st, _ = pool.rollout(gspec, raw.astype(np.float32), False)
                gate_labels.append(np.linalg.norm(st[:, 4:7] - np.asarray(var[0][3])[None], axis=1))
            gate_labels = np.stack(gate_labels)  # (S, K) displacement of the object caused by the nuisance motion
        true_cost = adapter.true_cost(states.astype(np.float64), None)
        # L3: CEM first action per goal variant, identical seed; noise floor = variant 1 with a second seed
        first, plans = [], []
        for vi, g in enumerate(goal_encs if not args.no_cem else []):
            planner.set_objective(ReprTargetDistMPCObjective(cfg, target_enc=g, alpha=alpha))
            planner.local_generator.manual_seed(int(ep_seed))
            res = planner.plan(z, steps_left=steps_left)
            plans.append(res.actions.cpu().numpy())
        if args.no_cem:
            plans, noise_plan = np.zeros((V, 1, 1)), np.zeros((1, 1))
        else:
            planner.set_objective(ReprTargetDistMPCObjective(cfg, target_enc=goal_encs[min(1, V - 1)], alpha=alpha))
            planner.local_generator.manual_seed(int(ep_seed) + 7)
            noise_plan = planner.plan(z, steps_left=steps_left).actions.cpu().numpy()
        moved = np.linalg.norm(states[:, 4:7] - np.asarray(env.unwrapped._get_obs())[4:7], axis=1)
        save_episode(path, costs=costs, true_cost=true_cost, puck_moved=moved, plans=np.array(plans), noise_plan=noise_plan,
                 hand=np.array([v[2][:3] for v in var]), obj=np.array([v[3] for v in var]),
                 goal_frames=np.array([v[1] for v in var]), has_plans=np.array(not args.no_cem),
                 wm_costs=(wm_costs if wm_costs is not None else np.zeros((1, 1, 1))), orb_cost=orb_cost, explain=explain_arr,
                 gate_err=(np.stack([g[2] for g in gate_dump]) if gate_dump else np.zeros((1, 1))),
                 gate_obj_move=(gate_labels if gate_labels is not None else np.zeros((1, 1))),
                 proj_costs=proj_costs, proj_rho=proj_rho)
        print(f"ep {ep:03d}: {V} goal variants, {time.time() - t0:.0f}s", flush=True)
    pool.close()
    os.environ["GOAL_EQUIV_ENCODER"] = args.encoder
    summarize(args.out_dir)


def gate_summary(g_err, g_move):
    """How well the reverse-cycle error separates nuisance motions that moved the object (>1 cm) from those that did not."""
    if not g_err:
        return None
    from scipy.stats import mannwhitneyu

    e, m = np.concatenate(g_err), np.concatenate(g_move)
    bad = m > 0.01
    out = {"n": int(len(e)), "frac_moving_object": float(bad.mean())}
    if bad.any() and (~bad).any():
        out["auc_cycle_error_detects_object_change"] = float(mannwhitneyu(e[bad], e[~bad]).statistic / (bad.sum() * (~bad).sum()))
    out["spearman_err_vs_object_move"] = float(spearmanr(e, m).correlation)
    return out


def _pad(rows):
    """List of per-episode rows (possibly of different lengths: the variant generator may accept fewer arm poses in some
    episodes) -> float array padded with NaN, so that nanmean works per variant index."""
    rows = [np.asarray(r, dtype=float) for r in rows]
    shape = tuple(max(r.shape[i] if r.ndim > i else 1 for r in rows) for i in range(max(r.ndim for r in rows)))
    out = np.full((len(rows),) + shape, np.nan)
    for k, r in enumerate(rows):
        out[(k,) + tuple(slice(0, s) for s in r.shape)] = r
    return out


def summarize(out_dir):
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("ep_"))
    sens, flips, rho, act, act_noise, flips_v0, top1, top10, moved = [], [], [], [], [], [], [], [], []
    acc_rows, wm_rows, orb_rows = [], [], []
    proj_rows, proj_sens, proj_rhos = [], [], []
    g_err, g_move = [], []
    for f in files:
        d = np.load(os.path.join(out_dir, f))
        c, t = d["costs"], d["true_cost"]
        V = len(c)
        for a, b in itertools.combinations(range(1, V), 2):  # variants 1..V-1 share the exact object pose
            sens.append(np.median(np.abs(c[a] - c[b]) / np.minimum(c[a], c[b])))
            i, j = np.triu_indices(c.shape[1], 1)
            flips.append(np.mean(np.sign(c[a, i] - c[a, j]) != np.sign(c[b, i] - c[b, j])))
            if d["has_plans"] if "has_plans" in d.files else True:
                act.append(np.linalg.norm(d["plans"][a][0] - d["plans"][b][0]))
            top1.append(float(np.argmin(c[a]) == np.argmin(c[b])))
            top10.append(len(set(np.argsort(c[a])[:10]) & set(np.argsort(c[b])[:10])) / 10)
        for a in range(1, V):
            i, j = np.triu_indices(c.shape[1], 1)
            flips_v0.append(np.mean(np.sign(c[0, i] - c[0, j]) != np.sign(c[a, i] - c[a, j])))
        rho.append([spearmanr(c[v], t).correlation for v in range(V)])
        moved.append(float(np.mean(t < t.max() - 1e-3)))
        if d["has_plans"] if "has_plans" in d.files else True:
            act_noise.append(np.linalg.norm(d["plans"][min(1, V - 1)][0] - d["noise_plan"][0]))
        # ordering accuracy against the true cost on clearly different pairs (true-cost gap above its median)
        i, j = np.triu_indices(c.shape[1], 1)
        gap = np.abs(t[i] - t[j])
        mm = gap > max(np.quantile(gap, 0.5), 1e-6)
        if "gate_err" in d.files and d["gate_err"].shape == d["gate_obj_move"].shape and d["gate_err"].shape[-1] > 1:
            g_err.append(d["gate_err"].ravel()); g_move.append(d["gate_obj_move"].ravel())
        if mm.sum() > 0 and "wm_costs" in d.files and d["wm_costs"].shape[-1] == c.shape[1]:
            wc = d["wm_costs"]
            wm_rows.append([[np.mean(np.sign(wc[si, v, i][mm] - wc[si, v, j][mm]) == np.sign(t[i] - t[j])[mm])
                             for v in range(V)] for si in range(wc.shape[0])])
            oc = d["orb_cost"]
            orb_rows.append(np.mean(np.sign(oc[i][mm] - oc[j][mm]) == np.sign(t[i] - t[j])[mm]))
        if mm.sum() > 0:
            acc_rows.append([np.mean(np.sign(c[v, i][mm] - c[v, j][mm]) == np.sign(t[i] - t[j])[mm]) for v in range(V)])
        if mm.sum() > 0 and "proj_costs" in d.files and d["proj_costs"].shape[-1] == c.shape[1]:
            pc = d["proj_costs"]  # (3, V, M)
            proj_rows.append([[np.mean(np.sign(pc[s, v, i][mm] - pc[s, v, j][mm]) == np.sign(t[i] - t[j])[mm]) for v in range(V)] for s in range(3)])
            proj_sens.append([np.median([np.median(np.abs(pc[s, a] - pc[s, b]) / np.maximum(np.minimum(pc[s, a], pc[s, b]), 1e-9))
                                         for a, b in itertools.combinations(range(1, V), 2)]) if V > 2 else np.nan for s in range(3)])
            proj_rhos.append(d["proj_rho"])
    rho = _pad(rho)
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
    out = {"encoder": os.environ.get("GOAL_EQUIV_ENCODER", "dinov2"),
        "spearman_latent_vs_true_by_puck_move (mean, n_episodes)": by_move,
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
        "L3_first_action_noise_floor_same_goal_other_seed": float(np.median(act_noise)) if act_noise else None,
        "L2_ordering_accuracy_wm_goalset(scale x variant)": np.nanmean(_pad(wm_rows), 0).round(3).tolist() if wm_rows else None,
        "gate": gate_summary(g_err, g_move),
        "explain_ratio_per_variant(mean, median)": ([[round(float(np.mean(e[:, 0])), 3), round(float(np.median(e[:, 0])), 3)] for e in [np.stack([np.load(os.path.join(out_dir, f))["explain"] for f in files])[:, v] for v in range(np.load(os.path.join(out_dir, files[0]))["explain"].shape[0])]]
                                                       if np.load(os.path.join(out_dir, files[0]))["explain"].any() else None),
        "L2_ordering_accuracy_mean_of_variants_goal": float(np.mean(orb_rows)) if orb_rows else None,
        "L2_ordering_accuracy_vs_true_cost_per_variant(clear pairs)": np.nanmean(_pad(acc_rows), 0).round(3).tolist() if acc_rows else None,
        "episodes_per_variant_index": np.sum(~np.isnan(_pad(acc_rows)), 0).tolist() if acc_rows else None,
        "proj_L2_ordering_accuracy(dperp,full,gated x variant)": np.nanmean(_pad(proj_rows), 0).round(3).tolist() if proj_rows else None,
        "proj_L1_median_rel_cost_change_between_equivalent_goals(dperp,full,gated)": np.nanmedian(_pad(proj_sens), 0).round(3).tolist() if proj_sens else None,
        "proj_rho_explained_change_per_variant(mean)": np.nanmean(_pad(proj_rhos), 0).round(3).tolist() if proj_rhos else None,
    }
    print(json.dumps(out, indent=1))
    save_json(os.path.join(out_dir, "summary.json"), out, indent=1)


if __name__ == "__main__":
    main()
