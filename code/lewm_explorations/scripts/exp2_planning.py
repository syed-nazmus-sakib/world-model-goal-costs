"""Kill test 2: closed-loop PushT planning with a swapped terminal value,
compute matched in world-model transitions per CEM iteration.

Protocol mirrors scripts/plan/config/pusht.yaml: start = dataset state, goal =
state 25 env steps later, budget 50 env steps, CEM H=5 blocks (x5 env steps),
receding horizon 5 blocks (open-loop 25 steps), 30 iters, top-k = samples/10.

For a method with `extra` transitions per candidate, samples are set to
  n = budget_per_iter // (H + extra)
so every method spends the same world-model compute per CEM iteration.

Usage:
  python scripts/exp2_planning.py --methods l2 probemin:K1:M8 cf_ridge:K1:M8:rho0.1 \
      --budgets 1500 --episodes 50 --seeds 0 1 2
"""
import argparse, json, os, time

import numpy as np
import torch
import h5py

from ctrlgeo import sim
from ctrlgeo.wm import FrozenLeWM, H5
from ctrlgeo.planner import TerminalValue, cem_plan


def parse_method(s):
    parts = s.split(':')
    kw = dict(kind=parts[0])
    for p in parts[1:]:
        if p.startswith('rho'): kw['rho'] = float(p[3:])
        elif p.startswith('beta'): kw['beta'] = float(p[4:])
        elif p.startswith('H'): kw['H'] = int(p[1:])
        elif p.startswith('R'): kw['R'] = int(p[1:])
        elif p[0] in 'KM': kw[p[0]] = int(p[1:])
    return kw


def sample_tasks(n, seed, offset=25):
    """Same sampler as eval_wm.py: uniform over rows with step_idx <= len-offset-1."""
    with h5py.File(H5, 'r') as f:
        ep_len = f['ep_len'][:]; off = f['ep_offset'][:]
        ep = f['episode_idx'][:]; st = f['step_idx'][:]
        valid = np.where(st <= ep_len[ep] - offset - 1)[0]
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(valid, n, replace=False))
        starts = f['state'][rows]; goals = f['state'][rows + offset]
    return starts, goals


def run(wm, method, budget, n_ep, seed, iters=30, eval_budget=50, H_default=5, offset=25):
    kw = parse_method(method)
    H = kw.pop('H', H_default)
    R = kw.pop('R', 0)
    value = TerminalValue(wm, seed=seed, **kw)
    if R:  # all samples pay H; only R of them pay the extra terminal cost
        n_samples = (budget - R * value.extra) // H
        if n_samples < 2 * R:
            raise ValueError(f'{method}: budget {budget} too small for R={R} (need >= {R * (value.extra + 2 * H)})')
        topk = max(2, min(R // 2, n_samples // 10))
        used = n_samples * H + R * value.extra
    else:
        n_samples = max(8, budget // (H + value.extra))
        topk = max(2, n_samples // 10)
        used = n_samples * (H + value.extra)
    starts, goals = sample_tasks(n_ep, seed, offset)
    envs = [sim.make_env(resolution=wm.res) for _ in range(n_ep)]
    goal_imgs = []
    for e, s, g in zip(envs, starts, goals):
        e._set_state(g); goal_imgs.append(e.render())
        e._set_state(s); e._set_goal_state(g)
    zg = wm.encode(np.stack(goal_imgs))
    done = np.zeros(n_ep, bool)
    gen = torch.Generator(device=wm.dev).manual_seed(seed)
    t0 = time.time(); steps = 0
    while steps < eval_budget:
        z0 = wm.encode(np.stack([e.render() for e in envs]))
        mu, _ = cem_plan(wm, z0, zg, value, H=H, n_samples=n_samples,
                         n_iters=iters, topk=topk, gen=gen, rerank=R)
        raw = wm.denorm_actions(mu[:, :5].cpu().numpy())  # receding 5 blocks = 25 steps
        for t in range(raw.shape[1]):
            if steps >= eval_budget: break
            for i, e in enumerate(envs):
                sim.step_raw(e, raw[i, t])
                ok, _ = e.eval_state(goals[i], sim.snap_to_state7(sim.snapshot(e)))
                done[i] |= ok
            steps += 1
    final = np.array([sim.task_dist(sim.snap_to_state7(sim.snapshot(e)), g)
                      for e, g in zip(envs, goals)])
    return dict(method=method, budget=budget, seed=seed, offset=offset, eval_budget=eval_budget, n_samples=n_samples, H=H, R=R, used=used,
                extra=value.extra, success=float(done.mean() * 100),
                final_dist=float(final.mean()), time=time.time() - t0,
                ep_success=done.astype(int).tolist(), ep_final=final.round(4).tolist())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=['l2'])
    ap.add_argument('--budgets', type=int, nargs='+', default=[1500],
                    help='WM transitions per CEM iteration per env (L2@300 samples = 1500)')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--offset', type=int, default=25, help='goal = dataset state this many env steps later')
    ap.add_argument('--eval_budget', type=int, default=50)
    ap.add_argument('--model', default=None, help='folder from ctrlgeo.train (default: quentinll/lewm-pusht)')
    ap.add_argument('--out', default='results/exp2/results.jsonl')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wm = FrozenLeWM(path=args.model)
    for budget in args.budgets:
        for m in args.methods:
            for s in args.seeds:
                try:
                    r = run(wm, m, budget, args.episodes, s, iters=args.iters,
                            eval_budget=args.eval_budget, offset=args.offset)
                except ValueError as ex:
                    print('SKIP', ex, flush=True); break
                r['model'] = args.model or 'quentinll/lewm-pusht'
                print(json.dumps({k: v for k, v in r.items() if not k.startswith('ep_')}), flush=True)
                with open(args.out, 'a') as f:
                    f.write(json.dumps(r) + '\n')
