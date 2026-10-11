"""Direction A, kill test K1 on OGBench cube-single. See results/exp6/PREREGISTRATION.md.

  MUJOCO_GL=egl python scripts/exp6_cube.py expert            # expert trajectories
  MUJOCO_GL=egl python scripts/exp6_cube.py data               # EXP/RND1/RND4 window datasets
  python -m ctrlgeo.train --data results/exp6/data/EXP.npz --out results/exp6/models/EXP_s0 --seed 0
  MUJOCO_GL=egl python scripts/exp6_cube.py cfbuild            # counterfactual ground truth
  MUJOCO_GL=egl python scripts/exp6_cube.py cf   --models results/exp6/models/*_s0
  MUJOCO_GL=egl python scripts/exp6_cube.py plan --models results/exp6/models/*_s0 --seeds 0 1 2
"""
import argparse, json, os, time
from multiprocessing import get_context

import numpy as np

OUT = 'results/exp6'
RES = 112
T_TRAJ, N_TRAJ, N_TEST = 500, 400, 40
WIN = 20  # env steps per training window (4 frames at stride 5)


# ------------------------------------------------------------------ workers
def _w_expert(seed):
    from ctrlgeo import cube_sim as cs
    env = cs.make_env(RES)
    return cs.expert_rollout(env, T_TRAJ, np.random.default_rng(seed))


def _w_windows(args):
    snap, seqs = args
    from ctrlgeo import cube_sim as cs
    env = _w_windows.env if hasattr(_w_windows, 'env') else cs.make_env(RES)
    _w_windows.env = env
    out = []
    for seq in seqs:
        cs.restore(env, snap)
        fr = [cs.render(env)]
        for blk in range(3):
            for a in seq[5 * blk:5 * blk + 5]:
                cs.step(env, a)
            fr.append(cs.render(env))
        out.append(np.stack(fr))
    return np.stack(out)


def _w_cf(args):
    snap, gsnap, blocks = args
    from ctrlgeo import cube_sim as cs
    env = _w_cf.env if hasattr(_w_cf, 'env') else cs.make_env(RES)
    _w_cf.env = env
    cs.restore(env, gsnap); gimg = cs.render(env); gcube = cs.cube_pos(env)
    cs.restore(env, snap); img0 = cs.render(env)
    imgs, cd = [], []
    for b in blocks:
        cs.restore(env, snap)
        for a in b:
            cs.step(env, a)
        imgs.append(cs.render(env)); cd.append(np.linalg.norm(cs.cube_pos(env) - gcube))
    return img0, gimg, np.stack(imgs), np.array(cd)


def pool():
    return get_context('spawn').Pool(30)


# ------------------------------------------------------------------ stages
def stage_expert():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with pool() as p:
        res = p.map(_w_expert, range(N_TRAJ))
    S = np.stack([r[0] for r in res]); A = np.stack([r[1] for r in res]); C = np.stack([r[2] for r in res])
    np.savez(f'{OUT}/expert.npz', snaps=S, actions=A, cubes=C)
    a = A[:-N_TEST].reshape(-1, A.shape[-1])
    json.dump(dict(a_mean=a.mean(0).tolist(), a_std=a.std(0).tolist()), open(f'{OUT}/action_stats.json', 'w'))
    print(f'expert: {S.shape} in {time.time() - t0:.0f}s; action mean {a.mean(0).round(2)} std {a.std(0).round(2)}')


def stage_data(n=50000, names=('EXP', 'RND1', 'RND4'), seed=0):
    E = np.load(f'{OUT}/expert.npz'); S, A = E['snaps'], E['actions']
    st = json.load(open(f'{OUT}/action_stats.json')); mu, sd = np.array(st['a_mean']), np.array(st['a_std'])
    rng = np.random.default_rng(seed)
    n_train = N_TRAJ - N_TEST
    anchors_all = np.array([(i, t) for i in range(n_train) for t in range(T_TRAJ - WIN)])
    seq = lambda it: A[it[0], it[1]:it[1] + WIN]
    marg = anchors_all[rng.choice(len(anchors_all), 4 * n, replace=True)]  # shared marginal pool
    os.makedirs(f'{OUT}/data', exist_ok=True)
    for name in names:
        B = 1 if name in ('EXP', 'RND1') else int(name[3:])
        anc = anchors_all[rng.choice(len(anchors_all), n // B, replace=False)]
        jobs, acts, ids, mp = [], [], [], 0
        for k, it in enumerate(anc):
            if name == 'EXP':
                seqs = seq(it)[None]
            else:
                seqs = np.stack([seq(m) for m in marg[mp:mp + B]]); mp += B
            jobs.append((S[it[0], it[1]], seqs)); acts.append(seqs); ids += [k] * B
        t0 = time.time()
        with pool() as p:
            frames = np.concatenate(p.map(_w_windows, jobs, chunksize=64))
        acts = np.concatenate(acts)
        an = ((acts - mu) / sd).reshape(len(acts), 4, -1).astype(np.float32)
        np.savez(f'{OUT}/data/{name}.npz', frames=frames, actions=an, anchor=np.array(ids))
        print(f'{name}: {frames.shape} {time.time() - t0:.0f}s  |a_norm| mean {np.abs(an).mean():.3f}', flush=True)


def test_tasks(n, seed, offset=25, min_move=0.05):
    E = np.load(f'{OUT}/expert.npz'); C = E['cubes']
    cand = [(i, t) for i in range(N_TRAJ - N_TEST, N_TRAJ) for t in range(T_TRAJ - offset)
            if np.linalg.norm(C[i, t + offset] - C[i, t]) >= min_move]
    rng = np.random.default_rng(1000 + seed)
    pick = rng.choice(len(cand), n, replace=False)
    return [cand[j] for j in pick], E


def stage_cfbuild(n=400, seed=7):
    st = json.load(open(f'{OUT}/action_stats.json')); mu, sd = np.array(st['a_mean']), np.array(st['a_std'])
    E = np.load(f'{OUT}/expert.npz'); S, A = E['snaps'], E['actions']
    rng = np.random.default_rng(seed)
    test = [(i, t) for i in range(N_TRAJ - N_TEST, N_TRAJ) for t in range(T_TRAJ - 30)]
    train = [(i, t) for i in range(N_TRAJ - N_TEST) for t in range(T_TRAJ - 5)]
    jobs, U = [], []
    for j in rng.choice(len(test), n, replace=False):
        i, t = test[j]
        fact = A[i, t:t + 5]
        marg = np.stack([A[a][b:b + 5] for a, b in (train[k] for k in rng.choice(len(train), 8))])
        gauss = np.clip(rng.standard_normal((8, 5, A.shape[-1])) * sd + mu, -1, 1)
        blocks = np.concatenate([fact[None], marg, gauss])
        U.append(((blocks - mu) / sd).reshape(len(blocks), 1, -1))
        jobs.append((S[i, t], S[i, t + 25], blocks))
    with pool() as p:
        res = p.map(_w_cf, jobs, chunksize=8)
    np.savez(f'{OUT}/cf_groundtruth.npz', img0=np.stack([r[0] for r in res]), gimg=np.stack([r[1] for r in res]),
             imgs=np.stack([r[2] for r in res]), cdist=np.stack([r[3] for r in res]),
             U=np.stack(U).astype(np.float32))
    print('cf ground truth built', len(res))


def load_wm(path):
    from ctrlgeo.wm import FrozenLeWM
    st = json.load(open(f'{OUT}/action_stats.json'))
    return FrozenLeWM(path=path, a_mean=st['a_mean'], a_std=st['a_std'])


def stage_cf(models):
    import torch
    D = np.load(f'{OUT}/cf_groundtruth.npz')
    for path in models:
        wm = load_wm(path)
        with torch.no_grad():
            n, A = D['U'].shape[:2]
            z0 = wm.encode(D['img0'])
            zt = wm.encode(D['imgs'].reshape(n * A, *D['imgs'].shape[2:])).view(n, A, -1)
            U = torch.tensor(D['U'], device='cuda')
            zp = wm.phi(z0.repeat_interleave(A, 0), U.view(n * A, 1, -1)).view(n, A, -1)
            pair = torch.cdist(zt, zt).pow(2)
            scale = pair.sum((1, 2)) / (A * (A - 1))
            err = ((zp - zt).pow(2).sum(-1) / scale[:, None]).cpu().numpy()
            d_pt = torch.cdist(zp[:, 1:], zt[:, 1:])
            top1 = (d_pt.argmin(-1) == torch.arange(A - 1, device='cuda')).float().mean().item()
        gm = lambda x: float(np.exp(np.log(x + 1e-8).mean()))
        r = dict(model=path, cf_err_geo=gm(err[:, 1:]), cf_top1=top1, factual_err_geo=gm(err[:, 0]),
                 marg_err_geo=gm(err[:, 1:9]), gauss_err_geo=gm(err[:, 9:]))
        print(json.dumps(r), flush=True)
        with open(f'{OUT}/cf_eval.jsonl', 'a') as f:
            f.write(json.dumps(r) + '\n')


def stage_plan(models, seeds, n_ep=50, budget=50, H=5, samples=300, iters=30):
    import torch
    from ctrlgeo import cube_sim as cs
    from ctrlgeo.planner import TerminalValue, cem_plan
    for path in models:
        wm = load_wm(path)
        value = TerminalValue(wm, kind='l2')
        for seed in seeds:
            tasks, E = test_tasks(n_ep, seed)
            S, C = E['snaps'], E['cubes']
            envs = [cs.make_env(RES) for _ in range(n_ep)]
            gimgs = []
            for e, (i, t) in zip(envs, tasks):
                cs.restore(e, S[i, t + 25]); gimgs.append(cs.render(e))
                cs.restore(e, S[i, t])
            gcube = np.stack([C[i, t + 25] for i, t in tasks])
            zg = wm.encode(np.stack(gimgs))
            done = np.zeros(n_ep, bool)
            gen = torch.Generator(device='cuda').manual_seed(seed)
            steps, t0 = 0, time.time()
            while steps < budget:
                z0 = wm.encode(np.stack([cs.render(e) for e in envs]))
                mu, _ = cem_plan(wm, z0, zg, value, H=H, n_samples=samples, n_iters=iters,
                                 topk=samples // 10, gen=gen)
                raw = wm.denorm_actions(mu[:, :5].cpu().numpy())
                for k in range(raw.shape[1]):
                    if steps >= budget:
                        break
                    for j, e in enumerate(envs):
                        cs.step(e, raw[j, k])
                        done[j] |= np.linalg.norm(cs.cube_pos(e) - gcube[j]) <= 0.04
                    steps += 1
            r = dict(model=path, seed=seed, success=float(done.mean() * 100), time=time.time() - t0,
                     ep_success=done.astype(int).tolist())
            print(json.dumps({k: v for k, v in r.items() if k != 'ep_success'}), flush=True)
            with open(f'{OUT}/planning.jsonl', 'a') as f:
                f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['expert', 'data', 'cfbuild', 'cf', 'plan'])
    ap.add_argument('--models', nargs='*', default=[])
    ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--n', type=int, default=50000)
    a = ap.parse_args()
    {'expert': stage_expert, 'data': lambda: stage_data(a.n), 'cfbuild': stage_cfbuild,
     'cf': lambda: stage_cf(a.models), 'plan': lambda: stage_plan(a.models, a.seeds)}[a.stage]()
