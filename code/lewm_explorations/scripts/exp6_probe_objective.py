"""Diagnostic (not part of the pre-registered K1): is the Cube data effect masked by the
L2 latent objective? Replace the CEM cost with a task-aligned one: a ridge probe
latent -> cube xyz (fitted offline on expert frames with privileged labels), cost =
||probe(z_T) - probe(z_goal)||^2. The goal is still given only as an image.
"""
import importlib.util, json
import numpy as np
import torch

from ctrlgeo import cube_sim as cs
from ctrlgeo.planner import cem_plan

spec = importlib.util.spec_from_file_location('e6', 'scripts/exp6_cube.py')
e6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e6)

E = np.load('results/exp6/expert.npz'); S, C = E['snaps'], E['cubes']
rng = np.random.default_rng(0)
env = cs.make_env(112)
idx = [(i, t) for i, t in zip(rng.integers(0, 360, 6000), rng.integers(0, 501, 6000))]
imgs, ys = [], []
for i, t in idx:
    cs.restore(env, S[i, t]); imgs.append(cs.render(env)); ys.append(C[i, t])
imgs, ys = np.stack(imgs), np.stack(ys)

for name in ('EXP_s0', 'RND1_s0', 'RND4_s0'):
    wm = e6.load_wm(f'results/exp6/models/{name}')
    Z = wm.encode(imgs)
    Zt, Yt = Z[:5000], torch.tensor(ys[:5000], device='cuda', dtype=torch.float32)
    X = torch.cat([Zt, torch.ones(len(Zt), 1, device='cuda')], 1)
    W = torch.linalg.solve(X.T @ X + 1e-2 * torch.eye(X.shape[1], device='cuda'), X.T @ Yt)
    probe = lambda z: torch.cat([z, torch.ones(len(z), 1, device=z.device)], 1) @ W
    pv = probe(Z[5000:]).cpu().numpy()
    r2 = 1 - ((pv - ys[5000:]) ** 2).sum() / ((ys[5000:] - ys[5000:].mean(0)) ** 2).sum()
    value = lambda zT, zg: (probe(zT) - probe(zg)).pow(2).sum(-1)
    for seed in (0, 1, 2):
        tasks_all, _ = e6.test_tasks(50, seed)
        done_all = []
        for c0 in range(0, len(tasks_all), 10):  # chunks of 10 envs to fit next to training jobs
            tasks = tasks_all[c0:c0 + 10]
            envs = [cs.make_env(112) for _ in tasks]
            g = []
            for ee, (i, t) in zip(envs, tasks):
                cs.restore(ee, S[i, t + 25]); g.append(cs.render(ee)); cs.restore(ee, S[i, t])
            gcube = np.stack([C[i, t + 25] for i, t in tasks]); zg = wm.encode(np.stack(g))
            done = np.zeros(len(tasks), bool); gen = torch.Generator(device='cuda').manual_seed(seed * 100 + c0); steps = 0
            while steps < 50:
                z0 = wm.encode(np.stack([cs.render(ee) for ee in envs]))
                mu, _ = cem_plan(wm, z0, zg, value, H=5, n_samples=300, n_iters=30, topk=30, gen=gen)
                raw = wm.denorm_actions(mu[:, :5].cpu().numpy())
                for k in range(raw.shape[1]):
                    if steps >= 50: break
                    for j, ee in enumerate(envs):
                        cs.step(ee, raw[j, k]); done[j] |= np.linalg.norm(cs.cube_pos(ee) - gcube[j]) <= 0.04
                    steps += 1
            done_all.append(done)
            del envs
        done = np.concatenate(done_all)
        r = dict(model=name, objective='cube_probe', probe_R2=round(float(r2), 3), seed=seed,
                 success=float(done.mean() * 100))
        print(json.dumps(r), flush=True)
        with open('results/exp6/planning_probe.jsonl', 'a') as f:
            f.write(json.dumps(r) + '\n')
