"""Calibrate Cube planning tasks: null actions, random actions, expert replay."""
import numpy as np, json
from ctrlgeo import cube_sim as cs
import importlib.util
spec = importlib.util.spec_from_file_location('e6', 'scripts/exp6_cube.py'); e6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e6)
env = cs.make_env(112)
for seed in (0, 1, 2):
    tasks, E = e6.test_tasks(50, seed); S, A, C = E['snaps'], E['actions'], E['cubes']
    res = {}
    for pol in ('null', 'random', 'expert'):
        rng = np.random.default_rng(seed); ok = []
        for i, t in tasks:
            cs.restore(env, S[i, t]); g = C[i, t + 25]; d = False
            for k in range(50):
                a = np.zeros(5) if pol == 'null' else rng.uniform(-1, 1, 5) if pol == 'random' else A[i, t + k] if t + k < A.shape[1] else np.zeros(5)
                cs.step(env, a); d |= np.linalg.norm(cs.cube_pos(env) - g) <= 0.04
            ok.append(d)
        res[pol] = 100 * np.mean(ok)
    print(json.dumps(dict(seed=seed, **res)))
