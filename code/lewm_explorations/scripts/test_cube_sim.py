import time, numpy as np
from ctrlgeo import cube_sim as cs
env = cs.make_env(112)
rng = np.random.default_rng(0)
t0 = time.time(); S, A, C = cs.expert_rollout(env, 300, rng); t1 = time.time()
print('expert 300 steps', round(t1 - t0, 2), 's; cube moved', np.round(np.linalg.norm(C[-1] - C[0]), 3), 'action std', A.std(0).round(2), 'mean |a|', np.abs(A).mean(0).round(2))
# determinism of restore + replay
cs.restore(env, S[100]); [cs.step(env, a) for a in A[100:120]]; s1 = cs.snapshot(env); i1 = cs.render(env)
cs.restore(env, S[100]); [cs.step(env, a) for a in A[100:120]]; s2 = cs.snapshot(env); i2 = cs.render(env)
print('replay determinism max|dstate|', np.abs(s1 - s2).max(), 'img', np.abs(i1.astype(int) - i2).max())
print('replay vs recorded traj max|dqpos|', np.abs(s1 - S[120]).max())
t0 = time.time(); [cs.render(env) for _ in range(100)]; print('render/s', 100 / (time.time() - t0))
t0 = time.time(); cs.restore(env, S[0]); [cs.step(env, np.zeros(5)) for _ in range(200)]; print('steps/s', 200 / (time.time() - t0))
moves = np.linalg.norm(C[25:] - C[:-25], axis=1); print('frac windows with cube move>4cm over 25 steps', (moves > 0.04).mean().round(2))
from PIL import Image; Image.fromarray(np.concatenate([cs.render(env) for _ in [cs.restore(env, S[k]) for k in (0, 100, 200, 300)] if True] if False else [ (cs.restore(env, S[k]), cs.render(env))[1] for k in (0, 100, 200, 300)], 1)).save('/tmp/claude-1000/-home-rmedu2025-Monaf-new/a0dcd6ec-f8dc-4837-b4eb-94b7dbe2bfca/scratchpad/cube.png')
