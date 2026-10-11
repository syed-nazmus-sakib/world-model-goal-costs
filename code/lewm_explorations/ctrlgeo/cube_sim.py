"""OGBench cube-single simulator helpers (ground truth for the Cube replication).

Exact save/restore via MuJoCo's integration state (includes mocap targets), the
OGBench Markov oracle as the expert collection policy (as in OGBench play data),
and the OGBench success criterion (cube within 4 cm of its goal position).
"""
import gymnasium as gym
import mujoco
import numpy as np
import ogbench  # noqa: F401  (registers envs)
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
ACT_DIM = 5


def make_env(res=112):
    env = gym.make('cube-single-v0', ob_type='pixels', width=res, height=res,
                   mode='data_collection', visualize_info=False, terminate_at_goal=False)
    env.reset(seed=0)
    return env


def _md(env):
    u = env.unwrapped
    return u._model, u._data


def snapshot(env):
    m, d = _md(env)
    s = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
    mujoco.mj_getState(m, d, s, STATE_SPEC)
    return s


def restore(env, s):
    m, d = _md(env)
    mujoco.mj_setState(m, d, s, STATE_SPEC)
    mujoco.mj_forward(m, d)


def render(env):
    return env.unwrapped.render().copy()


def cube_pos(env):
    return env.unwrapped._data.joint('object_joint_0').qpos[:3].copy()


def step(env, a):
    """One control step. Convention: always follow mj_step with mj_forward so the IK
    controller reads fresh kinematics; recording and replay then agree exactly."""
    out = env.unwrapped.step(np.clip(np.asarray(a, dtype=np.float64), -1, 1))
    m, d = _md(env)
    mujoco.mj_forward(m, d)
    return out


def expert_rollout(env, n_steps, rng, action_noise=0.1):
    """OGBench-style play collection: Markov oracle toward random targets + Gaussian noise.
    Returns per-step snapshots (n_steps+1) and actions (n_steps, 5)."""
    ob, info = env.reset(seed=int(rng.integers(1 << 31)))
    restore(env, snapshot(env))  # apply the mj_forward convention from the first state
    oracle = CubeMarkovOracle(env=env, min_norm=0.4)
    oracle.reset(ob, info)
    snaps, acts, cubes = [snapshot(env)], [], [cube_pos(env)]
    for _ in range(n_steps):
        if oracle.done:
            ob, info = env.unwrapped.set_new_target()
            oracle.reset(ob, info)
        a = oracle.select_action(ob, info)
        a = np.clip(a + rng.normal(0, action_noise, a.shape), -1, 1)
        ob, _, _, _, info = step(env, a)
        acts.append(a); snaps.append(snapshot(env)); cubes.append(cube_pos(env))
    return np.stack(snaps), np.stack(acts), np.stack(cubes)
