"""MeanTrap G2: oracle planners on TwoRoom-Trap (CPU only; mean-trap.md, 12).

Same tasks, World, CEM solver and WorldModelPolicy as eval_meantrap.py; only
the cost is different. Instead of a learned world model, each candidate plan
is scored by simulating the true TwoRoom-Trap dynamics: `move` and
`fall_check` below copy TwoRoomEnv.step / TwoRoomTrapEnv.step exactly
(checked by scripts/plan/check_meantrap_oracle.py). Cost = squared distance
from the agent to the goal at the plan's last step, the state-space twin of
LeWM's last-step objective.

- +oracle=expected (O1): K sampled futures per candidate. Falls are sampled
  with the env's per-step hazard and a fallen agent stays frozen; the cost is
  the mean over the K futures. All candidates in a CEM iteration share the
  same random draws, which are redrawn every iteration.
- +oracle=mean (O3): one rollout of the ideal deterministic (mean) model.
  From a position on the ice the agent moves (1 - hazard) of its step, the
  average of having fallen there or not; it never falls. The first step uses
  the true current state (known not fallen).

Usage (lab PC, CPU):
    python scripts/plan/eval_meantrap_oracle.py \
        eval.dataset_name=tworoom_trap_p0.5.lance \
        +fall_prob=0.5 +tasks=ice +oracle=expected seed=0
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from sklearn import preprocessing  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from eval_meantrap import (  # noqa: E402  (also registers the trap env)
    TASK_SETS,
    evaluate_tasks,
    get_dataset,
    make_tasks,
)
from stable_worldmodel.envs.two_room.trap_env import TwoRoomTrapEnv  # noqa: E402

ORACLES = ('expected', 'mean')
DOOR_MARGIN = 1.75  # TwoRoomEnv._apply_collisions


def dynamics_params(env):
    """Constants of TwoRoomEnv.step/_apply_collisions for this env."""
    if int(env.wall_axis) != 1:
        raise ValueError('oracle dynamics assume the vertical central wall')
    agent_r = float(env.variation_space['agent']['radius'].value.item())
    bs = float(env.BORDER_SIZE)
    c = float(env.WALL_CENTER)
    half = env.wall_thickness // 2
    return {
        'speed': float(env.variation_space['agent']['speed'].value.item()),
        'lo': bs + agent_r,
        'hi': env.IMG_SIZE - bs - agent_r,
        'center': c,
        'eff_left': c - half - agent_r,
        'eff_right': c + half + agent_r,
        'doors': tuple(
            (float(env.door_positions[i]), float(env.door_sizes[i]))
            for i in range(env.num_doors)
        ),
        'hazard': float(env.hazard),
    }


def move(pos, act, p):
    """TwoRoomEnv.step position update, vectorized (float32 like the env).

    pos (..., 2) float32 agent centers, act (..., 2) env actions; they
    broadcast against each other.
    """
    a = np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)
    nxt = pos + a * np.float32(p['speed'])
    # border clamp in float64, then stored as float32 (torch.tensor(...))
    x = np.clip(nxt[..., 0].astype(np.float64), p['lo'], p['hi'])
    y = np.clip(nxt[..., 1].astype(np.float64), p['lo'], p['hi'])
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # central wall, unless the new y is inside a door
    y64 = y.astype(np.float64)
    in_door = np.zeros(y.shape, dtype=bool)
    for c, s in p['doors']:
        in_door |= (c - s - DOOR_MARGIN <= y64) & (y64 <= c + s + DOOR_MARGIN)
    x64 = x.astype(np.float64)
    started_left = np.broadcast_to(
        pos[..., 0].astype(np.float64) < p['center'], x.shape)
    x = np.where(started_left & (x64 > p['eff_left']) & ~in_door,
                 np.float32(p['eff_left'] - 0.5), x)
    x = np.where(~started_left & (x64 < p['eff_right']) & ~in_door,
                 np.float32(p['eff_right'] + 0.5), x)
    return np.stack([x, y], axis=-1)


def on_ice(pos):
    """TwoRoomTrapEnv.on_ice, vectorized."""
    x, y = pos[..., 0], pos[..., 1]
    return ((TwoRoomTrapEnv.ICE_X[0] <= x) & (x < TwoRoomTrapEnv.ICE_X[1])
            & (TwoRoomTrapEnv.ICE_Y[0] <= y) & (y < TwoRoomTrapEnv.ICE_Y[1]))


def fall_check(fallen, new_pos, u, hazard):
    """TwoRoomTrapEnv.step: after moving, an agent on the ice falls if u <
    hazard (u ~ U[0, 1)); falling is permanent."""
    return fallen | (on_ice(new_pos) & (u < hazard))


class OracleCost:
    """Costable for CEMSolver: scores candidates in the true simulator."""

    def __init__(self, oracle, envs, action_scaler, k, seed):
        assert oracle in ORACLES, oracle
        self.oracle = oracle
        self.envs = envs
        self.scaler = action_scaler
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.params = None

    def _check_params(self):
        params = [dynamics_params(e) for e in self.envs]
        assert all(q == params[0] for q in params), 'envs differ'
        self.params = params[0]
        print(f'[oracle] dynamics: {self.params}')

    @staticmethod
    def _row(value, batch, dim=None):
        """(B, S, [T,] [dim]) solver info -> (B, [dim]) at the latest step."""
        a = value.detach().cpu().numpy() if torch.is_tensor(value) else (
            np.asarray(value))
        a = a[:, 0]  # every candidate carries the same copy
        if dim is None:
            return a.reshape(batch, -1)[:, -1]
        assert a.shape[-1] == dim, f'expected last dim {dim}: {a.shape}'
        return a.reshape(batch, -1, dim)[:, -1]

    def _check_rows(self, pos0, goal, fallen0):
        """Each planned row must be the current state of a real env."""
        env_pos = np.stack([e.agent_position.numpy() for e in self.envs])
        env_goal = np.stack([e.target_position.numpy() for e in self.envs])
        env_fallen = np.array([bool(e.fallen) for e in self.envs])
        for b in range(len(pos0)):
            match = (np.all(env_pos == pos0[b], axis=1)
                     & np.all(env_goal == goal[b], axis=1))
            assert match.any(), f'row {b} matches no env: {pos0[b]}'
            assert (env_fallen[match] == fallen0[b]).all(), 'fallen mismatch'

    def get_cost(self, info_dict, action_candidates):
        if self.params is None:
            self._check_params()
        B, S, H, D = action_candidates.shape
        # Same conversion as WorldModelPolicy.get_action: blocks of D values
        # are D/2 consecutive 2-D actions, un-normalized by the action scaler.
        flat = action_candidates.detach().cpu().numpy().astype(np.float32)
        act = self.scaler.inverse_transform(flat.reshape(-1, 2))
        act = np.asarray(act, dtype=np.float32).reshape(B, S, H * D // 2, 2)

        pos0 = self._row(info_dict['state'], B, 2).astype(np.float32)
        goal = self._row(info_dict['goal_state'], B, 2).astype(np.float32)
        fallen0 = self._row(info_dict['fallen'], B).astype(bool)
        self._check_rows(pos0, goal, fallen0)
        if self.oracle == 'expected':
            cost = self._expected(pos0, fallen0, goal, act)
        else:
            cost = self._mean(pos0, fallen0, goal, act)
        return torch.as_tensor(cost, dtype=torch.float32)

    def _expected(self, pos0, fallen0, goal, act):
        B, S, T, _ = act.shape
        hazard = self.params['hazard']
        u = self.rng.random((B, 1, self.k, T))  # shared by all candidates
        pos = np.broadcast_to(pos0[:, None, None], (B, S, self.k, 2)).copy()
        fallen = np.broadcast_to(fallen0[:, None, None], (B, S, self.k)).copy()
        for t in range(T):
            new = move(pos, act[:, :, None, t], self.params)
            pos = np.where(fallen[..., None], pos, new)
            fallen = fall_check(fallen, pos, u[..., t], hazard)
        d2 = ((pos.astype(np.float64) - goal[:, None, None]) ** 2).sum(-1)
        return d2.mean(-1)

    def _mean(self, pos0, fallen0, goal, act):
        B, S, T, _ = act.shape
        keep = np.float32(1.0 - self.params['hazard'])
        pos = np.broadcast_to(pos0[:, None], (B, S, 2)).copy()
        alive = np.broadcast_to(~fallen0[:, None], (B, S)).astype(np.float32)
        for t in range(T):
            new = move(pos, act[:, :, t], self.params)
            frac = alive if t == 0 else alive * np.where(
                on_ice(pos), keep, np.float32(1.0))
            pos = (pos + frac[..., None] * (new - pos)).astype(np.float32)
        return ((pos.astype(np.float64) - goal[:, None]) ** 2).sum(-1)


@hydra.main(version_base=None, config_path='./config', config_name='tworoom')
def run(cfg: DictConfig):
    if None in (cfg.get('fall_prob'), cfg.get('tasks'), cfg.get('oracle')):
        raise ValueError('set +fall_prob=<p> +tasks=<ice|control> '
                         '+oracle=<expected|mean>')
    fall_prob = float(cfg.fall_prob)
    task_set, oracle = str(cfg.tasks), str(cfg.oracle)
    if task_set not in TASK_SETS or oracle not in ORACLES:
        raise ValueError(f'tasks in {TASK_SETS}, oracle in {ORACLES}')
    if not str(cfg.eval.dataset_name).startswith('tworoom_trap_'):
        raise ValueError('set eval.dataset_name=tworoom_trap_p0.5.lance '
                         '(action normalization, as for the p0.5 model)')
    k = int(cfg.get('oracle_k', 64))

    # World and action normalization exactly as in eval_meantrap.py
    cfg.world.env_name = 'swm/TwoRoomTrap-v0'
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224),
                      fall_prob=fall_prob)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != 'action':
            process[f'goal_{col}'] = process[col]

    envs = [world.envs.envs[i].unwrapped for i in range(world.num_envs)]
    cost = OracleCost(oracle, envs, process['action'], k, seed=int(cfg.seed))
    solver = hydra.utils.instantiate(cfg.solver, cost=cost, device='cpu')
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config),
        process=process, transform={})
    world.set_policy(policy)

    starts, goals = make_tasks(task_set, cfg.eval.num_eval, cfg.seed)
    name = f'oracle_{oracle}'
    results_path = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), 'oracle',
        'meantrap')
    video_path = results_path / f'{name}_{task_set}_p{fall_prob}_seed{cfg.seed}'
    video_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = evaluate_tasks(world, starts, goals, cfg.eval.eval_budget,
                             reset_seed=1000 * int(cfg.seed), video=video_path)
    end_time = time.time()

    summary = (
        f'[meantrap] model={name} fall_prob={fall_prob} tasks={task_set} '
        f'seed={cfg.seed} success_rate={metrics["success_rate"]:.1f} '
        f'fall_rate={metrics["fall_rate"]:.1f} '
        f'ice_entry_rate={metrics["ice_entry_rate"]:.1f} n={len(starts)}'
        + (f' k={k}' if oracle == 'expected' else '')
    )
    print(summary)
    with (results_path / 'meantrap_results.txt').open('a') as f:
        f.write(summary + '\n')
    with (video_path / cfg.output.filename).open('a') as f:
        f.write('\n==== CONFIG ====\n' + OmegaConf.to_yaml(cfg) + '\n')
        f.write('==== RESULTS ====\n' + summary + '\n')
        f.write('episode start_xy goal_xy success fell entered_ice\n')
        for i in range(len(starts)):
            f.write(
                f'{i} {starts[i].astype(float).round(1).tolist()} '
                f'{goals[i].astype(float).round(1).tolist()} '
                f'{int(metrics["episode_successes"][i])} '
                f'{int(metrics["episode_falls"][i])} '
                f'{int(metrics["episode_ice_entries"][i])}\n'
            )
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
