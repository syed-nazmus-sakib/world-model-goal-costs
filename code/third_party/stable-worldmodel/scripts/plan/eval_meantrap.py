"""MeanTrap: evaluate a world model with MPC on TwoRoom-Trap start/goal tasks.

Based on eval_wm.py: the model, planner and image transforms are set up
exactly as there. Lines added or changed for MeanTrap are marked
``# MeanTrap``. Instead of start/goal pairs sampled from a dataset, it uses
fixed start/goal tasks around the ice patch (mean-trap.md, section 11):

- ``tasks=ice``: start above the ice and goal below it (first half of the
  episodes), or the reverse (second half). The straight line crosses the ice.
- ``tasks=control``: the same positions shifted 112 px into the right room,
  where there is no ice.

The goal image is the env's own rendering of the agent at the goal
(``_target_img``), held fixed for the whole episode like the dataset goal in
``World.evaluate``. Reports success, fall and ice-entry rates.

Usage (lab PC):
    python scripts/plan/eval_meantrap.py --config-name=tworoom \
        policy=lewm_trap_p0.5/weights_epoch_100.pt \
        eval.dataset_name=tworoom_trap_p0.5.lance \
        +fall_prob=0.5 +tasks=ice seed=42
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from collections import defaultdict  # MeanTrap
from copy import deepcopy  # MeanTrap
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

# MeanTrap: importing trap_env registers swm/TwoRoomTrap-v0
import stable_worldmodel.envs.two_room.trap_env  # noqa: F401
from stable_worldmodel.plot.video_utils import save_panel_videos  # MeanTrap

# MeanTrap: task geometry in pixels (mean-trap.md, section 11)
TASK_X = (45.0, 75.0)  # inside the ice columns [40, 80)
# start/goal rows are >= 16 px (the success radius) away from the ice rows
# [100, 140), so an agent can't reach the goal while still on the ice
ABOVE_Y = (68.0, 84.0)  # above the ice
BELOW_Y = (156.0, 172.0)  # below the ice
CONTROL_SHIFT_X = 112.0  # the same tasks in the right room (no ice)
TASK_SETS = ('ice', 'control')


def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    return dataset


def make_tasks(task_set, n, seed):
    """MeanTrap: start/goal positions, (n, 2) each, the same for every model.

    The first half of the episodes go from above the ice to below it, the
    second half the other way. ``control`` shifts every position into the
    right room.
    """
    rng = np.random.default_rng(seed)
    above = np.stack(
        [rng.uniform(*TASK_X, n), rng.uniform(*ABOVE_Y, n)], axis=1
    )
    below = np.stack(
        [rng.uniform(*TASK_X, n), rng.uniform(*BELOW_Y, n)], axis=1
    )
    down = (np.arange(n) < n // 2)[:, None]
    starts = np.where(down, above, below)
    goals = np.where(down, below, above)
    if task_set == 'control':
        starts[:, 0] += CONTROL_SHIFT_X
        goals[:, 0] += CONTROL_SHIFT_X
    return starts.astype(np.float32), goals.astype(np.float32)


def evaluate_tasks(world, starts, goals, eval_budget, reset_seed, video):
    """MeanTrap: World._evaluate_from_dataset, with start/goal tasks.

    Resets env i to starts[i] with target goals[i], gives the planner the
    env-rendered goal image (held fixed, as in dataset mode), runs at most
    eval_budget steps and records success, fall and ice entry per episode.
    """
    n = world.num_envs
    assert len(starts) == len(goals) == n
    options = [
        {'state': starts[i], 'target_state': goals[i]} for i in range(n)
    ]
    world.reset(seed=reset_seed, options=options)
    envs = [world.envs.envs[i].unwrapped for i in range(n)]

    # goal image: the env's own frame with the agent drawn at the goal
    goal_imgs = np.stack(
        [e._target_img.permute(1, 2, 0).cpu().numpy() for e in envs]
    )
    shape_prefix = world.infos['pixels'].shape[:2]
    frame_shape = world.infos['pixels'].shape[2:]
    assert goal_imgs.dtype == np.uint8, goal_imgs.dtype
    assert goal_imgs.shape[1:] == frame_shape, (goal_imgs.shape, frame_shape)
    world.infos['goal'] = np.broadcast_to(
        goal_imgs[:, None], shape_prefix + goal_imgs.shape[1:]
    ).copy()
    world.infos['goal_proprio'] = np.broadcast_to(
        goals[:, None], shape_prefix + goals.shape[1:]
    ).copy()
    goal_snapshot = {
        k: world.infos[k].copy() for k in ('goal', 'goal_proprio')
    }

    success = np.zeros(n, dtype=bool)
    fell = np.zeros(n, dtype=bool)
    entered_ice = np.zeros(n, dtype=bool)
    frames = defaultdict(list) if video else None

    def on_step(world, mask):
        world.infos.update(deepcopy(goal_snapshot))
        success[:] |= world.terminateds
        for i, e in enumerate(envs):
            fell[i] |= bool(e.fallen)
            entered_ice[i] |= e.on_ice(e.agent_position)
        if frames is not None:
            for i in range(n):
                f = world.infos['pixels'][i]
                frame = f[-1] if f.ndim > 3 else f
                frames[i].append(np.asarray(frame).copy())

    world._run(max_steps=eval_budget, mode='wait', on_step=on_step)

    # a fallen agent can never reach the goal
    assert not (success & fell).any(), 'success after a fall'
    if frames:
        save_panel_videos(
            Path(video), {'agent': frames, 'goal': list(goal_imgs)}
        )
    return {
        'success_rate': float(success.mean()) * 100.0,
        'fall_rate': float(fell.mean()) * 100.0,
        'ice_entry_rate': float(entered_ice.mean()) * 100.0,
        'episode_successes': success,
        'episode_falls': fell,
        'episode_ice_entries': entered_ice,
    }


# MeanTrap: default config is tworoom (eval_wm.py uses pusht)
@hydra.main(version_base=None, config_path='./config', config_name='tworoom')
def run(cfg: DictConfig):
    """Run MeanTrap evaluation of a world model vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # MeanTrap: required settings
    if cfg.get('fall_prob') is None or cfg.get('tasks') is None:
        raise ValueError('set +fall_prob=<p> and +tasks=<ice|control>')
    fall_prob = float(cfg.fall_prob)
    task_set = str(cfg.tasks)
    if task_set not in TASK_SETS:
        raise ValueError(f'tasks must be one of {TASK_SETS}, got {task_set}')
    if not str(cfg.eval.dataset_name).startswith('tworoom_trap_'):
        raise ValueError(
            'set eval.dataset_name to the model\'s training dataset '
            '(action/proprio normalization), e.g. tworoom_trap_p0.5.lance'
        )
    cfg.world.env_name = 'swm/TwoRoomTrap-v0'

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(
        **cfg.world,
        image_shape=(224, 224),
        fall_prob=fall_prob,  # MeanTrap
    )

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = cfg.get('policy', 'random')

    if policy != 'random':
        model = swm.wm.utils.load_pretrained(cfg.policy)
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = (
                'backbone' if hasattr(model, 'backbone') else 'encoder'
            )
            setattr(
                model,
                encoder_attr,
                torch.compile(getattr(model, encoder_attr)),
            )
            model.predictor = torch.compile(model.predictor)
        config = swm.PlanConfig(**cfg.plan_config)
        objective = hydra.utils.instantiate(cfg.objective)
        cost = swm.planning.ShootingCostEvaluator(model, objective)
        solver = hydra.utils.instantiate(cfg.solver, cost=cost)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    # MeanTrap: tasks and one output folder per run under <model>/meantrap/
    starts, goals = make_tasks(task_set, cfg.eval.num_eval, cfg.seed)
    results_path = results_path / 'meantrap'
    run_name = (
        f'{Path(str(cfg.policy)).stem}_{task_set}_p{fall_prob}_seed{cfg.seed}'
    )
    video_path = results_path / run_name

    world.set_policy(policy)

    video_path.mkdir(parents=True, exist_ok=True)
    print(
        f'[eval] saving videos to {video_path.resolve()} '
        '(one env_{i}.mp4 per env)'
    )

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    start_time = time.time()
    with autocast_ctx:
        metrics = evaluate_tasks(
            world,
            starts,
            goals,
            cfg.eval.eval_budget,
            reset_seed=1000 * int(cfg.seed),  # MeanTrap: disjoint per seed
            video=video_path,
        )
    end_time = time.time()

    # MeanTrap: one summary line per run
    summary = (
        f'[meantrap] model={cfg.policy} fall_prob={fall_prob} '
        f'tasks={task_set} seed={cfg.seed} '
        f'success_rate={metrics["success_rate"]:.1f} '
        f'fall_rate={metrics["fall_rate"]:.1f} '
        f'ice_entry_rate={metrics["ice_entry_rate"]:.1f} '
        f'n={len(starts)}'
    )
    print(summary)
    print(f'[eval] videos saved to {video_path.resolve()}')
    with (results_path / 'meantrap_results.txt').open('a') as f:
        f.write(summary + '\n')

    with (video_path / cfg.output.filename).open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(summary + '\n')
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
