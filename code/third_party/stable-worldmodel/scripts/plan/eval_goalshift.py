"""Script to evaluate a World Model using MPC on a dataset of episodes.

GoalShift version of eval_wm.py. Everything is identical to eval_wm.py except
the lines marked "GoalShift": the goal image given to the planner is
re-rendered from the same goal state with a changed look (GOAL_CONDITIONS).
The environment the agent acts in is never changed. PushT only.

Usage:
    python scripts/plan/eval_goalshift.py --config-name=pusht \
        policy=quentinll/lewm-pusht seed=0 +condition=bg_color
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

import importlib  # GoalShift
import time
import zlib  # GoalShift
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image  # GoalShift
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from stable_worldmodel.envs.pusht.env import PushT  # GoalShift


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


def episode_col(dataset):
    """Name of the episode-index column, robust across dataset formats.

    HDF5 lists every column (including index columns) in ``column_names``,
    but the Lance reader deliberately excludes its index columns
    (``episode_idx``/``step_idx``) from ``column_names`` and only exposes
    them via ``_schema_names``. Consult both so ``episode_idx`` is found
    regardless of format.
    """
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def get_episodes_length(dataset, episodes):
    col_name = episode_col(dataset)

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    return dataset


# ---------------------------------------------------------------------------
# GoalShift
# ---------------------------------------------------------------------------

# Variation values used to re-render the goal image. PushT defaults are:
# background white, agent RoyalBlue, block LightSlateGray.
GOAL_CONDITIONS = {
    'identity': {},  # control: same look as the dataset
    'bg_color': {
        'background.color': np.array([255, 228, 181], dtype=np.uint8)
    },  # Moccasin
    'agent_color': {
        'agent.color': np.array([220, 20, 60], dtype=np.uint8)
    },  # Crimson
    'block_color': {
        'block.color': np.array([255, 140, 0], dtype=np.uint8)
    },  # DarkOrange
    'no_agent': {},  # agent moved off-canvas before rendering
    # task-equivalent goals (2026-10-19): same block pose, agent drawn elsewhere
    # (fixed distance, random direction, not touching the block); the block-only
    # success is unchanged by construction
    'agent_moved_40': {},
    'agent_moved_80': {},
    'agent_moved_160': {},
}


class GoalRenderer:
    """Re-render a PushT goal state with the look of one GoalShift condition.

    Uses its own PushT instance, separate from the evaluation envs.
    """

    def __init__(self, condition, resolution):
        self.condition = condition
        self.env = PushT(resolution=resolution)

    def __call__(self, goal_state):
        state = np.array(goal_state, dtype=np.float64)  # copy
        assert state.shape == (7,), f'expected 7-d PushT state: {state.shape}'
        # PushT._set_state() runs one physics step: zero the agent velocity so
        # the agent is drawn at its recorded position.
        state[5:7] = 0.0
        if self.condition == 'no_agent':
            state[0:2] = -1000.0  # far outside the 512x512 canvas
        if self.condition.startswith('agent_moved_'):
            state[0:2] = self._moved_agent(state, float(self.condition.split('_')[-1]))
        options = {'state': state, 'goal_state': state}
        values = GOAL_CONDITIONS[self.condition]
        if values:
            options['variation_values'] = {
                k: v.copy() for k, v in values.items()
            }
        _, info = self.env.reset(seed=0, options=options)
        return info['goal']

    def _moved_agent(self, state, dist, min_gap=15.0, tries=500):
        """Agent position `dist` px from the goal agent position, inside the
        arena and at least `min_gap` px from the block. Deterministic per goal."""
        rng = np.random.default_rng(
            zlib.crc32(np.round(state[:5], 3).tobytes())
        )
        self.env.reset(seed=0, options={'state': state, 'goal_state': state})
        e = self.env
        r = max(s.radius for s in e.agent.shapes)
        for _ in range(tries):
            a = rng.uniform(0, 2 * np.pi)
            p = state[0:2] + dist * np.array([np.cos(a), np.sin(a)])
            if not (40 <= p[0] <= 472 and 40 <= p[1] <= 472):
                continue
            gap = min(s.point_query(tuple(p)).distance for s in e.block.shapes)
            if gap - r > min_gap:
                return p
        raise RuntimeError(f'no valid agent position at {dist} px')


def install_goal_shift(condition, goal_log):
    """Swap the dataset goal image for a re-rendered one (this process only).

    Wraps stable_worldmodel.world.world._extract_init_goal, which
    World.evaluate() calls to read the start and goal rows from the dataset.
    Each (dataset_goal, new_goal) pair is appended to goal_log.
    """
    world_module = importlib.import_module('stable_worldmodel.world.world')
    extract_init_goal = world_module._extract_init_goal
    renderer = None

    def extract_init_goal_with_shift(
        dataset, episodes_idx, start_steps, goal_offset
    ):
        nonlocal renderer
        init_rows, goal_rows, dataset_videos = extract_init_goal(
            dataset, episodes_idx, start_steps, goal_offset
        )
        for row in goal_rows:
            old_goal = row['goal']
            assert old_goal.dtype == np.uint8, old_goal.dtype
            assert old_goal.shape[0] == old_goal.shape[1], old_goal.shape
            if renderer is None:
                renderer = GoalRenderer(condition, old_goal.shape[0])
            new_goal = renderer(row['goal_state'])
            assert new_goal.shape == old_goal.shape, (
                new_goal.shape,
                old_goal.shape,
            )
            assert new_goal.dtype == np.uint8, new_goal.dtype
            goal_log.append((old_goal, new_goal))
            row['goal'] = new_goal
        return init_rows, goal_rows, dataset_videos

    world_module._extract_init_goal = extract_init_goal_with_shift


def install_block_success():
    """Also record block-only success (agent position ignored) on each env.

    The standard PushT success, returned unchanged, also requires the agent to
    be at its goal position, which an agent-free goal image does not show.
    Sets env.goalshift_block_success to True once the block alone has matched.
    """
    eval_state = PushT.eval_state

    def eval_state_with_block(self, goal_state, cur_state):
        result = eval_state(self, goal_state, cur_state)
        block_goal = np.array(goal_state, dtype=np.float64)  # copy
        block_goal[:2] = np.asarray(cur_state, dtype=np.float64)[:2]
        block_ok, _ = eval_state(self, block_goal, cur_state)
        self.goalshift_block_success = bool(
            getattr(self, 'goalshift_block_success', False) or block_ok
        )
        return result

    PushT.eval_state = eval_state_with_block



def install_action_guidance(w, action_scaler, null='stay', null_path=None):
    """Method candidate A (2026-10-22): action guidance at planning time.

    Each predictor step is run twice, with the candidate's newest action and with a
    null action, and the next latent is z_null + w * (z_a - z_null). w = 1 is the
    unchanged model. null='stay': the zero relative action (PushT agent holds its
    position), normalized like every planned action (training-free variant).
    null='learned': the null embedding learned by candA_finetune.py (null_emb.pth in
    the checkpoint folder), used in place of the action embedding.
    """
    from einops import rearrange
    from stable_worldmodel.wm.lewm.lewm import LeWM

    if null not in ('stay', 'learned'):
        raise ValueError(f'unknown null action {null!r}')
    learned = torch.load(null_path) if null == 'learned' else None
    zero = action_scaler.transform(
        np.zeros((1, action_scaler.mean_.shape[0]))
    )[0]

    def rollout(self, info, action_sequence, history_size=None):
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)
        H = info['pixels'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        info['action'] = torch.cat([act_past, action_sequence[:, :, :1]], dim=2)
        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )
        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        acts = torch.cat(
            [
                rearrange(act_past, 'b s ... -> (b s) ...'),
                rearrange(action_sequence, 'b s ... -> (b s) ...'),
            ],
            dim=1,
        )
        all_act_emb = self.action_encoder(acts)
        reps = acts.size(-1) // zero.shape[0]
        null_act = torch.as_tensor(
            np.tile(zero, reps), dtype=acts.dtype, device=acts.device
        )
        if learned is None:
            null_emb = self.action_encoder(null_act.expand(acts.size(0), 1, -1))
        else:
            null_emb = learned.to(all_act_emb)[None, None].expand(acts.size(0), 1, -1)
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))
        for t in range(T):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : H + t]
            z_a = self.predict(emb_trunc, act_trunc)[:, -1]
            if w == 1.0:
                emb_list.append(z_a)
                continue
            act_null = torch.cat([act_trunc[:, :-1], null_emb], dim=1)
            z_0 = self.predict(emb_trunc, act_null)[:, -1]
            emb_list.append(z_0 + w * (z_a - z_0))
        emb = torch.stack(emb_list, dim=1)
        info['predicted_emb'] = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        return info

    LeWM.rollout = rollout
    print(f'[guidance] action guidance installed: w={w}, null={null}')


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # GoalShift: pick the condition and install the two patches
    condition = cfg.get('condition', 'identity')
    if condition not in GOAL_CONDITIONS:
        raise ValueError(
            f'condition must be one of {list(GOAL_CONDITIONS)}, '
            f'got {condition!r}'
        )
    if cfg.world.env_name != 'swm/PushT-v1':
        raise ValueError('eval_goalshift.py only supports swm/PushT-v1')
    goal_log = []
    install_goal_shift(condition, goal_log)
    install_block_success()

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

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
        guide_w = cfg.get('guide_w', None)  # method candidate A
        if guide_w is not None:
            null = cfg.get('guide_null', 'stay')
            install_action_guidance(
                float(guide_w), process['action'], null,
                Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy, 'null_emb.pth')
                if null == 'learned' else None,
            )
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
        if cfg.get('metric_head', None):  # method C: learned task-aligned goal cost
            from metric_cost import MetricGoalCost

            objective = MetricGoalCost(cfg.metric_head, cfg.get('metric_mix', 0.0), cfg.get('metric_agg', 'last'))
        cost = swm.planning.ShootingCostEvaluator(model, objective)
        oracle = cfg.get('oracle', None)  # diagnosis D2: simulator in place of the world model
        if oracle:
            from oracle_cost import OracleCost

            cost = OracleCost(model, objective, oracle, process, transform['pixels'], cfg.get('oracle_workers', 14))
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
    # GoalShift: one folder per condition and seed
    tag = f'_off{cfg.eval.goal_offset_steps}_w{cfg.get("guide_w", 1.0)}_h{cfg.plan_config.horizon}_{cfg.get("oracle", "released")}_{Path(str(cfg.get("metric_head", "none"))).stem}{cfg.get("metric_mix", "")}{cfg.get("metric_agg", "")}'
    results_path = results_path / 'goalshift' / f'{condition}_seed{cfg.seed}{tag}'

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = episode_col(dataset)
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices), size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    # Index the full index columns directly: the Lance reader excludes
    # index columns (episode_idx/step_idx) from get_row_data, but get_col_data
    # exposes them (and both are already cached from the checks above).
    eval_episodes = dataset.get_col_data(col_name)[random_episode_indices]
    eval_start_idx = dataset.get_col_data('step_idx')[random_episode_indices]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)
    print(
        f'[eval] saving videos to {results_path.resolve()} '
        '(one env_{i}.mp4 per env)'
    )

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    if cfg.get('compile', False):
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            n = world.num_envs
            world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist()[:n],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=OmegaConf.to_container(
                    cfg.eval.get('callables'), resolve=True
                ),
                video=results_path,
            )
        print('Warmup done.')

    # GoalShift: count only the main evaluation below (not the warm-up)
    goal_log.clear()
    for env in world.envs.envs:
        env.unwrapped.goalshift_block_success = False

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ),
            video=results_path,
        )
    end_time = time.time()

    print(metrics)
    print(f'[eval] videos saved to {results_path.resolve()}')

    # GoalShift: block-only success, goal-image check, one-line summary
    block_success = np.array(
        [env.unwrapped.goalshift_block_success for env in world.envs.envs]
    )
    block_success_rate = float(block_success.mean()) * 100.0
    if np.any(np.asarray(metrics['episode_successes']) & ~block_success):
        print('[goalshift] WARNING: a standard success without block success')
    goal_mae = [
        np.abs(new.astype(np.float32) - old.astype(np.float32)).mean()
        for old, new in goal_log
    ]
    summary = (
        f'condition={condition} seed={cfg.seed} '
        f'goal_offset={cfg.eval.goal_offset_steps} budget={cfg.eval.eval_budget} '
        f'guide_w={cfg.get("guide_w", 1.0)} null={cfg.get("guide_null", "stay")} policy={cfg.policy} '
        f'horizon={cfg.plan_config.horizon} oracle={cfg.get("oracle", "released")} '
        f'metric={Path(str(cfg.get("metric_head", "none"))).stem} mix={cfg.get("metric_mix", 0.0)} agg={cfg.get("metric_agg", "last")} '
        f'success_rate={metrics["success_rate"]:.1f} '
        f'block_success_rate={block_success_rate:.1f} '
        f'goal_pixel_mae_mean={np.mean(goal_mae):.2f} '
        f'goal_pixel_mae_max={np.max(goal_mae):.2f} '
        f'n_goals={len(goal_log)}'
    )
    print(f'[goalshift] {summary}')
    old_goal, new_goal = goal_log[0]
    Image.fromarray(np.concatenate([old_goal, new_goal], axis=1)).save(
        results_path / 'goal_example_dataset_vs_shifted.png'
    )
    with (results_path.parent / 'goalshift_results.txt').open('a') as f:
        f.write(summary + '\n')

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
