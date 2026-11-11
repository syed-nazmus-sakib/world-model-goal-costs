"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def pix_shift(cfg):
    """exp-27: test-time appearance change, a warm tint of strength +pix_shift on every observed and goal frame; the
    graph keeps the latents of the unshifted training frames."""
    s = float(cfg.get('pix_shift', 0))
    if s <= 0:
        return []
    tint = torch.tensor([1.0, 1 - s / 2, 1 - s]).view(3, 1, 1)
    return [transforms.Lambda(lambda x: x * tint.to(x.device, x.dtype))]


def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            *pix_shift(cfg),
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


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    # RESERVE_GB=<n>: claim n GB of GPU memory at start and keep it in PyTorch's cache (freed tensors stay reserved for
    # this process), so other jobs on a shared GPU cannot take it while the model and graph load.  Fails at once if
    # the memory is not there, instead of minutes later in the middle of planning.
    if os.environ.get('RESERVE_GB') and torch.cuda.is_available():
        gb = float(os.environ['RESERVE_GB'])
        torch.empty(int(gb * 2**30), dtype=torch.uint8, device='cuda')  # freed at once, kept by the caching allocator
        print(f'[reserve] {gb:.0f} GB reserved: {torch.cuda.memory_reserved() / 2**30:.1f} GB held', flush=True)
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

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
    # stats_dataset=<name>: action / state normalisation from another dataset.  Needed when the evaluation file holds start
    # and goal frames only (official OGBench goals, ogb_tasks_data.py): the model was trained with the training data's statistics.
    stats_dataset = get_dataset(cfg, cfg.stats_dataset) if cfg.get('stats_dataset', None) else dataset
    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(
        dataset.get_col_data(col_name), return_index=True
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
    objective = None   # stays None for policy=random (no planner): the checks after the rollout must not fail on it

    if policy != 'random':
        if cfg.get('policy_object', None):  # a module saved with torch.save by our own training runs
            model = torch.load(cfg.policy_object, map_location='cpu', weights_only=False)
        else:
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
        if cfg.get('paa_nodes', None):  # PAA: aim CEM at graph anchors (code/lewm_explorations/paa/paa_cost.py)
            sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'lewm_explorations' / 'paa'))
            from paa_cost import PAAGoalCost

            objective = PAAGoalCost(cfg.paa_nodes, mode=cfg.get('paa_mode', 'fixed'), span=cfg.get('paa_span', 4),
                                    max_hops=cfg.get('paa_max_hops', 12), head=cfg.get('paa_head', None),
                                    tau=cfg.get('paa_tau', 0.5), pess=cfg.get('paa_pess', 1.0),
                                    agg=cfg.get('metric_agg', 'min'), ap_offset=cfg.get('paa_ap_offset', cfg.eval.goal_offset_steps),
                                    ap_step=cfg.plan_config.action_block * cfg.plan_config.receding_horizon,
                                    ap_k=cfg.get('ap_k', 8), gap_cal=cfg.get('gap_cal', False),
                                    knn=cfg.get('paa_knn', 10), thr_q=cfg.get('paa_thr_q', 0.5), frac=cfg.get('paa_frac', 1.0),
                                    switch=cfg.get('paa_switch', 1.0),
                                    handoff=cfg.get('paa_handoff', False),
                                    handoff_hops=cfg.get('paa_handoff_hops', 0),
                                    pool=cfg.get('paa_pool', False),
                                    drop_box=([cfg.paa_drop_x0, cfg.paa_drop_x1, cfg.paa_drop_y0, cfg.paa_drop_y1]
                                              if cfg.get('paa_drop_x0', None) is not None else None))
        elif cfg.get('ngc_nodes', None):  # NGC baseline (arXiv 2610.25013, concurrent work): graph-distance cost
            sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'lewm_explorations' / 'paa'))
            from ngc_cost import NGCObjective

            objective = NGCObjective(cfg.ngc_nodes, n_nodes=cfg.get('ngc_n', 5000), neighbors=cfg.get('ngc_k', 15),
                                     anchors=cfg.get('ngc_anchors', 4), seed=cfg.get('ngc_seed', 0),
                                     target=cfg.get('ngc_target', 'goal'), span=cfg.get('paa_span', 6),
                                     agg=cfg.get('metric_agg', 'last'), frac=cfg.get('paa_frac', 1.0))
        elif cfg.get('metric_head', None):  # method C: learned task-aligned goal cost (scripts/plan/metric_cost.py)
            from metric_cost import MetricGoalCost

            objective = MetricGoalCost(cfg.metric_head, cfg.get('metric_mix', 0.0), cfg.get('metric_agg', 'last'))
        cost = swm.planning.ShootingCostEvaluator(model, objective)
        solver = hydra.utils.instantiate(cfg.solver, cost=cost)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
        if cfg.get('ff', False):  # learned goal-conditioned policy (GCBC / GCIVL / GCIQL / HILP): same pairs and scoring
            policy = swm.policy.FeedForwardPolicy(model=model, process=process, transform=transform)
            if cfg.get('graph_waypoints', False):  # graph + policy baseline: waypoints from the latent graph
                sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'lewm_explorations' / 'paa'))
                from paa_cost import PAAGoalCost
                from graph_policy import GraphWaypointPolicy

                enc = swm.wm.utils.load_pretrained(cfg.graph_wm).to('cuda').eval().requires_grad_(False)
                graph = PAAGoalCost(cfg.graph_nodes, mode='fixed', span=cfg.get('paa_span', 6),
                                    max_hops=cfg.get('paa_max_hops', 12), knn=cfg.get('paa_knn', 10))
                policy = GraphWaypointPolicy(policy, enc, graph, dataset, transform['pixels'],
                                             every=cfg.get('waypoint_every', 5), img_dtype=img_dtype,
                                             hold=cfg.get('waypoint_hold', 0))
                print(f'[graph_waypoints] {cfg.graph_wm} graph, span {graph.span}, waypoint every '
                      f'{policy.every} steps', flush=True)
            if cfg.get('value_graph', False):  # exp-24 / exp-25: graph and subgoals from the policy's value head, no world model
                sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'lewm_explorations' / 'paa'))
                from value_graph import ValueGraph, ValueGraphPolicy

                vg = ValueGraph(model, cfg.graph_nodes, dataset, policy._prepare_info, mode=cfg.get('vg_mode', 'value'),
                                tau=cfg.get('vg_tau', 4.0), n_nodes=cfg.get('vg_nodes', 2000), cand=cfg.get('vg_cand', 50),
                                gamma=cfg.get('vg_gamma', 0.99))
                policy = ValueGraphPolicy(policy, vg, dataset, every=cfg.get('waypoint_every', 5), replan=cfg.get('vg_replan', 0.0))

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )
    if cfg.get('video_dir', None):  # one folder per run (parallel runs of the same policy would overwrite env_*.mp4)
        results_path = Path(cfg.video_dir)

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
    if cfg.get('exclude_eps', None):  # PAA harvest: never sample from the evaluation pairs' episodes
        ex = np.load(cfg.exclude_eps)
        valid_indices = valid_indices[~np.isin(dataset.get_col_data(col_name)[valid_indices], ex)]
        print(f'excluded {len(ex)} episodes', flush=True)
    print(len(valid_indices), 'valid starting points found for evaluation.')

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

    goal_eps = goal_steps = None
    if cfg.get('cross_goal', False):  # stitching test: each goal is a random state of a different episode
        g2 = np.random.default_rng(cfg.seed + 1000)
        all_eps = dataset.get_col_data(col_name)
        all_steps = dataset.get_col_data('step_idx')
        goal_eps, goal_steps = [], []
        for e in eval_episodes:
            while True:
                j = int(g2.integers(len(all_eps)))
                if all_eps[j] != e:
                    break
            goal_eps.append(int(all_eps[j])); goal_steps.append(int(all_steps[j]))
        goal_eps, goal_steps = np.asarray(goal_eps), np.asarray(goal_steps)
        world.cross_goal = (goal_eps, goal_steps)
        print(f'[cross_goal] goals from other episodes, budget {cfg.eval.eval_budget}', flush=True)
    if cfg.get('pairs_npz', None):  # record the sampled start-goal pairs (PAA: exclusion lists, paired analysis)
        np.savez(cfg.pairs_npz, ep=np.asarray(eval_episodes), start=np.asarray(eval_start_idx),
                 offset=cfg.eval.goal_offset_steps, seed=cfg.seed,
                 **({} if goal_eps is None else {'goal_ep': goal_eps, 'goal_step': goal_steps}))
        if cfg.get('pairs_only', False):
            print('[pairs] saved', cfg.pairs_npz, flush=True)
            return
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    # PAA: drop the evaluation pairs' episodes (paa_noexclude: the graph comes from another dataset, e.g. random-play data)
    if hasattr(policy, 'solver') and hasattr(objective, 'set_exclude') and not cfg.get('paa_noexclude', False):
        objective.set_exclude(np.unique(eval_episodes if goal_eps is None else np.concatenate([eval_episodes, goal_eps])))
    if hasattr(policy, 'graph') and not cfg.get('paa_noexclude', False):  # graph-waypoint policy: same exclusion
        policy.graph.set_exclude(np.unique(eval_episodes if goal_eps is None else np.concatenate([eval_episodes, goal_eps])))
    _dec_t = []  # wall time per policy decision (ms), CUDA-synchronised; printed in the summary line
    _get_action = policy.get_action
    # Live progress, one line every PROGRESS_SEC seconds (default 60; 0 = off): step, pairs solved so far, time per step and
    # an ETA.  The ETA assumes every step until the budget; the run ends earlier once all pairs are solved.
    _prog = {'t0': time.perf_counter(), 'last': 0.0, 'n': 0, 'solved': None,
             'every': float(os.environ.get('PROGRESS_SEC', '60')), 'budget': int(cfg.eval.eval_budget)}

    def _fmt_t(sec):
        sec = int(sec); return f'{sec // 3600}h{sec % 3600 // 60:02d}m' if sec >= 3600 else f'{sec // 60}m{sec % 60:02d}s'

    def _progress():
        _prog['n'] += 1
        term = getattr(world, 'terminateds', None)
        if term is not None:
            _prog['solved'] = np.asarray(term, bool) | (_prog['solved'] if _prog['solved'] is not None else False)
        now = time.perf_counter()
        if _prog['every'] <= 0 or now - _prog['last'] < _prog['every']:
            return
        _prog['last'] = now
        n, b, el = _prog['n'], _prog['budget'], now - _prog['t0']
        solved = 0 if _prog['solved'] is None else int(np.sum(_prog['solved']))
        per = el / n
        mem = f' | GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak' if torch.cuda.is_available() else ''
        print(f'[progress] step {n}/{b} ({100 * n / b:.0f}%) | solved {solved}/{world.num_envs} | {per:.1f} s/step | '
              f'elapsed {_fmt_t(el)} | ETA <= {_fmt_t(per * max(b - n, 0))}{mem}', flush=True)

    def _timed_get_action(*a, **k):
        _progress()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _get_action(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _dec_t.append(1e3 * (time.perf_counter() - t0))
        return out

    policy.get_action = _timed_get_action
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

    trace = None
    if cfg.get('trace_npz', None):  # diagnosis: every `trace_every` steps, small numeric infos + latent distance to goal
        trace, every = {'t': []}, int(cfg.get('trace_every', 5))
        tstate = {'n': 0, 'zg': None}

        def _enc(arr):
            arr = np.asarray(arr)
            arr = arr[:, -1] if arr.ndim == 5 else arr
            with torch.no_grad():
                im = torch.stack([transform['pixels'](np.ascontiguousarray(x)) for x in arr]).to('cuda', img_dtype)
                return model.encode({'pixels': im[:, None]})['emb'][:, 0].float()

        def trace_fn(w):
            n = tstate['n']; tstate['n'] += 1
            if n % every:
                return
            if tstate['zg'] is None:
                tstate['zg'] = _enc(w.infos['goal'])
            trace['t'].append(n)
            trace.setdefault('lat_dist', []).append(((_enc(w.infos['pixels']) - tstate['zg']) ** 2).sum(-1).cpu().numpy())
            trace.setdefault('done', []).append(np.asarray(w.terminateds).copy())
            for k, v in w.infos.items():
                if k in ('pixels', 'goal') or not isinstance(v, np.ndarray) or v.dtype.kind not in 'fiub':
                    continue
                last = v[:, -1] if v.ndim >= 2 and v.shape[0] == w.num_envs and v.shape[1] > 0 else v
                if last.shape[0] == w.num_envs and last[0].size <= 64:
                    trace.setdefault(k, []).append(np.asarray(last, dtype=np.float32).copy())

        world.trace_fn = trace_fn

    if cfg.get('kidnap_at', None):  # stress test: after kidnap_at env steps, every unsolved env is put back at its start
        # state (the first callable of the config, e.g. set_state(state=proprio)); budget and goal are unchanged.  A
        # planner that keeps an elapsed-step clock (AP) now underestimates the remaining distance; replanning from the
        # current state does not.  The frame the planner sees is refreshed by the next environment step.
        K = int(cfg.kidnap_at)
        spec = OmegaConf.to_container(cfg.eval.callables, resolve=True)[0]
        start_rows = [dataset.load_chunk(np.array([e]), np.array([s]), np.array([s + 1]))[0]
                      for e, s in zip(eval_episodes, eval_start_idx)]

        def _v(row, col):
            v = row[col]
            v = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
            return np.asarray(v)[-1]

        kst = {'n': 0, 'moved': 0, 'solved': None}
        prev_fn = getattr(world, 'trace_fn', None)

        def kidnap_fn(w):
            if prev_fn is not None:
                prev_fn(w)
            kst['n'] += 1
            term = np.asarray(w.terminateds, bool).ravel()
            kst['solved'] = term if kst['solved'] is None else (kst['solved'] | term)  # success is sticky
            if kst['n'] == K:
                done = kst['solved']
                for i in range(w.num_envs):
                    if done[i]:
                        continue
                    args = {a: _v(start_rows[i], d['value']) for a, d in spec.get('args', {}).items()
                            if d.get('value') in start_rows[i]}
                    getattr(w.envs.envs[i].unwrapped, spec['method'])(**args)
                    kst['moved'] += 1
                print(f'[kidnap] step {K}: {kst["moved"]} unsolved envs put back at their start state', flush=True)

        world.trace_fn = kidnap_fn

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
    if trace is not None:
        alog = np.asarray(getattr(objective, 'anchor_log', []), dtype=np.int64).reshape(-1, 3)
        wlog = np.asarray(getattr(policy, 'log', []), dtype=np.int64).reshape(-1, 3)  # graph-waypoint policy
        np.savez(cfg.trace_npz, success=np.asarray(metrics['episode_successes']), ep=np.asarray(eval_episodes), waypoint_log=wlog,
                 start=np.asarray(eval_start_idx), offset=cfg.eval.goal_offset_steps, anchor_log=alog,
                 **{f'tr_{k}': np.stack(v) for k, v in trace.items() if len(v)})
        print(f'[trace] {cfg.trace_npz} keys={sorted(trace)}', flush=True)
    if cfg.get('dump_npz', None):  # PAA harvest: final-frame latents for label-free attainment labels
        px = np.asarray(world.infos['pixels'])
        px = px[:, -1] if px.ndim == 5 else px
        with torch.no_grad():
            imgs = torch.stack([transform['pixels'](np.ascontiguousarray(x)) for x in px]).to('cuda', img_dtype)
            final_emb = model.encode({'pixels': imgs[:, None]})['emb'][:, 0].float().cpu().numpy()
        np.savez(cfg.dump_npz, final_emb=final_emb, success=np.asarray(metrics['episode_successes']),
                 ep=np.asarray(eval_episodes), start=np.asarray(eval_start_idx),
                 offset=cfg.eval.goal_offset_steps, budget=cfg.eval.eval_budget)
        print(f'[dump] {cfg.dump_npz}', flush=True)
    print(f'[eval] videos saved to {results_path.resolve()}')
    if hasattr(policy, 'stats'):  # graph-waypoint policy: how often the goal was replaced and waypoints were reached
        print('[waypoints] ' + ' '.join(f'{k}={v:.3f}' for k, v in policy.stats().items()), flush=True)
    if hasattr(objective, 'timing'):  # graph build / shortest-path time
        print('[graph_timing] ' + ' '.join(f'{k}={v:.3f}' for k, v in objective.timing().items()), flush=True)
    eps_bits = ''.join('1' if bool(x) else '0' for x in np.asarray(metrics['episode_successes']).ravel())
    ms = np.asarray(_dec_t[1:] if len(_dec_t) > 1 else _dec_t, dtype=np.float64)  # first decision includes warm-up
    print(  # one-line summary for the method C queues
        f'[summary] env={cfg.world.env_name} seed={cfg.seed} goal_offset={cfg.eval.goal_offset_steps} '
        f'budget={cfg.eval.eval_budget} horizon={cfg.plan_config.horizon} '
        f'metric={Path(str(cfg.get("metric_head", "none"))).stem} agg={cfg.get("metric_agg", "last")} '
        f'paa={cfg.get("paa_mode", "none") if cfg.get("paa_nodes", None) else "none"}:{cfg.get("paa_span", "-")}:{Path(str(cfg.get("paa_head", "none"))).stem} '
        f'success_rate={metrics["success_rate"]:.1f} n={cfg.eval.num_eval} '
        f'ms_per_decision={float(ms.mean()) if len(ms) else float("nan"):.1f} decisions={len(_dec_t)} eps={eps_bits}'
    )

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
