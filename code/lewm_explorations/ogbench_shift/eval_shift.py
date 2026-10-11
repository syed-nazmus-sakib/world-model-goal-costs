"""Evaluate a trained agent from the dual-goal-representations repo (Park et al. codebase:
GCIVL / CRL / GCFBC / dual / TRA / VIP ...) on OGBench-Shift.

Run from INSIDE the dual-goal-representations repo root, with this folder on PYTHONPATH:
  PYTHONPATH=/path/to/ogbench_shift MUJOCO_GL=egl python /path/to/ogbench_shift/eval_shift.py \
      --env_name=visual-cube-single-play-v0 --agent=agents/gcivl/pixel/dual.py \
      --restore_path=exp/goal_representation/Debug/<run> --restore_epoch=500000 \
      --out=shift_results.csv  [same --agent.* overrides you trained with]
Writes one CSV row per (shift, severity, task) with success rate.
"""
import csv, os, random
from collections import defaultdict

import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import FrameStackWrapper
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
import ogbench
from shift import FAMILIES, ShiftedGoalEnv

try:
    from utils.datasets import VIPDataset
except ImportError:
    VIPDataset = None

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'visual-cube-single-play-v0', 'OGBench dataset name.')
flags.DEFINE_string('restore_path', None, 'Run directory with params_<epoch>.pkl.')
flags.DEFINE_integer('restore_epoch', None, 'Checkpoint epoch.')
flags.DEFINE_integer('seed', 0, 'Seed.')
flags.DEFINE_integer('eval_episodes', 20, 'Episodes per task per condition.')
flags.DEFINE_list('shifts', ['none'] + FAMILIES, 'Shift families.')
flags.DEFINE_list('severities', ['1', '2', '3'], 'Severities.')
flags.DEFINE_string('out', 'shift_results.csv', 'Output CSV.')
config_flags.DEFINE_config_file('agent', 'agents/gcivl/original.py', lock_config=False)


def main(_):
    config = FLAGS.agent
    random.seed(FLAGS.seed); np.random.seed(FLAGS.seed)
    raw_env, train_ds, _ = ogbench.make_env_and_datasets(FLAGS.env_name, compact_dataset=True)
    dataset_class = {'GCDataset': GCDataset, 'HGCDataset': HGCDataset, 'VIPDataset': VIPDataset}[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(norm=config['norm'], **train_ds), config)
    diff = train_dataset.get_diff()
    ex = train_dataset.sample(1)
    agent = agents[config['agent_name']].create(FLAGS.seed, ex['observations'], ex['actions'], config,
                                                ex_goals=ex['value_goals'] if config['oraclerep'] else None)
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    agent = jax.device_put(agent, device=jax.devices('cpu')[0])

    shifted = ShiftedGoalEnv(raw_env, shift='none', seed=FLAGS.seed)
    env = FrameStackWrapper(shifted, config['frame_stack']) if config['frame_stack'] else shifted
    task_infos = raw_env.unwrapped.task_infos
    new = not os.path.exists(FLAGS.out)
    with open(FLAGS.out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['env', 'agent', 'restore_path', 'shift', 'severity', 'task_id', 'task', 'success'])
        for s in FLAGS.shifts:
            for sev in ([1] if s == 'none' else [int(x) for x in FLAGS.severities]):
                shifted.shift, shifted.sev = s, sev
                for tid in range(1, len(task_infos) + 1):
                    info, _, _ = evaluate(agent=agent, env=env, task_id=tid, config=config,
                                          num_eval_episodes=FLAGS.eval_episodes, num_video_episodes=0,
                                          diff=diff)
                    w.writerow([FLAGS.env_name, config['agent_name'], FLAGS.restore_path, s, sev, tid,
                                task_infos[tid - 1]['task_name'], info['success']])
                    f.flush()
                    print(f'{s}@{sev} task{tid}: success={info["success"]:.2f}', flush=True)


if __name__ == '__main__':
    app.run(main)
