from pathlib import Path

import hydra
import logging
import numpy as np

import stable_worldmodel as swm
# MeanTrap: importing trap_env registers swm/TwoRoomTrap-v0
import stable_worldmodel.envs.two_room.trap_env  # noqa: F401
from stable_worldmodel.envs.two_room import ExpertPolicy

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    """Run data collection script.

    MeanTrap copy of collect_tworooms.py: same expert, noise, world settings
    and number of trajectories, but in swm/TwoRoomTrap-v0. Lines changed for
    MeanTrap are marked. Usage:
        python scripts/data/collect_tworoom_trap.py +fall_prob=0.5
    """

    # MeanTrap: fall_prob must be given explicitly (+fall_prob=0.5 or 0.0)
    if cfg.get('fall_prob') is None:
        raise ValueError('set +fall_prob=<value>, e.g. +fall_prob=0.5')
    fall_prob = float(cfg.fall_prob)

    world = swm.World(
        'swm/TwoRoomTrap-v0',  # MeanTrap
        **cfg.world,
        render_mode='rgb_array',
        fall_prob=fall_prob,  # MeanTrap
    )
    world.set_policy(ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05))

    options = cfg.get('options')
    rng = np.random.default_rng(cfg.seed)

    world.collect(
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / f'tworoom_trap_p{fall_prob}.lance',  # MeanTrap
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    logger.info(' 🎉🎉🎉 Completed data collection for tworoom 🎉🎉🎉')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()
