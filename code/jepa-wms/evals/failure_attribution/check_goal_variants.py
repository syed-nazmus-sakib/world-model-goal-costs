"""Check goal-image fidelity and object drift, and save an arm-variant montage."""

import os
from pathlib import Path


def goal_check_paths(task):
    root = Path(os.environ.get("JEPAWM_HOME", Path(__file__).resolve().parents[4]))
    logs = Path(os.environ.get("JEPAWM_LOGS", root / "artifacts" / "raw_logs"))
    return root / "configs" / f"mw_jepa-wm_{task}.yaml", logs / "failure_attribution" / f"goal_variants_check_{task}.png"


def main():
    import argparse

    import numpy as np
    from PIL import Image

    from evals.failure_attribution.goal_variants import (
        find_mw_wrapper, hook_expert_capture, make_arm_variants, robot_free_frame,
    )
    from evals.failure_attribution.mw_sim import object_qpos, restore
    from evals.failure_attribution.run import build, load_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="push")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    config, output = goal_check_paths(args.task)
    if not config.is_file():
        parser.error(f"Config not found: {config}")
    cfg, agent, env, evaluator = build(load_args(config))
    hook_expert_capture(evaluator)
    rows = []
    for ep in range(args.episodes):
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        env.reset(seed=ep_seed, task_idx=-1)
        u = env.proprio_env.unwrapped
        u._freeze_rand_vec, u.seeded_rand_vec = False, True
        env.seed(ep_seed)
        _, goal_obs, _, ok = evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=-1)
        try:
            var = make_arm_variants(evaluator.expert_env, evaluator.expert_init_snap,
                                    evaluator.expert_final_snap, 4, seed=ep)
            goal = goal_obs["visual"][-1].numpy().transpose(1, 2, 0)
            diff = np.abs(var[0][1].astype(int) - goal).max()
            mwu = find_mw_wrapper(evaluator.expert_env).unwrapped
            objects = []
            for variant in var:
                restore(mwu, variant[0])
                objects.append(np.round(object_qpos(mwu)[:3], 4).tolist())
            hands = [np.round(variant[2][:3], 3).tolist() for variant in var]
            print("rejections:", getattr(make_arm_variants, "reasons", None),
                  "| max object drift:", round(getattr(make_arm_variants, "drift", 0.0), 4))
            print(f"ep {ep}: expert ok {ok} | variant0 max pixel diff {diff} | {len(var) - 1} arm variants")
            print("objects:", objects, "hands:", hands)
            hidden = robot_free_frame(evaluator.expert_env, var[0][0])
            frames = [goal] + [variant[1] for variant in var[1:]]
            frames += [np.zeros_like(goal)] * (5 - len(frames))
            rows.append(np.concatenate(frames + [hidden], axis=1))
        finally:
            evaluator.expert_env.close()
            evaluator.expert_env = None
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(output)
    env.close()
    print(f"saved montage: {output}")


if __name__ == "__main__":
    main()
