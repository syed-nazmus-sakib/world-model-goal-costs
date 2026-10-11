"""Sanity check of goal_variants on Metaworld push: variant 0 == the eval's goal image, objects untouched."""
import os
import numpy as np
from PIL import Image
from evals.failure_attribution.goal_variants import hook_expert_capture, make_arm_variants, robot_free_frame
from evals.failure_attribution.run import build, load_args

W = os.environ["JEPAWM_LOGS"]
import sys
TASK = sys.argv[1] if len(sys.argv) > 1 else "push"
args_eval = load_args(os.path.join(os.path.dirname(W), "configs", f"mw_jepa-wm_{TASK}.yaml"))
cfg, agent, env, evaluator = build(args_eval)
hook_expert_capture(evaluator)
rows = []
for ep in range(3):
    ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
    env.reset(seed=ep_seed, task_idx=-1)
    u = env.proprio_env.unwrapped
    u._freeze_rand_vec, u.seeded_rand_vec = False, True
    env.seed(ep_seed)
    init_obs, goal_obs, _, ok = evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=-1)
    var = make_arm_variants(evaluator.expert_env, evaluator.expert_init_snap, evaluator.expert_final_snap, 4, seed=ep)
    g = goal_obs["visual"][-1].numpy().transpose(1, 2, 0)
    d0 = np.abs(var[0][1].astype(int) - g).max()
    from evals.failure_attribution.mw_sim import object_qpos
    from evals.failure_attribution.goal_variants import find_mw_wrapper
    from evals.failure_attribution.mw_sim import restore as _r
    mwu = find_mw_wrapper(evaluator.expert_env).unwrapped
    puck = []
    for v in var:
        _r(mwu, v[0]); puck.append(np.round(object_qpos(mwu)[:3], 4).tolist())
    hand = [np.round(v[2][:3], 3).tolist() for v in var]
    print("   rejections:", getattr(make_arm_variants, "reasons", None), "| max object drift of accepted variants:", round(getattr(make_arm_variants, "drift", 0.0), 4))
    print(f"ep {ep}: expert ok {ok} | variant0 vs eval goal max pixel diff {d0} | {len(var) - 1} arm variants")
    print("   puck:", puck)
    print("   hand:", hand)
    rf = robot_free_frame(evaluator.expert_env, var[0][0])
    rows.append(np.concatenate([g] + [v[1] for v in var[1:]] + [rf], 1))
Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(W, "failure_attribution", f"goal_variants_check_{TASK}.png"))
print("saved montage")
