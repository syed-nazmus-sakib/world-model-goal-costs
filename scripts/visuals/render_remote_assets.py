"""Render existing Push-T plans and compute real DINOv2 patch discrepancies.

Runs in the existing RunPod conda environment. No planning or training is run.
Inputs/outputs live in a separate visual_story/source_data directory.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    out = Path(args.data)
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

    d = np.load(out / "pusht_episode.npz")
    mean = np.array([-0.0087, 0.0068], dtype=np.float32)
    std = np.array([0.2019, 0.2002], dtype=np.float32)
    arrays, checks = {}, {}
    for mode in ["wm", "sim_latent", "sim_true"]:
        env = PushTEnv(with_velocity=True, with_target=True, render_size=224)
        env.shape = str(d["shape"].item())
        env.seed(int(d["ep_seed"]))
        env.reset_to_state = d["init_state"].copy()
        env.reset()
        # Render the evaluation target instead of the env's decorative default
        # target. This changes only the reward/target overlay, not the physics.
        env.goal_pose = d["goal_state"][2:5].copy()
        frames = [env._render_frame("rgb_array").copy()]
        states = [env._get_obs().copy()]
        actions = d[f"{mode}_plan"].reshape(-1, 2) * std + mean
        for action in actions:
            env.step(action)
            frames.append(env._render_frame("rgb_array").copy())
            states.append(env._get_obs().copy())
        err = float(np.max(np.abs(states[-1].astype(float) - d[f"{mode}_final_state"])))
        checks[mode] = {"max_abs_final_state_error": err, "recorded_success": bool(d[f"{mode}_success"])}
        if err > 0.01:
            raise RuntimeError(f"Replay differs from saved result for {mode}: {err}")
        arrays[f"{mode}_frames"] = np.array(frames)
        arrays[f"{mode}_states"] = np.array(states)
        env.close()
    env = PushTEnv(with_velocity=True, with_target=True, render_size=224)
    env.shape = str(d["shape"].item())
    env.reset_to_state = d["goal_state"].copy()
    env.reset()
    env.goal_pose = d["goal_state"][2:5].copy()
    arrays["goal_frame"] = env._render_frame("rgb_array").copy()
    env.close()
    np.savez_compressed(out / "pusht_replay.npz", **arrays)
    (out / "replay_validation.json").write_text(json.dumps(checks, indent=2))
    print("Push-T replay validated:", checks, flush=True)

    import torch
    torch.set_num_threads(2)
    model = torch.hub.load("/root/.cache/torch/hub/facebookresearch_dinov2_main", "dinov2_vits14", source="local", pretrained=True).eval()
    for task in ["drawer", "push"]:
        z = np.load(out / f"{task}_offline_episode.npz")
        frames = z["goal_frames"]
        inp = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255
        inp = (inp - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.inference_mode():
            features = model.forward_features(inp)["x_norm_patchtokens"].cpu().numpy()
        # A fixed real rendered scene (variant 2), compared against two goals.
        patch = ((features[[0, 3]] - features[2][None]) ** 2).mean(-1).reshape(2, 16, 16)
        np.savez_compressed(out / f"{task}_dino_maps.npz", frames=frames, features=features, patch_mse=patch, reference_variant=2, goal_variants=[0, 3])
        print(task, "DINOv2 patch MSE", patch.mean((1, 2)).tolist(), flush=True)


if __name__ == "__main__":
    main()
