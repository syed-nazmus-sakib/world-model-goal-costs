# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Method C on Metaworld (2026-10-22), step 1: frozen JEPA-WM encodings of offline expert episodes, to train a learned
goal cost (mw_head_train.py).

For each task, `--episodes` episodes of the released Metaworld dataset (the world model's own training data) are
decoded; every `--stride`-th frame and the last frame are encoded with the eval's `model.encode` (visual patches and
proprio embedding, exactly what the planning cost sees). A frame-format check compares the eval's simulator start
frame with the dataset's first frames, as stored and flipped.

Saves vis (N, P, D) float16, prop (N, Dp) float16, ep (N,) episode id, t (N,) step, task (N,) task index, tasks.

Usage (from code/jepa-wms):
    python -m evals.failure_attribution.mw_head_data --fname <mw eval config> --tasks mw-drawer-open,mw-push --out <npz>
"""

import argparse
import glob
import io
import os

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from tensordict import TensorDict

from evals.failure_attribution.run import build, load_args


def frames_of(video_bytes):
    with av.open(io.BytesIO(video_bytes)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


@torch.no_grad()
def encode(model, frames, proprio, batch_size=128):
    vis, prop = [], []
    for s in range(0, len(frames), batch_size):
        td = TensorDict(
            {
                "visual": torch.from_numpy(frames[s : s + batch_size]).permute(0, 3, 1, 2).unsqueeze(1),
                "proprio": torch.from_numpy(proprio[s : s + batch_size]).unsqueeze(1),
            },
            batch_size=[len(frames[s : s + batch_size])],
        )
        e = model.encode(td.to("cuda:0"))
        vis.append(e["visual"][:, 0].flatten(1, -2).half().cpu())  # (b, P, D)
        prop.append(e["proprio"][:, 0].flatten(1).half().cpu())  # (b, Dp)
    return torch.cat(vis).numpy(), torch.cat(prop).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--tasks", required=True, help="comma list, dataset names (mw-drawer-open, ...)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=100, help="per task")
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    tasks = args.tasks.split(",")

    args_eval = load_args(args.fname)
    cfg, agent, env, evaluator = build(args_eval)
    model = agent.model
    dset_dir = os.path.join(os.environ.get("JEPAWM_DSET", ""), "Metaworld", "data")
    files = sorted(glob.glob(os.path.join(dset_dir, "*.parquet")))

    # frame-format check on the config's task: simulator start frame vs dataset first frames
    cfg_task, sim0 = cfg.task_specification.task, None
    try:
        ep_seed = (cfg.local_seed * cfg.local_seed) % (2**32 - 2)
        env.reset(seed=ep_seed, task_idx=-1)
        init_obs, _, _, _ = evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=-1)
        sim0 = init_obs["visual"].reshape(-1, *init_obs["visual"].shape[-3:])[-1].permute(1, 2, 0).numpy().astype(np.float32)
    except Exception as e:  # the check is informative only
        print(f"[format check] skipped: {e!r}", flush=True)

    V, P, EP, T, K = [], [], [], [], []
    ep_id, firsts, count = 0, [], {t: 0 for t in tasks}
    for f in files:
        if all(c >= args.episodes for c in count.values()):
            break
        names = pq.read_table(f, columns=["task"]).column("task").to_pylist()
        rows = [i for i, t in enumerate(names) if t in count and count[t] < args.episodes]
        if not rows:
            continue
        tab = pq.read_table(f, columns=["task", "video", "states"])
        for i in rows:
            r = tab.slice(i, 1).to_pylist()[0]
            task = r["task"]
            if count[task] >= args.episodes:
                continue
            fr = frames_of(r["video"]["bytes"])
            st = np.asarray(r["states"], dtype=np.float32)
            L = min(len(fr), len(st))
            keep = np.unique(np.r_[np.arange(0, L, args.stride), L - 1])
            if task == cfg_task and len(firsts) < 50:
                firsts.append(fr[0].astype(np.float32))
            v, p = encode(model, np.ascontiguousarray(fr[keep]), np.ascontiguousarray(st[keep, :4]))
            V.append(v), P.append(p)
            EP.append(np.full(len(keep), ep_id)), T.append(keep), K.append(np.full(len(keep), tasks.index(task)))
            ep_id += 1
            count[task] += 1
        print(f"{os.path.basename(f)}: {count}", flush=True)
    if firsts and sim0 is not None:
        fs = np.stack(firsts)
        mae = np.abs(fs - sim0[None]).mean((1, 2, 3)).min()
        mae_flip = np.abs(fs[:, ::-1] - sim0[None]).mean((1, 2, 3)).min()
        print(f"[format check] {cfg_task}: min MAE sim start vs dataset first frames {mae:.1f}, flipped {mae_flip:.1f}", flush=True)
    np.savez(
        args.out, vis=np.concatenate(V), prop=np.concatenate(P), ep=np.concatenate(EP), t=np.concatenate(T),
        task=np.concatenate(K), tasks=np.array(tasks),
    )
    print("saved", args.out, np.concatenate(V).shape, flush=True)


if __name__ == "__main__":
    main()
