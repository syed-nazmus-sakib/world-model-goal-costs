"""Merge the planner's visited frames (run_closed --dump_dir, kind 5) with the expert and goal frames of the same pilot-data
episodes, so that mw_pilot_heads.load can label them (it takes the object's start, the start distance and the expert's travel
from the kind-0 frames of each episode file).

    python -m evals.failure_attribution.mw_merge_planner_frames --pilot <train dir> --dump <dump dir> --out <out dir>

The pilot and dump episodes must come from the same seeds (pilot --ep_start 100, run_closed --ep_start 100). Only episodes
present in both are written; the expert replay is checked against the dump's start state (hand within 1 cm).
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n_out, n_frames = 0, 0
    for f in sorted(glob.glob(os.path.join(args.dump, "ep_*.npz"))):
        name = os.path.basename(f)
        pf = os.path.join(args.pilot, name)
        if not os.path.exists(pf):
            print(f"{name}: no pilot episode, skipped")
            continue
        d, p = np.load(f), np.load(pf)
        keep = np.isin(p["kind"], (0, 1))
        ex0 = np.where((p["kind"] == 0) & (p["t"] == p["t"][p["kind"] == 0].min()))[0][0]
        drift = float(np.linalg.norm(p["state"][ex0, :3] - d["state"][0, :3]))
        if drift > 0.01:
            print(f"{name}: start hand differs by {drift:.3f} m between pilot and dump, skipped")
            continue
        out = {k: np.concatenate([p[k][keep], d[k]]) for k in ("vis", "prop", "state", "kind", "t", "var", "start")}
        for k in ("starts", "contact", "replay_err", "goal_path_diff"):
            if k in p.files:
                out[k] = p[k]
        np.savez(os.path.join(args.out, name), **out)
        n_out += 1
        n_frames += int(len(d["kind"]))
        print(f"{name}: {int(keep.sum())} expert/goal + {len(d['kind'])} planner frames, planner success {bool(d['success'])}")
    print(f"wrote {n_out} episodes, {n_frames} planner frames")


if __name__ == "__main__":
    main()
