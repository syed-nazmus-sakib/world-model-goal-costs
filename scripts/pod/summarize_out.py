#!/usr/bin/env python3
"""One table from a pod's /root/out backup: every closed-loop run folder with a summary.txt (Metaworld, 12 episodes) and every
LeWM goalshift / eval_wm log line. Usage: python scripts/pod/summarize_out.py <backup dir> [--md]"""
import argparse
import glob
import json
import os
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--md", action="store_true", help="markdown table")
    args = ap.parse_args()
    rows = []
    for summ in sorted(glob.glob(os.path.join(args.root, "**", "summary.txt"), recursive=True)):
        d = os.path.dirname(summ)
        rel = os.path.relpath(d, args.root)
        for line in open(summ):
            m = re.match(r"\s+(sim_\w+|wm)\s+(\d+)/(\d+)\s*=", line)
            if m:
                eps = sorted(glob.glob(os.path.join(d, "ep_*.npz")))
                succ = ""
                try:
                    succ = "".join("S" if bool(np.load(f, allow_pickle=True)[f"{m[1]}_success"]) else "." for f in eps)
                except Exception:
                    pass
                rows.append((rel, m[1], int(m[2]), int(m[3]), succ))
    for log in sorted(glob.glob(os.path.join(args.root, "**", "*.log"), recursive=True)):
        try:
            txt = open(log, errors="ignore").read()
        except Exception:
            continue
        for m in re.finditer(r"\[(goalshift|summary)\].*?success_rate=([0-9.]+)", txt):
            rows.append((os.path.relpath(log, args.root), m[1], float(m[2]), 100, ""))
    if args.md:
        print("| run | planner | success | episodes |\n|---|---|---|---|")
        for rel, pl, k, n, succ in rows:
            print(f"| `{rel}` | {pl} | {k}/{n} | `{succ}` |" if succ else f"| `{rel}` | {pl} | {k}% | |")
    else:
        for rel, pl, k, n, succ in rows:
            print(f"{rel:60s} {pl:18s} {k}/{n}  {succ}")
    json.dump(rows, open(os.path.join(args.root, "summary_rows.json"), "w"))


if __name__ == "__main__":
    main()
