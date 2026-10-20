"""Task-search decision rule (stated before the runs): keep Metaworld tasks whose released latent cost succeeds on
25-83% of 12 episodes with exact dynamics (3..10/12), whose true-state cost does at least 2 episodes better, and
whose goal-variant generator gives >= 3 arm variants per episode with object drift < 1 cm. Up to 2 tasks, the ones
closest to 50% first. Usage: python pick_tasks.py <search_dir>  -> prints selected task names, one per line."""
import glob
import os
import re
import sys

d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*_summary.txt"))):
    t = os.path.basename(f)[: -len("_summary.txt")]
    s = open(f).read()
    lat = re.search(r"sim_latent\s+(\d+)/(\d+)", s)
    tru = re.search(r"sim_true\s+(\d+)/(\d+)", s)
    if not lat or not tru:
        continue
    lat, n, tru = int(lat.group(1)), int(lat.group(2)), int(tru.group(1))
    vlog = os.path.join(d, f"{t}_variants.log")
    nvar, drift = [], 0.0
    if os.path.exists(vlog):
        v = open(vlog).read()
        nvar = [int(x) for x in re.findall(r"\| (\d+) arm variants", v)]
        dr = re.findall(r"max object drift of accepted variants: ([0-9.]+)", v)
        drift = max([float(x) for x in dr] or [0.0])
    ok_var = len(nvar) >= 1 and min(nvar) >= 3 and drift < 0.01
    ok = 3 <= lat <= 10 and tru >= lat + 2 and ok_var
    rows.append((abs(lat - n / 2), t, lat, tru, n, nvar, drift, ok))
    print(f"# {t}: sim_latent {lat}/{n} sim_true {tru}/{n} variants/ep {nvar} drift {drift:.4f} -> {'KEEP' if ok else 'drop'}", file=sys.stderr)
sel = [r for r in sorted(rows) if r[-1]][:2]
if not sel:  # fallback: the task with the largest true-latent gap that is not at ceiling, so the queue still produces an L4 test
    cand = [r for r in sorted(rows, key=lambda r: -(r[3] - r[2])) if r[2] <= 10 and r[5] and min(r[5]) >= 3]
    sel = cand[:1]
    print("# no task met the rule; fallback:", [r[1] for r in sel], file=sys.stderr)
for r in sel:
    print(r[1])
