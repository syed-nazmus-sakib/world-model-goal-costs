"""Table of the official-goal runs (scripts/exp/exp31_ogb_goals.sh): success per task and the mean over tasks, per method.

Reads summary.txt (lines `<method>_t<task>_s<seed> [summary] ... success_rate=<x> n=<episodes>`).  A cell is the mean over the
finished seeds; `mean` is the OGBench score: the average over the tasks (given only when every task has a run).
usage: python ogb_table.py <summary.txt>
"""
import re
import sys
from collections import defaultdict

ORDER = ['random', 'released', 'relmin', 'ga', 'gcbc', 'gcivl', 'gciql', 'hilp', 'pldm', 'pldmga', 'dwm']
NAME = {'random': 'random actions', 'released': 'LeWM', 'relmin': 'LeWM + min-over-plan', 'ga': 'Graph Anchors (LeWM)',
        'gcbc': 'GCBC', 'gcivl': 'GCIVL', 'gciql': 'GCIQL', 'hilp': 'HILP', 'pldm': 'PLDM', 'pldmga': 'Graph Anchors (PLDM)',
        'dwm': 'DINO-WM (10 episodes)'}


def main():
    cell = defaultdict(dict)   # method -> task -> {seed: rate}
    for line in open(sys.argv[1], errors='replace'):
        m = re.match(r'(\w+?)_t(\d+)_s(\d+) .*success_rate=([0-9.]+)', line)
        if m:
            cell[m.group(1)].setdefault(int(m.group(2)), {})[int(m.group(3))] = float(m.group(4))
    tasks = sorted({t for v in cell.values() for t in v})
    if not tasks:
        print('no finished run yet'); return
    print(f'{"method":26s}' + ''.join(f'{"task " + str(t):>9s}' for t in tasks) + f'{"mean":>9s}  seeds')
    for k in ORDER + sorted(set(cell) - set(ORDER)):
        if k not in cell:
            continue
        per = {t: sum(cell[k][t].values()) / len(cell[k][t]) for t in cell[k]}
        row = ''.join(f'{per[t]:9.1f}' if t in per else f'{"-":>9s}' for t in tasks)
        mean = f'{sum(per.values()) / len(per):9.1f}' if len(per) == len(tasks) else f'{"-":>9s}'
        print(f'{NAME.get(k, k):26s}{row}{mean}  {sorted({s for t in cell[k].values() for s in t})}')


if __name__ == '__main__':
    main()
