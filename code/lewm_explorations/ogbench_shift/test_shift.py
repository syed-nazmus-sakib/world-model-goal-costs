"""Sanity test: render nominal vs shifted goals for each family; save a montage;
check the observation stream is unaffected and the nominal goal is reproduced."""
import sys, numpy as np, ogbench
from PIL import Image
from shift import ShiftedGoalEnv, FAMILIES

envs = sys.argv[1:] or ['visual-cube-single-play-v0', 'visual-antmaze-medium-navigate-v0', 'visual-scene-play-v0']
rows = []
for name in envs:
    base = ogbench.make_env_and_datasets(name, env_only=True)
    w = ShiftedGoalEnv(base, shift='none')
    ob0, info0 = w.reset(seed=0, options=dict(task_id=1))
    # fidelity: re-rendering the captured goal state with no shift reproduces OGBench's goal
    rer = w.render_state(w.last_goal_snap)
    fid = np.abs(rer.astype(int) - info0['goal'].astype(int)).max()
    assert fid <= 2, f'{name}: re-render mismatch (max |diff| {fid})'
    row = [info0['goal']]
    for fam in FAMILIES:
        for sev in (1, 3):
            w.shift, w.sev = fam, sev
            ob, info = w.reset(seed=0, options=dict(task_id=1))
            # the observation stream must be untouched by the shift (compare to a fresh nominal render)
            assert np.array_equal(ob, w.u.render()), f'{name}/{fam}: observation changed!'
            row.append(info['goal'])
    rows.append(np.concatenate(row, 1))
    d = np.abs(row[1].astype(int) - row[0].astype(int)).mean()
    print(f'{name}: ok, {len(row) - 1} shifted goals, mean |diff| first shift = {d:.1f}')
Image.fromarray(np.concatenate(rows, 0)).resize((len(row) * 96, len(rows) * 96), Image.NEAREST).save('shift_montage.png')
print('saved shift_montage.png  (col 0 = nominal goal, then', ', '.join(f'{f}@1,{f}@3' for f in FAMILIES), ')')
