"""OGBench-Shift: re-render the GOAL image of OGBench visual tasks under appearance
shifts while the observation stream stays nominal. The physical goal state is
identical; only how it looks changes.

Usage
    env = ogbench.make_env_and_datasets('visual-cube-single-play-v0', env_only=True)
    env = ShiftedGoalEnv(env, shift='floor_color', severity=2, seed=0)
    ob, info = env.reset(options=dict(task_id=1))
    info['goal']          # shifted goal image (64x64x3 uint8)
    info['goal_nominal']  # the original OGBench goal image

Shift families (severity 1..3):
    floor_color   tint of the floor/grid material
    light         global light intensity / colour
    robot_color   recolour robot materials (manip) / agent body (maze)
    object_color  recolour cube(s)  (manip; single-cube only, colour is semantic otherwise)
    camera        small viewpoint perturbation of the rendering camera (geometric shift)
    blur, noise, jpeg   image-space corruptions applied to the rendered goal
"""
import io

import gymnasium
import mujoco
import numpy as np

FAMILIES = ['floor_color', 'light', 'robot_color', 'object_color', 'camera', 'blur', 'noise', 'jpeg']


class ShiftedGoalEnv(gymnasium.Wrapper):
    """Apply directly around the raw OGBench env (inside any FrameStack wrapper)."""

    def __init__(self, env, shift='floor_color', severity=1, seed=0):
        assert shift in FAMILIES + ['none'], shift
        super().__init__(env)
        self.shift, self.sev = shift, severity
        self.u = env.unwrapped
        self.rng = np.random.default_rng(seed)
        self._snaps = {}
        self._patch_observation()

    # -- capture the simulator state behind every observation computed in reset --
    def _patch_observation(self):
        u = self.u
        fname = 'compute_observation' if hasattr(u, 'compute_observation') else 'get_ob'
        orig = getattr(u, fname)
        model, data = self._md()

        def wrapped(*a, **k):
            ob = orig(*a, **k)
            # full MjData copy: OGBench renders goals right after mj_step, when body poses are
            # those of the previous substep, so qpos + mj_forward would NOT reproduce the image.
            snap = mujoco.MjData(model)
            mujoco.mj_copyData(snap, model, data)
            self._snaps[id(ob)] = snap
            return ob
        setattr(u, fname, wrapped)

    def _md(self):
        u = self.u
        model = getattr(u, '_model', None) or u.model
        data = getattr(u, '_data', None) or u.data
        return model, data

    def reset(self, *a, **k):
        self._snaps.clear()
        ob, info = self.env.reset(*a, **k)
        info['goal_nominal'] = info['goal']
        snap = self._snaps.get(id(info['goal']))
        assert snap is not None, 'goal snapshot not captured'
        self.last_goal_snap = snap
        if self.shift != 'none':
            info['goal'] = self.render_state(snap)
        self._snaps.clear()
        return ob, info

    # -- rendering under shift --------------------------------------------------
    def render_state(self, snap):
        model, data = self._md()
        cur = mujoco.MjData(model)
        mujoco.mj_copyData(cur, model, data)
        saved = self._apply(model)
        try:
            mujoco.mj_copyData(data, model, snap)   # exact goal-time state, no re-integration
            img = self.u.render().copy()
        finally:
            self._restore(model, saved)
            mujoco.mj_copyData(data, model, cur)
        return self._image_shift(img)

    def _mat_ids(self, model, pred):
        out = []
        for i in range(model.nmat):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i) or ''
            if pred(n):
                out.append(i)
        return out

    def _geom_ids(self, model, pred):
        out = []
        for i in range(model.ngeom):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
            if pred(n):
                out.append(i)
        return out

    def _rand_color(self):
        return np.concatenate([self.rng.uniform(0.1, 1.0, 3), [1.0]])

    def _apply(self, model):
        s, sev = self.shift, self.sev
        saved = dict(mat_rgba=model.mat_rgba.copy(), geom_rgba=model.geom_rgba.copy(),
                     light_diffuse=model.light_diffuse.copy(), light_ambient=model.light_ambient.copy(),
                     cam_pos=model.cam_pos.copy(), cam_quat=model.cam_quat.copy())
        mix = [0.35, 0.65, 1.0][sev - 1]   # how far towards the random colour
        if s == 'floor_color':
            for i in self._mat_ids(model, lambda n: n in ('grid', 'floor', 'groundplane')):
                model.mat_rgba[i] = (1 - mix) * model.mat_rgba[i] + mix * self._rand_color()
        elif s == 'light':
            scale = [0.6, 0.35, 0.15][sev - 1] if self.rng.random() < 0.5 else [1.5, 2.2, 3.0][sev - 1]
            tint = 1 + mix * self.rng.uniform(-0.5, 0.5, 3)
            model.light_diffuse[:] = np.clip(model.light_diffuse * scale * tint, 0, 1)
            model.light_ambient[:] = np.clip(model.light_ambient * scale, 0, 1)
        elif s == 'robot_color':
            ids = self._mat_ids(model, lambda n: n.startswith('ur5e/') or n in ('self', 'self_white'))
            c = self._rand_color()
            for i in ids:
                model.mat_rgba[i] = (1 - mix) * model.mat_rgba[i] + mix * c
            # some bodies colour geoms directly
            for i in self._geom_ids(model, lambda n: 'torso' in n or 'leg' in n or 'ankle' in n):
                model.geom_rgba[i] = (1 - mix) * model.geom_rgba[i] + mix * c
        elif s == 'object_color':
            c = self._rand_color()
            for i in self._geom_ids(model, lambda n: n.startswith('object_') or 'cube' in n):
                model.geom_rgba[i] = (1 - mix) * model.geom_rgba[i] + mix * c
        elif s == 'camera':
            cid = model.ncam - 1  # 'front_pixels' (manip) / 'back' (maze)
            for nm in ('front_pixels', 'back'):
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, nm)
                if j >= 0:
                    cid = j
            d = [0.03, 0.06, 0.10][sev - 1]
            model.cam_pos[cid] += self.rng.uniform(-d, d, 3) * np.maximum(np.abs(model.cam_pos[cid]), 1)
            ang = [3, 6, 10][sev - 1] * np.pi / 180 * self.rng.uniform(-1, 1)
            q = np.array([np.cos(ang / 2), 0, 0, np.sin(ang / 2)])
            model.cam_quat[cid] = _qmul(model.cam_quat[cid], q)
        return saved

    def _restore(self, model, saved):
        model.mat_rgba[:] = saved['mat_rgba']; model.geom_rgba[:] = saved['geom_rgba']
        model.light_diffuse[:] = saved['light_diffuse']; model.light_ambient[:] = saved['light_ambient']
        model.cam_pos[:] = saved['cam_pos']; model.cam_quat[:] = saved['cam_quat']

    def _image_shift(self, img):
        s, sev = self.shift, self.sev
        if s == 'blur':
            from PIL import Image, ImageFilter
            return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur([0.6, 1.0, 1.6][sev - 1])))
        if s == 'noise':
            n = self.rng.normal(0, [8, 16, 28][sev - 1], img.shape)
            return np.clip(img + n, 0, 255).astype(np.uint8)
        if s == 'jpeg':
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format='JPEG', quality=[30, 15, 7][sev - 1])
            return np.asarray(Image.open(buf).convert('RGB'))
        return img


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
