"""PushT simulator helpers: exact snapshot/restore, contact classification, task distance.

Ground truth only — the planning scores never see anything from this module.
"""
import numpy as np
import gymnasium as gym
import stable_worldmodel  # noqa: F401  (registers swm/ envs)

# Stratum labels
FREE, PRECONTACT, CONTACT = 0, 1, 2
STRATA = {FREE: 'free', PRECONTACT: 'pre-contact', CONTACT: 'contact'}


def make_env(resolution=224):
    env = gym.make('swm/PushT-v1', resolution=resolution)
    env.reset(seed=0)
    return env.unwrapped


def snapshot(e):
    a, b = e.agent, e.block
    return np.array([*a.position, *a.velocity, *b.position, b.angle,
                     *b.velocity, b.angular_velocity], dtype=np.float64)


def restore(e, snap):
    """Set bodies exactly (no extra physics step, unlike env._set_state)."""
    a, b = e.agent, e.block
    a.position = tuple(snap[0:2]); a.velocity = tuple(snap[2:4])
    b.position = tuple(snap[4:6]); b.angle = float(snap[6])
    b.velocity = tuple(snap[7:9]); b.angular_velocity = float(snap[9])
    e.space.reindex_shapes_for_body(a)
    e.space.reindex_shapes_for_body(b)


def snap_to_state7(snap):
    return np.array([snap[0], snap[1], snap[4], snap[5], snap[6] % (2 * np.pi),
                     snap[2], snap[3]])


def step_raw(e, action):
    """One env step with a raw (unnormalised) 2-d action, bypassing wrappers."""
    e.step(np.asarray(action, dtype=np.float64))


def agent_block_gap(e):
    """Signed distance (px) between agent circle surface and block surface."""
    p = e.agent.position
    r = max(s.radius for s in e.agent.shapes)
    d = min(s.point_query(p).distance for s in e.block.shapes)
    return d - r


def classify(gap, pre_thresh=30.0, contact_thresh=2.0):
    if gap <= contact_thresh:
        return CONTACT
    if gap <= pre_thresh:
        return PRECONTACT
    return FREE


def angle_diff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def task_dist(state7, goal7):
    """Normalised distance matching the success test: pos < 20px, angle < pi/9.

    Values < 1 on every term <=> success-ish. Uses block pose + agent position.
    """
    s, g = np.asarray(state7), np.asarray(goal7)
    block = np.linalg.norm(s[..., 2:4] - g[..., 2:4], axis=-1) / 20.0
    agent = np.linalg.norm(s[..., 0:2] - g[..., 0:2], axis=-1) / 20.0
    ang = angle_diff(s[..., 4], g[..., 4]) / (np.pi / 9)
    return block + ang + 0.5 * agent


def render_state(e, snap):
    restore(e, snap)
    return e.render()
