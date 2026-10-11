"""Shared, mutually exclusive attribution of world-model failures."""

import numpy as np


def failure_masks(wm, sim_latent, sim_true):
    wm, latent, true = (np.asarray(x, dtype=bool) for x in (wm, sim_latent, sim_true))
    if wm.shape != latent.shape or wm.shape != true.shape:
        raise ValueError("Planner outcomes must describe the same episodes")
    fail = ~wm
    return {
        "dynamics": fail & latent,
        "cost": fail & ~latent & true,
        "search": fail & ~latent & ~true,
    }
