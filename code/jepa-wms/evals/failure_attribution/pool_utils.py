"""Bounded worker waits without importing CUDA or simulation packages."""

import multiprocessing as mp
import os


def rollout_map(pool, function, jobs, timeout=None):
    timeout = float(os.environ.get("JEPAWM_ROLLOUT_TIMEOUT", "300")) if timeout is None else timeout
    if timeout <= 0:
        raise ValueError("JEPAWM_ROLLOUT_TIMEOUT must be positive")
    try:
        return pool.map_async(function, jobs).get(timeout=timeout)
    except mp.TimeoutError as error:
        pool.terminate()
        pool.join()
        raise RuntimeError(
            f"Simulator workers did not return within {timeout:g}s. "
            "Check worker errors; rerun to resume completed episodes. "
            "Set JEPAWM_ROLLOUT_TIMEOUT for slower machines."
        ) from error
