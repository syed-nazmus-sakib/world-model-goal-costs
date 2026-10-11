"""Atomic outputs keep interrupted runs from leaving half-written episode files."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, prefix=".tmp-", delete=False) as handle:
            temporary = handle.name
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def save_json(path, data, **kwargs):
    _atomic_write(path, lambda handle: json.dump(data, handle, **kwargs))


def save_episode(path, **arrays):
    _atomic_write(path, lambda handle: np.savez(handle, **arrays), binary=True)
