"""Portable locations used by the LeWM experiments (no ML imports)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SWM_ROOT = PROJECT_ROOT / "code" / "third_party" / "stable-worldmodel"
H5 = str(Path(os.environ.get(
    "PUSHT_EXPERT_H5", str(Path.home() / ".stable_worldmodel" / "datasets" / "pusht_expert_train.h5")
)).expanduser())
