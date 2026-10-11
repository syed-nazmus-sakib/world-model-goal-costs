# Source provenance and licenses

This repository bundles upstream source snapshots so a GitHub clone includes the
local evaluation additions. The two snapshots are ordinary folders, not Git
submodules. Large datasets, weights, caches, and raw runs live under `artifacts/`
and are excluded from Git; its README and small curated results are included.

| Source | Upstream revision | Bundled location | License |
|---|---|---|---|
| [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` | `code/jepa-wms/` | [upstream license](code/jepa-wms/LICENSE), with [third-party notices](code/jepa-wms/THIRD-PARTY-LICENSES.md) |
| [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | `8b2e8c685dd7f11d4189f5c09d9b6bfffd5cdca4` | `code/third_party/stable-worldmodel/` | [MIT license](code/third_party/stable-worldmodel/LICENSE) |

The upstream licenses and notices remain in their source folders. Bundling them
does not change their terms or grant a different license for those files.

Project additions include the failure-attribution, goal-decomposition, metric,
hidden-physics and action-cycle evaluations in `code/jepa-wms/evals/`, their
simulator tests, the LeWM study in `code/lewm_explorations/`, and the GoalShift /
TwoRoomTrap additions described in `code/README.md`. The JEPA-WMs ignore rules
have also been adjusted to include our two `run.py` entry points.

During preparation for GitHub, the nested `.git` folders were preserved locally
under `artifacts/local_git/{jepa-wms,stable-worldmodel}.git`. They are excluded
from Git. The local copies retain the upstream commit history; a fresh clone
obtains the bundled snapshots and the revisions documented above.

Small result arrays under `results/` are deliberately tracked as research
evidence. Original per-episode files are excluded under `artifacts/raw_logs/`.
Earlier summaries corrected during maintenance are preserved locally in
`artifacts/maintenance_backup_2026-10-12/`.
