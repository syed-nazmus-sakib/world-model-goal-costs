#!/usr/bin/env bash
# Pack every model trained by the HPC jobs, so they can be reused later without retraining (they are NOT in
# ALL_results.tgz, which holds logs and result files only).  Final checkpoint of each model only (older epochs skipped).
#
#   bash scripts/hpc/pack_models.sh            -> out/MODELS.tgz  (a few GB: please share it with a Drive/Box link)
#
# Packed (when present under $STABLEWM_HOME/checkpoints):
#   lewm_pml          our LeWM on PointMaze-Large (J1; every PointMaze-Large result depends on it)
#   bl_<env>_<m>*     baselines: GCBC, GCIVL, HILP, GCIQL policies, PLDM and DINO-WM world models (J2-J3, J6-J7, J9-J15)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CK="${STABLEWM_HOME:-$HOME/.stable_worldmodel}/checkpoints"
OUT="${OUT_ROOT:-$ROOT/out}"; mkdir -p "$OUT"
[[ -d "$CK" ]] || { echo "no checkpoint folder at $CK (set STABLEWM_HOME)"; exit 1; }
cd "$CK"
items=(); for d in lewm_pml bl_*; do [[ -e "$d" ]] && items+=("$d"); done
[[ ${#items[@]} -gt 0 ]] || { echo "no trained models found in $CK"; exit 1; }
echo "packing from $CK:"; for d in "${items[@]}"; do echo "  $d ($(du -sh "$d" | cut -f1))"; done
tar czf "$OUT/MODELS.tgz" --exclude='older_epochs' --exclude='*.tmp' "${items[@]}" \
  && echo "DONE: $OUT/MODELS.tgz ($(du -h "$OUT/MODELS.tgz" | cut -f1)) -- please share it with a Drive/Box link"
