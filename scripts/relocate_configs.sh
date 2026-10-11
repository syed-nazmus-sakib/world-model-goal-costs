#!/usr/bin/env bash
# Convert legacy absolute artifact paths to environment-variable placeholders.
# Current configs already use these placeholders; moving the checkout needs no rewrite.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$PROJECT_ROOT" <<'PY'
import re
import sys
from pathlib import Path

changed = 0
for path in sorted((Path(sys.argv[1]) / 'configs').glob('*.yaml')):
    original = path.read_text()
    variables = {'raw_logs': 'JEPAWM_LOGS', 'checkpoints': 'JEPAWM_CKPT'}
    updated = re.sub(
        r'(?m)^(\s*(?:folder|checkpoint):\s*)/[^\n]*?/artifacts/(raw_logs|checkpoints)/',
        lambda match: match[1] + '${' + variables[match[2]] + '}/',
        original,
    )
    if updated != original:
        path.write_text(updated)
        changed += 1
print(f'{changed} config(s) converted; paths now come from env/paths.sh.')
PY
