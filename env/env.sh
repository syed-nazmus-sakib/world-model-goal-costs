# Source this from anywhere:  source env/env.sh
# Paths are derived from this file's location, so the project folder can be moved freely.
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || return 1
source "$(dirname "${BASH_SOURCE[0]}")/conda.sh" || return 1
project_activate jepa-wms || return 1
