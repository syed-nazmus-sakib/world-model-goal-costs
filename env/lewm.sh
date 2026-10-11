# Activate the earlier LeWM study from any working directory.
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh" || return 1
source "$(dirname "${BASH_SOURCE[0]}")/conda.sh" || return 1
project_activate swm || return 1
export PYTHONPATH="$JEPAWM_HOME/code/lewm_explorations${PYTHONPATH:+:$PYTHONPATH}"
export PUSHT_EXPERT_H5="${PUSHT_EXPERT_H5:-$HOME/.stable_worldmodel/datasets/pusht_expert_train.h5}"
mkdir -p "$JEPAWM_HOME/artifacts/lewm_explorations_raw"
if [[ ! -e "$JEPAWM_HOME/code/lewm_explorations/results" && ! -L "$JEPAWM_HOME/code/lewm_explorations/results" ]]; then
    ln -s ../../artifacts/lewm_explorations_raw "$JEPAWM_HOME/code/lewm_explorations/results"
fi
