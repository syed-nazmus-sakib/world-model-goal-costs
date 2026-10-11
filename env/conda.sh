# Locate Conda without a machine-specific installation path.
project_conda_root() {
    if [[ -n "${CONDA_ROOT:-}" ]]; then
        printf '%s\n' "$CONDA_ROOT"
    elif command -v conda >/dev/null 2>&1; then
        conda info --base
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        printf '%s\n' "$HOME/miniconda3"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        printf '%s\n' "$HOME/anaconda3"
    elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
        printf '%s\n' "$HOME/miniforge3"
    else
        printf '%s\n' 'Conda not found. Set CONDA_ROOT to your Conda installation.' >&2
        return 1
    fi
}

project_activate() {
    local project_conda
    project_conda="$(project_conda_root)" || return 1
    if [[ ! -f "$project_conda/etc/profile.d/conda.sh" ]]; then
        printf 'Conda activation script missing under %s\n' "$project_conda" >&2
        return 1
    fi
    source "$project_conda/etc/profile.d/conda.sh" || return 1
    conda activate "$1"
}
