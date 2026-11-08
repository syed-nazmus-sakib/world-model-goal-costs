#!/usr/bin/env bash
# Runs J17 and J19 at once (or any jobs given in PAIRS), each with its own output folder, on whichever GPUs have free memory right now.
#
#   bash scripts/hpc/run_j17_j19.sh                 # default: J17 -> out/hpc4 and J19 -> out/hpc6
#   PLAN_ONLY=1 bash scripts/hpc/run_j17_j19.sh     # only show which job would go to which GPU
#   PAIRS="J17:out/hpc4 J19:out/hpc6" NEED_GB=40 bash scripts/hpc/run_j17_j19.sh
#
# How it decides: it reads the FREE memory of every GPU (so a GPU already busy with another job, e.g. J18, counts only
# with what is left on it) and gives each job the GPU with the most free memory, reserving NEED_GB (default 40) per job.
# Two jobs may share one large GPU if it has room for both. A job that fits nowhere starts on the first job's GPU as
# soon as that job has finished. Jobs run detached (closing the terminal does not stop them); each writes only to its
# own folder, so results never overwrite each other, and each folder ends with its own ALL_results.tgz.
# Re-running this script after a time limit continues every job where it stopped (finished runs are skipped).
# Logs: <folder>.launch.log next to each output folder; progress: tail -f <folder>/run_all.log
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PAIRS="${PAIRS:-J17:out/hpc4 J19:out/hpc6}"; NEED="${NEED_GB:-40}"

FREE=()   # indexed by GPU number
while IFS=', ' read -r idx mib; do FREE[$idx]=$(( mib / 1024 )); done \
    < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
[[ ${#FREE[@]} -gt 0 ]] || { echo "nvidia-smi found no GPU"; exit 1; }
echo "free GPU memory now:$(for g in $(printf '%s\n' "${!FREE[@]}" | sort -n); do printf ' GPU %s %s GB,' "$g" "${FREE[$g]}"; done)"
echo "free host RAM now: $(free -g 2>/dev/null | awk '/^Mem:/ {print $7}') GB"

plan=()
# GPU=<id>: put every job on that GPU and no other (on a cluster where some GPUs are reserved for others).  The DINO-WM
# experiments wait for free memory on it themselves.  Without GPU, each job goes to the GPU with the most free memory.
if [[ -n "${GPU:-}" ]]; then
    [[ -n "${FREE[$GPU]:-}" ]] || { echo "GPU $GPU not found by nvidia-smi"; exit 1; }
    for p in $PAIRS; do plan+=("${p%%:*} ${p#*:} $GPU"); done
fi
for p in $([[ -z "${GPU:-}" ]] && echo $PAIRS); do
    j="${p%%:*}"; o="${p#*:}"; best=""
    for g in "${!FREE[@]}"; do
        (( FREE[$g] >= NEED )) && { [[ -z "$best" ]] || (( FREE[$g] > FREE[$best] )); } && best=$g
    done
    if [[ -n "$best" ]]; then FREE[$best]=$(( FREE[$best] - NEED )); plan+=("$j $o $best"); else plan+=("$j $o later"); fi
done
echo "plan:"; for x in "${plan[@]}"; do read -r j o g <<< "$x"
    if [[ "$g" == later ]]; then echo "  $j -> $o, starts when the first job has finished (no GPU with $NEED GB free now)"
    else echo "  $j -> $o on GPU $g"; fi; done
[[ "${PLAN_ONLY:-0}" == 1 ]] && exit 0

DET=""; command -v setsid > /dev/null && DET="setsid"
first_pid=""; first_gpu=""
for x in "${plan[@]}"; do
    read -r j o g <<< "$x"; mkdir -p "$o"
    if [[ "$g" == later ]]; then
        [[ -n "$first_pid" ]] || { echo "  $j: no free GPU at all; free one and run this script again"; continue; }
        $DET nohup bash -c "while kill -0 $first_pid 2>/dev/null; do sleep 300; done
            GPU=$first_gpu OUT_ROOT='$o' JOBS='$j' bash scripts/hpc/run_1gpu.sh" > "$o.launch.log" 2>&1 < /dev/null &
        echo "  $j queued behind the first job (log: $o.launch.log)"
    else
        $DET nohup env GPU="$g" OUT_ROOT="$o" JOBS="$j" bash scripts/hpc/run_1gpu.sh > "$o.launch.log" 2>&1 < /dev/null &
        pid=$!; [[ -z "$first_pid" ]] && { first_pid=$pid; first_gpu=$g; }
        echo "  $j started on GPU $g (log: $o.launch.log, progress: $o/run_all.log)"
        sleep 120   # let its environment check and smoke test start before the next job does the same
    fi
done
echo "All started. You can close the terminal. When done, please send: $(for x in "${plan[@]}"; do read -r j o g <<< "$x"; printf '%s/ALL_results.tgz ' "$o"; done)"
