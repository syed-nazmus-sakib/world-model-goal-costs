#!/bin/bash
# Live view of the evaluation runs under an output folder: the latest [progress] line of every run log written to in the
# last 15 minutes, plus finished results and failures since then.  Refreshes every 60 s (Ctrl-C to stop).
# usage: bash scripts/hpc/progress.sh [out]        one-shot: ONCE=1 bash scripts/hpc/progress.sh [out]
O=${1:-out}
show() {
  clear 2>/dev/null; echo "$(date '+%F %T')  runs active in the last 15 min under $O/"; echo
  find "$O" -name '*.log' -mmin -15 ! -name queue.log ! -name 'lane*.log' ! -name job.log ! -name run_all.log ! -name fetch.log ! -name nodes.log ! -name 'ckpt*.log' ! -name 'train_*.log' 2>/dev/null | sort | while read -r f; do
    p=$(grep -a '^\[progress\]' "$f" | tail -1 | sed 's/^\[progress\] //')
    s=$(grep -a '\[summary\]' "$f" | grep -oE 'success_rate=[0-9.]+' | tail -1)
    e=$(grep -aE 'OutOfMemoryError|Error:' "$f" | tail -1 | cut -c1-90)
    r=${f#$O/}; r=${r%.log}
    if [ -n "$s" ]; then echo "  DONE    $r  $s"
    elif [ -n "$e" ]; then echo "  FAILED  $r  $e"
    elif [ -n "$p" ]; then echo "  RUNNING $r  $p"
    else echo "  LOADING $r  (model / graph, no step yet)"; fi
  done
  echo; echo "GPU memory:"; nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /'
}
if [ "${ONCE:-0}" = 1 ]; then show; else while true; do show; sleep 60; done; fi
