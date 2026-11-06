#!/bin/bash
# Start colab_run.sh detached in its own screen session, so it outlives the shell (and Claude Code) that started it.
# usage: [GPU=L4] [HIGHMEM=1] [ATTACH=1] bash scripts/colab/launch.sh <sess> <chain.sh> "<local:remote ...>" <cap hours>
# watch: screen -ls; tail -f artifacts/colab_runs/logs/x_<sess>.log     stop the watcher only: pkill -f "colab_run.sh <sess>" (screen -X quit leaves it running)
C=/Volumes/ns_external/Shared/jepa-wms-project/scripts/colab
screen -dmS "$1" env GPU="${GPU:-L4}" HIGHMEM="${HIGHMEM:-}" ATTACH="${ATTACH:-0}" caffeinate -i bash "$C/colab_run.sh" "$@"
sleep 1; screen -ls | grep -F ".$1" || { echo "screen session $1 did not start"; exit 1; }
