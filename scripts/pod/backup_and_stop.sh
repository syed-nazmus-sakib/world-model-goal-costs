#!/bin/bash
# Laptop-side watcher: wait until <remote_out>/ALL_DONE exists on the pod (or the deadline passes), copy the results
# (without the large encoding files) to the home folder first and to the project if the drive is mounted, then stop
# the pod through the RunPod API. The pod is stopped only if the home copy holds the queue log.
# The API key is read from the project's .env (variable runpod_api) and never printed. Stopping wipes the container disk.
# The pod images have no rsync: the copy is tar over ssh.
# usage: bash scripts/pod/backup_and_stop.sh <pod_id> <ssh_host> <ssh_port> <remote_out> <name> [deadline_s=21600]
set -u
ID=$1; HOST=$2; PORT=$3; REMOTE=$4; NAME=$5; DEADLINE=$(( $(date +%s) + ${6:-21600} ))
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HOME_DST=$HOME/jepa_backup_$NAME
PROJ_DST=$ROOT/artifacts/lewm_explorations_raw/$NAME
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -i $HOME/.ssh/id_ed25519 -p $PORT root@$HOST"
while true; do
  $SSH "test -f $REMOTE/ALL_DONE" 2>/dev/null && { echo "ALL_DONE seen $(date)"; break; }
  [ $(date +%s) -gt $DEADLINE ] && { echo "deadline reached $(date)"; break; }
  sleep 300
done
mkdir -p $HOME_DST
for i in 1 2 3; do
  $SSH "cd $REMOTE && tar czf - --exclude='*/train/ep_*.npz' --exclude='*/test/ep_*.npz' ." | tar xzf - -C $HOME_DST && break
  sleep 30
done
N=$(find $HOME_DST -type f | wc -l); echo "home copy: $N files in $HOME_DST"
[ -d "$ROOT/code" ] && mkdir -p $PROJ_DST && cp -R $HOME_DST/. $PROJ_DST/ && echo "project copy: $PROJ_DST"
for i in $(seq 1 24); do [ -f "$ROOT/.env" ] && break; echo "waiting for the project drive (.env)"; sleep 300; done
if [ $N -gt 3 ] && [ -f $HOME_DST/queue.log ] && [ -f "$ROOT/.env" ]; then
  KEY=$(sed -n -E 's/^runpod_api[[:space:]]*=[[:space:]]*//p' "$ROOT/.env" | sed -E 's/^["'"'"']//; s/["'"'"']$//; s/[[:space:]]+$//')
  curl -s -m 30 -o /dev/null -w "stop http %{http_code}\n" -X POST -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods/$ID/stop
else
  echo "BACKUP INCOMPLETE: pod left running"
fi
