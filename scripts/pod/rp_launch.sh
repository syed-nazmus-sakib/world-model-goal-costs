#!/bin/bash
# Laptop side: find a pod (find_pod.sh), upload the code bundle, a pod chain script and extra files, start the chain
# detached, then pull /root/out every 10 min into $DEST until the chain writes DONE; finally mark PULLED so the pod
# removes itself. The laptop also deletes the pod through the REST API once DONE is pulled, or LCAP_H hours after launch
# (default 10): the pod-scoped RUNPOD_API_KEY cannot be trusted to work (runpodctl get pod: Unauthorized, 2026-11-05).
# scp -O (legacy protocol) + timeouts: the pods' sftp-server sometimes never closes after a finished upload (2026-11-05).
# ATTACH=1: the pod in $DEST/pod already runs its chain; only pull and delete. Keys come from .env, never printed.
# usage: KEYVAR=runpod_api NAME=jwm-dino-a DEST=<dir> ENVS='DISTS="150 50"' [LCAP_H=10] [ATTACH=1] bash scripts/pod/rp_launch.sh <chain.sh> "<local:remote ...>"
set -u
cd /Volumes/ns_external/Shared/jepa-wms-project
CHAIN=$1; FILES=${2:-}; DEST=${DEST:?set DEST}; mkdir -p "$DEST"; PD=$DEST/pod; mkdir -p $PD
LOG=$DEST/launch.log; log() { echo "$(date -u +%F' '%T) $*" >> $LOG; }
KEYVAR=${KEYVAR:-runpod_api}; T0=$(date +%s); LCAP=$(( ${LCAP_H:-10} * 3600 ))
KEY=$(sed -n -E "s/^$KEYVAR[[:space:]]*=[[:space:]]*//p" .env | sed -E 's/^["'"'"']//; s/["'"'"']$//; s/[[:space:]]+$//' | tr -d '\n')
delpod() { curl -s -m 60 -o /dev/null -w "%{http_code}" -X DELETE -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods/$(cat $PD/pod_id); }
log "launch start (${NAME:-}, $CHAIN, attach=${ATTACH:-0})"
if [ "${ATTACH:-0}" != 1 ]; then
NEED_EGL=${NEED_EGL:-0} PODDIR=$PD bash scripts/pod/find_pod.sh >> $LOG 2>&1 || { log "no pod found"; exit 1; }
fi
S=$PD/pod_ssh.sh; read PORT IP < $PD/pod_addr
cp_up() { for i in 1 2 3 4; do timeout 900 scp -O -q -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 -P $PORT "$1" root@$IP:"$2" && return 0; sleep 20; done; log "upload failed: $1"; return 1; }
COPYFILE_DISABLE=1 tar czf $PD/code.tgz --exclude='._*' --exclude='.git' --exclude='__pycache__' --exclude='outputs' \
  code/third_party/stable-worldmodel code/lewm_explorations/paa scripts/pod scripts/hpc env/conda.sh env/paths.sh
if [ "${ATTACH:-0}" != 1 ]; then
cp_up $PD/code.tgz /root/ga_code.tgz || exit 1
cp_up "$CHAIN" /root/chain.sh || exit 1
for f in $FILES; do bash $S "mkdir -p $(dirname ${f#*:})"; cp_up "${f%%:*}" "${f#*:}" || exit 1; done
bash $S "command -v runpodctl > /dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ] && echo SELFSTOP_OK || echo SELFSTOP_MISSING" >> $LOG
bash $S "cd /root && ${ENVS:-} nohup bash /root/chain.sh > /root/chain.nohup 2>&1 < /dev/null &" && log "chain started on $(cat $PD/pod_id) $IP:$PORT"
bash $S "sleep 5; pgrep -f /root/chain.sh > /dev/null" || { log "chain did not start: $(bash $S 'tail -3 /root/chain.nohup' 2>&1)"; }
fi
F=0
while true; do
  sleep 600
  [ $(( $(date +%s) - T0 )) -ge $LCAP ] && { log "laptop cap ${LCAP_H:-10}h: final pull, then delete"; timeout 300 bash $S "cd /root && tar czf /root/out.tgz out" 2>/dev/null; \
    timeout 900 scp -O -q -o BatchMode=yes -o ConnectTimeout=30 -i ~/.ssh/id_ed25519 -P $PORT root@$IP:/root/out.tgz $DEST/out_final.tgz 2>/dev/null; log "delete pod: http $(delpod)"; exit 0; }
  if timeout 300 bash $S "cd /root && tar czf /root/out.tgz out" 2>/dev/null && \
     timeout 900 scp -O -q -o BatchMode=yes -o ConnectTimeout=30 -i ~/.ssh/id_ed25519 -P $PORT root@$IP:/root/out.tgz $DEST/out.tgz.tmp 2>/dev/null; then
    F=0; mv $DEST/out.tgz.tmp $DEST/out.tgz; tar xzf $DEST/out.tgz -C $DEST 2>/dev/null
    log "pulled: $(tail -1 $DEST/out/chain.log 2>/dev/null | cut -c1-160)"
    if [ -f $DEST/out/DONE ]; then bash $S "touch /root/out/PULLED"; log "DONE pulled; delete pod: http $(delpod)"; exit 0; fi
  else
    F=$((F + 1)); log "pull failed ($F; network down or pod removed)"
    [ $F -ge 36 ] && { log "no contact for 6 h, giving up (pod has its own cap)"; exit 1; }
  fi
done
