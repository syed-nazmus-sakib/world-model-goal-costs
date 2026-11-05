#!/bin/bash
# Pod side: released DINO-WM Push-T checkpoint at long range (N_DINO pairs per distance, seed 0, same pairs as the graph anchor).
# Expects /root/ga_code.tgz (code bundle) and /root/dinowm/{config.yaml,dinowm_noprop_pusht_weights.ckpt}.
# Writes /root/out/{chain.log,dinowm_pusht/}, then /root/out/DONE. The pod removes itself once the laptop has written
# /root/out/PULLED, or WAIT_H hours after DONE (default 2), or CAP_H hours after start (default 14), so a lost laptop
# connection cannot keep it billing. Results already pulled survive; the laptop pulls every 10 min.
# usage (on the pod, detached): DISTS="150 50" nohup bash /root/run_dinowm_pod.sh > /root/chain.nohup 2>&1 &
set -u
DISTS=${DISTS:-"150 100 75 50"}; N_DINO=${N_DINO:-20}; WAIT_H=${WAIT_H:-2}; CAP_H=${CAP_H:-14}
O=/root/out; mkdir -p $O; L=$O/chain.log; log() { echo "$(date -u +%F' '%T) $*" >> $L; }
selfremove() { log "removing pod ($1)"; runpodctl remove pod "${RUNPOD_POD_ID:?}" >> $L 2>&1 || runpodctl stop pod "$RUNPOD_POD_ID" >> $L 2>&1; }
( sleep $((CAP_H * 3600)); selfremove "cap ${CAP_H}h reached" ) &
log "POD_CHAIN_START dinowm dists=[$DISTS] n=$N_DINO"
mkdir -p /root/jepa-wms-project && tar xzf /root/ga_code.tgz -C /root/jepa-wms-project 2>/dev/null
bash /root/jepa-wms-project/scripts/pod/setup_swm_venv.sh > $O/setup.log 2>&1 || { log "setup FAILED: $(tail -2 $O/setup.log)"; touch $O/DONE; }
if [ ! -f $O/DONE ]; then
  log "setup: $(grep SWM $O/setup.log | tail -1)"
  cd /root/jepa-wms-project
  for d in $DISTS; do
    DOFFS=$d N_DINO=$N_DINO DINO_CFG=/root/dinowm/config.yaml DINO_CKPT=/root/dinowm/dinowm_noprop_pusht_weights.ckpt \
      bash scripts/pod/run_baselines.sh pusht dinowm $O/dinowm_pusht >> $O/laneA.log 2>&1
    log "dinowm d$d: $(grep -a "dinowm_d${d}_" $O/dinowm_pusht/summary.txt 2>/dev/null | grep -oE 'success_rate=[0-9.]+' | tail -1)"
  done
  log "POD_CHAIN_DONE"; touch $O/DONE
fi
for i in $(seq 1 $((WAIT_H * 60))); do [ -f $O/PULLED ] && break; sleep 60; done
selfremove "$([ -f $O/PULLED ] && echo pulled || echo "no pull after ${WAIT_H}h")"
