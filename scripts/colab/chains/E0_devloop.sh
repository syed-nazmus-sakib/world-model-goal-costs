#!/bin/bash
# Development session for the exp-NN quick tests: unpacks the saved checkpoints, then runs every script dropped into
# /content/queue/ one at a time (output /content/out/<name>.log) until /content/STOP exists.
source /content/head.sh
log "GA_CHAIN_START E0 devloop"
for t in /content/ck/*.tgz; do [ -f "$t" ] && tar xzf "$t" -C $CK --exclude='._*'; done
log "checkpoints: $(ls $CK | tr '\n' ' ')"
mkdir -p /content/queue /content/qdone
while [ ! -f /content/STOP ]; do
  for s in /content/queue/*.sh; do
    [ -f "$s" ] || continue
    n=$(basename $s .sh); log "$n start"; bash $s > $O/$n.log 2>&1; log "$n exit $?"; mv $s /content/qdone/
  done
  sleep 15
done
log "GA_CHAIN_DONE"
