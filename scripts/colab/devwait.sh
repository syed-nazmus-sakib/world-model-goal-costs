#!/bin/bash
# Wait for a queued development test, then print the end of its log.  usage: bash scripts/colab/devwait.sh <sess> <name> [lines]
S=$1; N=$2; L=${3:-40}
SSHO=(-o "ProxyCommand=colab ssh --proxy-mode -s $S" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
      -o ConnectTimeout=60 -i "$HOME/.ssh/id_ed25519")
until timeout 90 ssh "${SSHO[@]}" "root@colab-$S" "grep -q ' $N exit' /content/out/chain.log" 2>/dev/null; do sleep 45; done
until timeout 120 ssh "${SSHO[@]}" "root@colab-$S" "grep ' $N ' /content/out/chain.log | cut -c1-160; tail -$L /content/out/$N.log | cut -c1-240" 2>/dev/null; do sleep 30; done
