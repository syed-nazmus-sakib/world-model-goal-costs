#!/bin/bash
# Quick tests on a running development session (chains/E0_devloop.sh): refresh the code on the VM, then queue one script.
# usage: bash scripts/colab/devq.sh <sess> <name> '<shell lines run on the VM after sourcing head.sh>'
set -u
PROJ=/Volumes/ns_external/Shared/jepa-wms-project; S=$1; N=$2; CMD=$3; T=$(mktemp -d)
SSHO=(-o "ProxyCommand=colab ssh --proxy-mode -s $S" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
      -o ServerAliveInterval=30 -o ConnectTimeout=60 -i "$HOME/.ssh/id_ed25519")
( cd "$PROJ" && COPYFILE_DISABLE=1 tar czf "$T/up.tgz" --exclude='._*' --exclude='__pycache__' code/lewm_explorations/paa/*.py \
    code/third_party/stable-worldmodel/scripts/plan/eval_wm.py code/third_party/stable-worldmodel/scripts/plan/config/objective scripts/pod scripts/exp scripts/hpc )
printf '#!/bin/bash\nsource /content/head.sh\nexport CK\n%s\n' "$CMD" > "$T/$N.sh"
# Colab takes one SSH connection at a time (HTTP 429 otherwise; the watcher's pulls count), so every step retries
try() { for i in 1 2 3 4 5 6 7 8 9 10; do "$@" 2>/dev/null && return 0; sleep 20; done; echo "failed: $1"; return 1; }
try timeout 300 scp "${SSHO[@]}" "$T/up.tgz" "root@colab-$S:/content/up.tgz" && try timeout 120 scp "${SSHO[@]}" "$T/$N.sh" "root@colab-$S:/content/$N.tmp" \
  && try timeout 120 ssh "${SSHO[@]}" "root@colab-$S" "tar xzf /content/up.tgz -C /content/jepa-wms-project && mkdir -p /content/queue && mv /content/$N.tmp /content/queue/$N.sh && echo queued $N"
rm -rf "$T"
