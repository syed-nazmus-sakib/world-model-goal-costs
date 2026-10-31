#!/bin/bash
# Create a cheap pod, keep it only if CUDA initialises and the NVIDIA EGL driver is present; otherwise terminate it and
# try again (up to 8 pods). After two bad pods on the same host IP, that GPU type is dropped from the list.
# usage: PODDIR=<dir> [NAME=..] [KEYVAR=..] bash scripts/pod/find_pod.sh
# Writes pod_id, pod_ssh.sh, pod_addr into $PODDIR on success and prints FOUND; prints GAVE_UP otherwise. Key from .env, never printed.
cd /Volumes/ns_external/Shared/jepa-wms-project
KEYVAR=${KEYVAR:-runpod_api}
NAME=${NAME:-jepa-wms-p}; PODDIR=${PODDIR:?set PODDIR to a folder for pod_id, pod_addr, pod_ssh.sh}; mkdir -p "$PODDIR"
KEY=$(sed -n -E "s/^$KEYVAR[[:space:]]*=[[:space:]]*//p" .env | sed -E 's/^["'"'"']//; s/["'"'"']$//; s/[[:space:]]+$//' | tr -d '\n')
[ -n "$KEY" ] || { echo "no key for $KEYVAR"; exit 1; }
PUB=$(cat ~/.ssh/id_ed25519.pub)
GPUS='"NVIDIA RTX A5000","NVIDIA RTX 4000 Ada Generation","NVIDIA GeForce RTX 3080 Ti","NVIDIA RTX A4500","NVIDIA GeForce RTX 4070 Ti","NVIDIA L4","NVIDIA RTX A6000","NVIDIA A40","NVIDIA GeForce RTX 4090","NVIDIA GeForce RTX 3090"'
LAST_BAD_IP=""; BAD_COUNT=0
for try in $(seq 1 40); do
  cat > "$PODDIR/pod.json" <<EOT
{"name":"$NAME","imageName":"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
 "gpuTypeIds":[$GPUS],"gpuTypePriority":"availability",
 "gpuCount":1,"containerDiskInGb":80,"volumeInGb":0,"ports":["22/tcp"],"minVCPUPerGPU":12,"minRAMPerGPU":24,
 "cloudType":"COMMUNITY","supportPublicIp":true,"env":{"PUBLIC_KEY":"$PUB"}}
EOT
  R=$(curl -s -m 60 -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" https://rest.runpod.io/v1/pods -d @"$PODDIR/pod.json")
  rm -f "$PODDIR/pod.json"
  if ! echo "$R" | grep -q '"id"'; then echo "try $try: no instance"; sleep 60; continue; fi
  ID=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
  GPU=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin).get('machine',{}).get('gpuTypeId'))")
  PRICE=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin).get('costPerHr'))")
  echo "try $try: created $ID ($GPU, \$$PRICE/h)"
  PORT=""; IP=""
  for i in $(seq 1 36); do
    Q=$(curl -s -m 30 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" https://api.runpod.io/graphql \
      -d "{\"query\":\"query { pod(input:{podId:\\\"$ID\\\"}) { desiredStatus runtime { ports { ip isIpPublic privatePort publicPort } } } }\"}")
    read PORT IP < <(echo "$Q" | python3 -c "
import json,sys
try:
    r=json.load(sys.stdin)['data']['pod']; p=[x for x in (r['runtime'] or {}).get('ports') or [] if x['privatePort']==22 and x['isIpPublic']]
    print(p[0]['publicPort'], p[0]['ip']) if p else print('', '')
except Exception: print('', '')")
    [ -n "$PORT" ] && break; sleep 10
  done
  OK=""
  if [ -n "$PORT" ]; then
    for i in 1 2 3; do
      CHK=$(ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 -p $PORT root@$IP 'bash -s' < scripts/pod/check_egl.sh 2>/dev/null)
      [ -n "$CHK" ] && break; sleep 20
    done
    echo "  $IP:$PORT -> $(echo $CHK | tr '\n' ' ')"
    echo "$CHK" | grep -q CUDA_OK && echo "$CHK" | grep -q EGL_OK && OK=1
  else
    echo "  no public SSH port"
  fi
  if [ -n "$OK" ]; then
    echo $ID > "$PODDIR/pod_id"; echo "$PORT $IP" > "$PODDIR/pod_addr"
    printf '#!/bin/bash\nexec ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -i $HOME/.ssh/id_ed25519 -p %s root@%s "$@"\n' $PORT $IP > "$PODDIR/pod_ssh.sh"
    echo "FOUND $ID $GPU $IP:$PORT"; exit 0
  fi
  curl -s -m 60 -o /dev/null -w "  terminated $ID (http %{http_code})\n" -X DELETE -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods/$ID
  if [ "$IP" = "$LAST_BAD_IP" ]; then BAD_COUNT=$((BAD_COUNT + 1)); else LAST_BAD_IP=$IP; BAD_COUNT=1; fi
  if [ $BAD_COUNT -ge 2 ] && [ -n "$GPU" ]; then GPUS=$(echo "$GPUS" | sed "s/\"$GPU\",\{0,1\}//; s/,\$//"); echo "  dropping $GPU -> [$GPUS]"; fi
  sleep 30
done
echo GAVE_UP
