# Sourced by the DINO-WM experiments, whose planning needs far more GPU memory than the others (about 40 GB at batch 10) and
# which die at once on a GPU shared with other jobs.  wait_vram <GB>: returns when the job's GPU has that much free memory.
wait_vram() {
  local need=$1 g=${CUDA_VISIBLE_DEVICES:-0} free
  g=${g%%,*}
  while true; do
    free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%d", $1 / 1024}')
    [ -z "$free" ] && return 0                 # no nvidia-smi: nothing to wait for
    [ "$free" -ge "$need" ] && return 0
    echo "$(date -u +%F' '%T) GPU $g has $free GB free, waiting for $need GB"; sleep 300
  done
}
