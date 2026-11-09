#!/usr/bin/env bash
# Run a command on the first PPU of a list that has no foreign compute process (waits up to 2 h).
#   bash tools/when_idle.sh 0,1 -- CMD ARGS...   (CUDA_VISIBLE_DEVICES is set to the chosen PPU)
set -uo pipefail
DEVS=${1//,/ }; shift; [ "${1:-}" = "--" ] && shift
T=/workspace/quactlize-decode-gemv-20261108/tools
for _ in $(seq 720); do
  for d in $DEVS; do
    if python3 "$T/device_idle.py" "$d" 0 > /dev/null; then
      echo "when_idle: device $d" >&2
      CUDA_VISIBLE_DEVICES=$d exec "$@"
    fi
  done
  sleep 10
done
echo "when_idle: no idle device" >&2; exit 3
