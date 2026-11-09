#!/usr/bin/env bash
# ACU (detailed, forced-cold) on single pattern-probe configs: one report per config line.
#   bash tools/pattern_acu.sh DEV OUTDIR < configs
set -uo pipefail
source /workspace/quactlize-decode-gemv-20261108/env.sh
DEV=${1:?dev}; OUT=${2:?outdir}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$DEV
i=0
while read -r line; do
  [ -n "$line" ] || continue
  python3 /workspace/quactlize-decode-gemv-20261108/tools/device_idle.py "$DEV" 3600 > /dev/null || exit 3
  name=$(echo "$line" | tr ' ' '_')
  echo "$line" | acu --set detailed --cache-control all --kernel-name-base demangled \
      -k "regex:frag|rowtile|flat" -s 6 -c 1 -f -o "$OUT/$name" /dev/shm/qgemv/pattern_probe > "$OUT/$name.log" 2>&1 ||
      echo "ACU FAIL $line"
  i=$((i+1))
done
echo "done $i"
