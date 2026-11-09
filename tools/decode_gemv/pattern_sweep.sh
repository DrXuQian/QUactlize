#!/usr/bin/env bash
# Access-pattern + size sweep under Asys on an idle PPU (GPU kernel durations, cold ring).
#   bash tools/pattern_sweep.sh DEVS(e.g. 0,1) OUTDIR [CONFIGS]   (default: the generated step-1 list)
set -uo pipefail
source /workspace/quactlize-decode-gemv-20261108/env.sh
DEV=${1:-0}; OUT=${2:?outdir}
P=/dev/shm/qgemv/pattern_probe
T=/workspace/quactlize-decode-gemv-20261108/tools
mkdir -p "$OUT"
cfg="$OUT/configs.txt"
if [ -n "${3:-}" ]; then cp "$3" "$cfg"; else
: > "$cfg"
# A. Access pattern on Q4 low-plane shapes: 51200x5120 (rows 1280) and 5120x25600 (rows 6400).
for shape in "51200 1280" "5120 6400"; do
  read -r n rows <<<"$shape"
  for t in 2 4 8; do
    for w in "4 2" "2 4" "1 8" "8 1" "4 1" "2 8"; do
      read -r wn wk <<<"$w"
      for u in 1 2 4 8; do echo "frag $n $rows $t $wn $wk $u" >> "$cfg"; done
    done
  done
  for wk in 4 8 16; do for u in 1 2 4 8; do echo "rows $n $rows 0 0 $wk $u" >> "$cfg"; done; done
  for c in "8 64" "8 128" "8 256" "4 256" "8 1024" "8 4096"; do
    read -r t g <<<"$c"; echo "flat $n $rows $t $g 1 4" >> "$cfg"
  done
done
# B. Size sweep (decode-sized reads): flat with several grids, frag T=2 1x8 / 2x8.
for rows in 8 256 512 1024 2048 4096 8192 16384; do
  for g in 64 128 256 512; do echo "flat 1024 $rows 8 $g 1 4" >> "$cfg"; done
  for u in 2 4 8; do echo "frag 1024 $rows 2 1 8 $u" >> "$cfg"; echo "frag 1024 $rows 2 2 8 $u" >> "$cfg"; done
done
fi
# Wait (up to 2 h) for the first of the listed devices with no foreign compute process.
DEVS=${DEV//,/ }
DEV=""
for _ in $(seq 720); do
  for d in $DEVS; do
    if python3 "$T/device_idle.py" "$d" 0 > /dev/null; then DEV=$d; break 2; fi
  done
  sleep 10
done
[ -n "$DEV" ] || exit 3
export CUDA_VISIBLE_DEVICES=$DEV
echo "device $DEV" > "$OUT/device.txt"
asys profile -t hggc --force-overwrite true -o "$OUT/trace" "$P" < "$cfg" > "$OUT/probe.out" 2>&1
python3 "$T/device_idle.py" "$DEV" 0 > "$OUT/after.txt" 2>&1
asys export --force-overwrite true -o "$OUT/trace.sqlite" "$OUT"/trace*.asysrep > /dev/null 2>&1
python3 "$T/pattern_asys.py" --sqlite "$OUT/trace.sqlite" --stdout "$OUT/probe.out" --json "$OUT/kernels.jsonl" > "$OUT/kernels.txt"
rm -f "$OUT/trace.sqlite"
echo done >> "$OUT/after.txt"
