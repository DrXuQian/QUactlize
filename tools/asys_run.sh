#!/usr/bin/env bash
# ONE COMMAND: capture under asys, export, check, and read kernel-only time.
#
# WHY THIS EXISTS. The three steps were being run by hand, and each one has a way of failing that looks like
# success from the outside. `asys -o X` writes X.asysrep, not a database, so the reader met "unable to open
# database file" and named a line of Python. A capture can complete, weigh 18 MB, and hold ZERO kernel activities
# because device timestamps could not be calibrated -- the size comes from host-API rows alone. And a run whose
# arguments were wrong produced "all experts empty; nothing to measure" followed by "No kernels were profiled":
# two correct refusals, neither naming the cause. Every one of those cost a round trip.
#
#   tools/asys_run.sh out ./bin 256 8 512 2048 32 3 8                     the whole table
#   MOE_ONLY='i4 32x128:128 w32x32 s3 bc0->0' tools/asys_run.sh out ./bin 256 4096 512 2048 32 4 8
#
# BENCH_FLOOR=0 IS THE DEFAULT HERE, and only here. The launch-rate probe issues 201 nop launches, which under a
# profiler is 201 of 222 activities describing the probe rather than the kernel. Set BENCH_FLOOR=1 to keep it.
#
# ASYS=<path> overrides the binary. The capture flags are the form this box uses; export takes no --type.
set -Eeuo pipefail

ASYS="${ASYS:-$(command -v asys || true)}"
[ -n "$ASYS" ] || { echo "[asys-run] no asys on PATH; set ASYS=<path>" >&2; exit 2; }

OUT="${1:?usage: asys_run.sh <outname> <binary> [args...]}"; shift
BIN="${1:?usage: asys_run.sh <outname> <binary> [args...]}"; shift
[ -x "$BIN" ] || { echo "[asys-run] $BIN is not executable" >&2; exit 2; }

REP="${OUT}.asysrep"; DB="${OUT}.sqlite"; LOG="${OUT}.log"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[asys-run] capture -> $REP"
echo "[asys-run]   MOE_ONLY=${MOE_ONLY-<all configs>}  BENCH_FLOOR=${BENCH_FLOOR:-0}"
"$ASYS" profile --hggc-memory-usage=true -t hggc,hgtx,acdnn,acblas -f true -o "$OUT" \
  env BENCH_FLOOR="${BENCH_FLOOR:-0}" MOE_REPS="${MOE_REPS:-1}" MOE_VERBOSE=1 \
      ${MOE_ONLY+MOE_ONLY="$MOE_ONLY"} "$BIN" "$@" 2>&1 | tee "$LOG"

# THE HARNESS CAN REFUSE AND STILL EXIT THROUGH THE PROFILER. "all experts empty" means the arguments described no
# work at all; without this the next message is "No kernels were profiled", which reads like a profiler fault.
if grep -q "all experts empty" "$LOG"; then
  echo "[asys-run] the harness measured nothing: 'all experts empty'. Check the shape arguments," >&2
  echo "[asys-run] not the profiler -- argv[1] is the expert count and a non-numeric value becomes 0." >&2
  exit 1
fi
if [ -n "${MOE_ONLY-}" ] && ! grep -qF "$MOE_ONLY" "$LOG"; then
  echo "[asys-run] MOE_ONLY='$MOE_ONLY' never appeared in the run's output." >&2
  echo "[asys-run] It is a SUBSTRING match, spaces included; a typo profiles nothing and says nothing." >&2
  exit 1
fi

[ -f "$REP" ] || { echo "[asys-run] $REP was not written; the capture failed" >&2; exit 1; }
echo "[asys-run] export -> $DB"
"$ASYS" export -o "$DB" "$REP"
[ -f "$DB" ] || { echo "[asys-run] export produced no $DB" >&2; exit 1; }

# CHECK THE KERNEL TABLE BEFORE READING. An empty one is indistinguishable from a good capture until someone looks,
# and it is the shape a calibration failure takes: the report is large because the host-API rows are all there.
N="$(sqlite3 "$DB" "SELECT COUNT(*) FROM HGPTI_ACTIVITY_KIND_KERNEL;" 2>/dev/null || echo 0)"
echo "[asys-run] kernel activities: $N"
if [ "$N" -eq 0 ]; then
  echo "[asys-run] ZERO kernel activities: this capture cannot answer anything about kernel time." >&2
  echo "[asys-run] Look for 'can't find time calibration info for device id' above. Device activities are" >&2
  echo "[asys-run] dropped when calibration fails, so the report can be large and still hold no timing." >&2
  exit 1
fi

echo
python3 "$ROOT/tools/asys_kernel_time.py" "$DB" --log "$LOG"
