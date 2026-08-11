#!/usr/bin/env bash
# #107b's dense-only PPU gate.  This does not run or update any grouped/MoE
# number: it proves the mixed-input Stream-K wiring, its absolute-K seam, and
# the 128-thread fixup cohort before a future grouped scheduler is attempted.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_streamk_ab
ARTIFACT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-dense-streamk-107b.XXXXXX")"
BUILD_ROOT="$ARTIFACT_ROOT/ppu0010"
BUILD_LOG="$ARTIFACT_ROOT/build.log"
LIST_LOG="$ARTIFACT_ROOT/config.log"
GATE_LOG="$ARTIFACT_ROOT/seam-gate.log"
NP_LOG="$ARTIFACT_ROOT/a0-non-persistent.log"
P_LOG="$ARTIFACT_ROOT/a0-persistent.log"
SK_LOG="$ARTIFACT_ROOT/a0-streamk.log"

fail() {
  printf '[107b] FAIL: %s\n' "$*" >&2
  printf '[107b] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

find_one() {
  local description="$1"
  shift
  local -a hits=()
  mapfile -t hits < <(find "$@" -print)
  if [ "${#hits[@]}" -ne 1 ]; then
    printf '[107b] %s candidates (%d):\n' "$description" "${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]:-<none>}" >&2
    fail "expected exactly one $description"
  fi
  printf '%s\n' "${hits[0]}"
}

run_case() {
  local label="$1" log="$2"
  shift 2
  printf '\n== %s ==\n' "$label"
  if ! "$BIN" "$@" 2>&1 | tee "$log"; then
    fail "$label returned nonzero"
  fi
  grep -q '^  Disposition: Passed$' "$log" \
    || fail "$label did not report a passed numerical check"
  grep -Eq '^  \[dense kernel-span-upper\] n=20 median=[0-9.]+ us .*distinct-event-pairs=20 ' "$log" \
    || fail "$label did not report 20 distinct event-pair kernel spans"
}

require_verify_buckets() {
  local label="$1" log="$2"
  local bucket
  for bucket in DP SK-whole SK-split; do
    [ "$(grep -Ec "^  \\[dense verify bucket=${bucket}\\] tiles=[0-9]+ outputs=[0-9]+ mismatches=[0-9]+ max_abs=[^ ]+ max_rel_sym=[^ ]+ max_half_ulp=[0-9]+ nonfinite=[0-9]+$" "$log")" -eq 1 ] \
      || fail "$label did not report exactly one complete ${bucket} error bucket"
  done
}

# A0 is deliberately diagnostic: the device result that opened this item was
# Failed, and accepting only rc=0 would delete the evidence we are here to
# classify.  Accept only the program's two documented verdict exits, require a
# complete verdict plus all three buckets, and reject crashes or partial logs.
run_diagnostic_case() {
  local label="$1" log="$2"
  shift 2
  printf '\n== %s ==\n' "$label"
  set +e
  "$BIN" "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
    fail "$label exited rc=$rc instead of a numerical Passed/Failed verdict"
  fi
  [ "$(grep -Ec '^  Disposition: (Passed|Failed)$' "$log")" -eq 1 ] \
    || fail "$label did not report exactly one numerical disposition"
  grep -Eq '^  \[dense kernel-span-upper\] n=20 median=[0-9.]+ us .*distinct-event-pairs=20 ' "$log" \
    || fail "$label did not report 20 distinct event-pair kernel spans"
  require_verify_buckets "$label" "$log"
}

printf '[107b] root-sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf '[107b] actlize-sha=%s\n' "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
printf '[107b] artifacts=%s\n' "$ARTIFACT_ROOT"

printf '\n== build isolated ppu001 four-warp target ==\n'
if ! env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 TARGET="$TARGET" \
    QUANT=int4 BENCH_GS=128 "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"; then
  fail 'ppu001 dense Stream-K build failed'
fi

BUILD_MAKE="$(find_one 'Stream-K build.make' "$BUILD_ROOT" -type f \
    -path "*${TARGET}.dir/build.make")"
mapfile -t ARCHS < <(grep -oE -- '-arch=ppu_[0-9]+' "$BUILD_MAKE" | sort -u)
printf '[107b][arch] build.make=%s\n' "$BUILD_MAKE"
printf '[107b][arch] unique hgcc arch flags:'
printf ' %s' "${ARCHS[@]:-<none>}"
printf '\n'
if [ "${#ARCHS[@]}" -ne 1 ] || [ "${ARCHS[0]}" != '-arch=ppu_10' ]; then
  fail 'target must contain only -arch=ppu_10'
fi

BIN="$(find_one 'dense Stream-K binary' "$BUILD_ROOT" -type f \
    -name "$TARGET" -perm -u+x)"
if ! "$BIN" --list_configs 2>&1 | tee "$LIST_LOG"; then
  fail '--list_configs failed'
fi
grep -Eq '^  .*tile 64x128x64  warp 64x32  stages 2$' "$LIST_LOG" \
  || fail 'isolated target does not expose exactly the reviewed tile geometry'
if [ "$(grep -Ec '^  .*tile [0-9]+x[0-9]+x[0-9]+  warp [0-9]+x[0-9]+  stages [0-9]+$' "$LIST_LOG")" -ne 1 ]; then
  fail 'isolated target exposes more or fewer than one tactic row'
fi

# This shape has one output tile and 68 K tiles.  With the scheduler's minimum
# eight-tile slice, forced Stream-K must make 8 ordered contributors.  The
# seams after tiles 9 and 27 are inside gs128 groups (TileK=64), so restarting
# metadata at local K=0 cannot pass this gate.
run_case 'exact absolute-K/fixup seam gate plus repeated-launch lock gate' "$GATE_LOG" \
  --m=64 --n=128 --k=4352 --l=1 --g=128 --mode=1 \
  --alpha=.75 --beta=.5 --iterations=20 --streamk_gate

python3 - "$GATE_LOG" <<'PY' || fail 'exact seam/decomposition audit failed'
import pathlib, re, sys

text = pathlib.Path(sys.argv[1]).read_text()
d = re.search(
    r"\[dense streamk decomposition\] actual=(\w+) real_cu=(\d+) ctas_per_cu=(\d+) "
    r"workers=(\d+) scheduler_workers=(\d+) sk_tiles=(\d+) sk_units=(\d+) "
    r"dp_units=(\d+) units=(\d+) splits=(\d+) separate=(\d+)", text)
if not d:
    raise SystemExit("missing decomposition witness")
actual, cu, ctas, workers, sched, sk_tiles, sk_units, dp, units, splits, separate = d.groups()
nums = list(map(int, (cu, ctas, workers, sched, sk_tiles, sk_units, dp, units, splits, separate)))
cu, ctas, workers, sched, sk_tiles, sk_units, dp, units, splits, separate = nums
if not (actual == "StreamK" and workers == cu * ctas == sched and
        (sk_tiles, sk_units, dp, units, splits, separate) == (1, 8, 0, 8, 1, 0)):
    raise SystemExit(f"wrong decomposition: {d.group(0)}")

g = re.search(
    r"\[dense scheduler=streamk\].*grid=\((\d+),(\d+),(\d+)\) physical_cta=(\d+) "
    r"block_threads=(\d+)", text)
if not g:
    raise SystemExit("missing physical-grid witness")
gx, gy, gz, physical, threads = map(int, g.groups())
if gx * gy * gz != physical or physical != workers or threads != 128:
    raise SystemExit(f"grid/barrier mismatch: {g.group(0)} workers={workers}")

required = (
    "[streamk witness] fixup_work=8 epilogue_work=1 separate_reduction_work=0",
    "[streamk CPU-FP32] outputs=8192 bad=0 bitdiff=0",
    "BIT-EXACT",
    "distinct-event-pairs=20 warmup-event-pairs=1 includes-launch-idle=1 lock-reset-before-start=1",
    "MBU=N/A (StreamK partial-C traffic is per-tile and not yet surfaced)",
)
missing = [x for x in required if x not in text]
if missing:
    raise SystemExit("missing exact gate marker(s): " + ", ".join(missing))
print(f"[107b][gate] PASS workers={workers} decomposition=1x8 witness=8/1/0 grid={gx}x{gy}x{gz}")
PY

# Same binary, geometry, event protocol, and 20-launch median for all three
# dense arms.  This is diagnostic only: the 107b value is mechanism proof, not
# a grouped/MoE speedup and it does not alter C1 or S068.
COMMON=(--m=2048 --n=4096 --k=4096 --l=1 --g=128 --mode=1 --iterations=20)
run_case 'A0 non-persistent control' "$NP_LOG" "${COMMON[@]}"
run_case 'A0 serial-persistent control' "$P_LOG" "${COMMON[@]}" --persistent
run_diagnostic_case 'A0 Stream-K diagnostic' "$SK_LOG" "${COMMON[@]}" --streamk

require_verify_buckets 'A0 non-persistent control' "$NP_LOG"
require_verify_buckets 'A0 serial-persistent control' "$P_LOG"
grep -Eq '^  \[dense verify partition\] DP=[0-9]+ SK-whole=[0-9]+ SK-split=[0-9]+ peer_excess=[0-9]+ qk_cells=[0-9]+ coverage=exact-once$' "$SK_LOG" \
  || fail 'A0 Stream-K did not prove an exact scheduler-derived DP/SK partition'

grep -q 'lock-reset-before-start=0' "$NP_LOG" || fail 'non-persistent timing identity drifted'
grep -q 'lock-reset-before-start=0' "$P_LOG" || fail 'persistent timing identity drifted'
grep -q 'lock-reset-before-start=1' "$SK_LOG" || fail 'Stream-K did not reset locks outside every event'
grep -q 'actual=StreamK' "$SK_LOG" || fail 'A0 Stream-K silently fell back to another decomposition'
grep -q 'MBU=N/A (StreamK partial-C traffic is per-tile and not yet surfaced)' "$SK_LOG" \
  || fail 'A0 Stream-K fabricated a uniform-split C traffic number'

python3 - "$NP_LOG" "$P_LOG" "$SK_LOG" <<'PY' || fail 'A0 median extraction failed'
import pathlib, re, sys

vals = {}
for label, path in zip(("non-persistent", "persistent", "streamk"), sys.argv[1:]):
    text = pathlib.Path(path).read_text()
    m = re.search(r"\[dense kernel-span-upper\] n=20 median=([0-9.]+) us", text)
    if not m:
        raise SystemExit(f"no median for {label}")
    vals[label] = float(m.group(1))
if min(vals.values()) <= 0:
    raise SystemExit(f"non-positive median: {vals}")
print("[107b][A0] kernel-span medians: " +
      " ".join(f"{k}={v:.3f}us" for k, v in vals.items()))
print(f"[107b][A0] persistent/non-persistent={vals['persistent']/vals['non-persistent']:.6f} "
      f"streamk/non-persistent={vals['streamk']/vals['non-persistent']:.6f}")
PY

printf '\n[107b] PASS: dense absolute-K/fixup/worker seam and repeated launches proved on ppu001\n'
printf '[107b] COLLECTED: A0 DP/SK-whole/SK-split error buckets; A0 correctness is the printed disposition, not this script exit\n'
printf '[107b] NOTE: A0 ratios are dense diagnostics; no grouped/MoE result is changed or claimed\n'
printf '[107b] artifacts preserved at %s\n' "$ARTIFACT_ROOT"
