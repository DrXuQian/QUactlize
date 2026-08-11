#!/usr/bin/env bash
# Independent Marlin scheduler/cooperative decode comparison.  All three arms
# use one binary, one exact-by-construction fixture, one tactic/artifact, and
# one distinct-event-pair kernel-span protocol.  Only the scheduler changes.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_marlin_ab
ARTIFACT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-dense-marlin.XXXXXX")"
BUILD_ROOT="$ARTIFACT_ROOT/ppu0010"
BUILD_LOG="$ARTIFACT_ROOT/build.log"

fail() {
  printf '[marlin-scheduler] FAIL: %s\n' "$*" >&2
  printf '[marlin-scheduler] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

find_one() {
  local description="$1"
  shift
  local -a hits=()
  mapfile -t hits < <(find "$@" -print)
  if [ "${#hits[@]}" -ne 1 ]; then
    printf '[marlin-scheduler] %s candidates (%d):\n' "$description" "${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]:-<none>}" >&2
    fail "expected exactly one $description"
  fi
  printf '%s\n' "${hits[0]}"
}

printf '[marlin-scheduler] root-sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf '[marlin-scheduler] actlize-sha=%s\n' "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
printf '[marlin-scheduler] artifacts=%s\n' "$ARTIFACT_ROOT"

printf '\n== build isolated ppu001 DP/Stream-K/Marlin target ==\n'
if ! env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 TARGET="$TARGET" \
    QUANT=int4 BENCH_GS=128 "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"; then
  fail 'ppu001 Marlin scheduler target failed to build'
fi

BIN="$(find_one 'Marlin comparison binary' "$BUILD_ROOT" -type f \
    -name "$TARGET" -perm -u+x)"
COMMON=(--m=1 --n=4096 --k=4096 --l=1 --g=128 --mode=1
        --alpha=1 --beta=0 --iterations=20 --streamk_exact_fixture)

run_arm() {
  local label="$1"
  shift
  local log="$ARTIFACT_ROOT/${label}.log"
  printf '\n== decode arm: %s ==\n' "$label"
  if ! "$BIN" "${COMMON[@]}" "$@" 2>&1 | tee "$log"; then
    fail "$label arm returned nonzero"
  fi
  grep -Eq '^  \[streamk fixture exactness\] fixture=a0-exact shape=1x4096x4096 .* -> ORDER-INDEPENDENT\+FP16-EXACT$' "$log" \
    || fail "$label did not use the exact common decode fixture"
  grep -Eq '^  \[dense kernel-span-upper\] n=20 median=[0-9.]+ us .*distinct-event-pairs=20 .*includes-launch-idle=1 ' "$log" \
    || fail "$label did not report 20 distinct event-pair kernel spans"
  grep -Eq "^  \[CUTLASS w4 gs=128 cfg=16x128:128 w16x32 s3 bc0->0 scheduler=${label}\] M=1 +[0-9.]+ us" "$log" \
    || fail "$label did not report the common tactic and scheduler identity"
}

run_arm non-persistent
run_arm streamk --streamk
run_arm marlin --marlin

grep -Eq '^  \[dense marlin decomposition\] real_cu=[0-9]+ occupancy_api=[0-9]+ Q=32 Kt=32 G=[0-9]+ I=[0-9]+ active=[0-9]+ handoffs=[1-9][0-9]* workspace=[0-9]+$' \
  "$ARTIFACT_ROOT/marlin.log" \
  || fail 'Marlin arm did not exercise a real cross-CTA stripe'
grep -Eq '^  \[CUTLASS .* scheduler=marlin\].*Marlin-C valid_elements=[1-9][0-9]* peer_excess=[1-9][0-9]* logical_RW=[1-9][0-9]* MODEL-ONLY/not-a-DRAM-counter$' \
  "$ARTIFACT_ROOT/marlin.log" \
  || fail 'Marlin arm did not surface its predicated FP32 cooperative traffic'
grep -Eq '^  \[dense kernel-span-upper\].*lock-reset-before-start=1$' \
  "$ARTIFACT_ROOT/streamk.log" \
  || fail 'Stream-K timing did not reset its workspace outside the event'
for arm in non-persistent marlin; do
  grep -Eq '^  \[dense kernel-span-upper\].*lock-reset-before-start=0$' \
    "$ARTIFACT_ROOT/${arm}.log" \
    || fail "$arm unexpectedly used the Stream-K host reset path"
done

printf '\n[marlin-scheduler] PASS: same fixture/tactic/protocol DP vs Stream-K vs Marlin\n'
printf '[marlin-scheduler] artifacts preserved at %s\n' "$ARTIFACT_ROOT"
