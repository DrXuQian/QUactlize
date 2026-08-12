#!/usr/bin/env bash
# Independent Marlin scheduler/cooperative decode comparison plus BPC ladder.
# Every invocation uses one binary, one exact-by-construction fixture, one
# tactic/artifact, and one distinct-event-pair kernel-span protocol.
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
printf '[marlin-scheduler] binary=%s\n' "$BIN"
COMMON=(--m=1 --n=4096 --k=4096 --l=1 --g=128 --mode=1
        --alpha=1 --beta=0 --iterations=20 --streamk_exact_fixture)

run_arm() {
  local label="$1"
  local scheduler="${label%%-bpc*}"
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
  grep -Eq "^  \[CUTLASS w4 gs=128 cfg=16x128:128 w16x32 s3 bc0->0 scheduler=${scheduler}\] M=1 +[0-9.]+ us" "$log" \
    || fail "$label did not report the common tactic and scheduler identity"
  local dispositions
  dispositions="$(grep -Ec '^  Disposition: Passed( .*)?$' "$log" || true)"
  [ "$dispositions" -eq 1 ] \
    || fail "$label reported $dispositions Passed dispositions instead of exactly one"
}

validate_marlin_point() {
  local bpc="$1" grid="$2" iters="$3" active="$4" idle="$5"
  local handoffs="$6" max_peers="$7" label="$8"
  local log="$ARTIFACT_ROOT/${label}.log"
  local resident_warps=$((OCCUPANCY_API * 4))

  grep -Eq "^  \[dense marlin decomposition\] real_cu=72 occupancy_api=${OCCUPANCY_API} blocks_per_cu=${bpc} Q=32 Kt=32 G=${grid} I=${iters} active=${active} idle=${idle} handoffs=${handoffs} max_peers=${max_peers} workspace=[0-9]+$" \
    "$log" \
    || fail "Marlin B=${bpc} did not lower to its pinned CU72 stripe geometry"
  grep -Eq "^  \[dense scheduler=marlin\] logical_cta=32 cu=72 occupancy_api=${OCCUPANCY_API} grid=\(${grid},1,1\) physical_cta=${grid} block_threads=128 warps/cta=4 resident_warps/cu=${resident_warps}$" \
    "$log" \
    || fail "Marlin B=${bpc} did not launch the pinned (${grid},1,1) grid"
  grep -Eq "^  \[CUTLASS .* scheduler=marlin\].*Marlin-C valid_elements=[1-9][0-9]* peer_excess=${handoffs} logical_RW=[1-9][0-9]* MODEL-ONLY/not-a-DRAM-counter$" \
    "$log" \
    || fail "Marlin B=${bpc} did not surface its pinned handoff traffic model"
  local lock_repeats
  lock_repeats="$(grep -Ec '^  \[dense marlin lock fingerprint\] repeat=[1-8]/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$' \
    "$log" || true)"
  [ "$lock_repeats" -eq 8 ] \
    || fail "Marlin B=${bpc} lock lifecycle produced $lock_repeats/8 stable bit-exact fingerprints"
  for repeat in {1..8}; do
    local repeat_count
    repeat_count="$(grep -Ec "^  \\[dense marlin lock fingerprint\\] repeat=${repeat}/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$" \
      "$log" || true)"
    [ "$repeat_count" -eq 1 ] \
      || fail "Marlin B=${bpc} fingerprint repeat ${repeat}/8 appeared $repeat_count times"
  done
  grep -Eq '^  \[dense marlin lock protocol\] fixture_identity=a0-exact shape=1x4096x4096 repeats=8 stable=1 all-bitexact=1 same-workspace=1 external-lock-reset=0$' \
    "$log" \
    || fail "Marlin B=${bpc} did not close the repeated same-workspace lock protocol"
  grep -Eq '^  \[dense kernel-span-upper\].*lock-reset-before-start=0$' "$log" \
    || fail "Marlin B=${bpc} unexpectedly used a host lock reset"
}

run_arm non-persistent
run_arm streamk --streamk
# B=1 deliberately carries no --marlin-blocks-per-cu flag.  It is the legacy
# default-path control against which the explicit B=2/4/6 ladder is compared.
run_arm marlin --marlin

# The exact instantiated-kernel cap is runtime evidence, not the literal 6 in
# this fixture's requested ladder.  Derive it from the no-flag B=1 control and
# refuse the whole explicit ladder before launching any illegal point.
mapfile -t OCCUPANCY_VALUES < <(sed -nE \
  's/^  \[dense marlin decomposition\] real_cu=72 occupancy_api=([0-9]+) blocks_per_cu=1 Q=32 Kt=32 .*/\1/p' \
  "$ARTIFACT_ROOT/marlin.log")
[ "${#OCCUPANCY_VALUES[@]}" -eq 1 ] \
  || fail "B=1 did not report exactly one CU72 occupancy_api value"
OCCUPANCY_API="${OCCUPANCY_VALUES[0]}"
[[ "$OCCUPANCY_API" =~ ^[1-9][0-9]*$ ]] \
  || fail "B=1 reported invalid occupancy_api=$OCCUPANCY_API"
printf '[marlin-scheduler] B=1 runtime occupancy_api=%s (exact instantiated-kernel cap)\n' \
  "$OCCUPANCY_API"
for requested in 2 4 6; do
  if [ "$requested" -gt "$OCCUPANCY_API" ]; then
    printf '[marlin-scheduler] NOT RUN: requested B=%d exceeds B=1 occupancy_api=%d\n' \
      "$requested" "$OCCUPANCY_API" >&2
    fail 'BPC ladder contains an illegal point for this exact kernel'
  fi
done

validate_marlin_point 1 72 15 69 3 66 4 marlin
for bpc in 2 4 6; do
  case "$bpc" in
    2) expected=(144 8 128 16 96 4) ;;
    4) expected=(288 4 256 32 224 8) ;;
    6) expected=(432 3 342 90 331 12) ;;
  esac
  run_arm "marlin-bpc${bpc}" --marlin "--marlin-blocks-per-cu=${bpc}"
  validate_marlin_point "$bpc" "${expected[@]}" "marlin-bpc${bpc}"
done

grep -Eq '^  \[dense kernel-span-upper\].*lock-reset-before-start=1$' \
  "$ARTIFACT_ROOT/streamk.log" \
  || fail 'Stream-K timing did not reset its workspace outside the event'
for arm in non-persistent marlin marlin-bpc2 marlin-bpc4 marlin-bpc6; do
  grep -Eq '^  \[dense kernel-span-upper\].*lock-reset-before-start=0$' \
    "$ARTIFACT_ROOT/${arm}.log" \
    || fail "$arm unexpectedly used the Stream-K host reset path"
done

printf '\n[marlin-scheduler] PASS: same fixture/tactic/protocol DP vs Stream-K vs Marlin B={1,2,4,6}; every Marlin lock lifecycle 8/8 stable bit-exact\n'
printf '[marlin-scheduler] artifacts preserved at %s\n' "$ARTIFACT_ROOT"
