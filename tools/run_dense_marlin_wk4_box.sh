#!/usr/bin/env bash
# Classic-aligned dense Marlin decode: exact WK4 artifact + 2N x 4K CTA.
# This target is Marlin-only.  It sweeps only scheduler blocks_per_cu; B values
# above Gemm::maximum_active_blocks() are reported NOT RUN and the binary's
# fail-closed upper bound is checked without launching a kernel.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_marlin_wk4_ab
ARTIFACT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-dense-marlin-wk4.XXXXXX")"
BUILD_ROOT="$ARTIFACT_ROOT/ppu0010"
BUILD_LOG="$ARTIFACT_ROOT/build.log"

fail() {
  printf '[marlin-wk4] FAIL: %s\n' "$*" >&2
  printf '[marlin-wk4] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

find_one() {
  local description="$1"; shift
  local -a hits=()
  mapfile -t hits < <(find "$@" -print)
  if [ "${#hits[@]}" -ne 1 ]; then
    printf '[marlin-wk4] %s candidates (%d):\n' "$description" "${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]:-<none>}" >&2
    fail "expected exactly one $description"
  fi
  printf '%s\n' "${hits[0]}"
}

printf '[marlin-wk4] root-sha=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
printf '[marlin-wk4] actlize-sha=%s\n' "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
printf '[marlin-wk4] artifacts=%s\n' "$ARTIFACT_ROOT"

if ! env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 TARGET="$TARGET" \
    QUANT=int4 BENCH_GS=128 "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"; then
  fail 'classic-aligned Marlin target failed to build'
fi
BIN="$(find_one 'classic-aligned Marlin binary' "$BUILD_ROOT" -type f \
    -name "$TARGET" -perm -u+x)"
printf '[marlin-wk4] binary=%s\n' "$BIN"

COMMON=(--marlin --streamk_exact_fixture --m=1 --n=4096 --k=4096
        --l=1 --g=128 --mode=1 --alpha=1 --beta=0 --iterations=20)

run_point() {
  local bpc="$1" label="bpc${1}" log="$ARTIFACT_ROOT/bpc${1}.log"
  local -a bflag=()
  # B=1 is the default-compatibility arm.  Omitting the option is deliberate:
  # a default that merely equals an explicit override is not proof that the
  # historical call path stayed unchanged.
  if [ "$bpc" -ne 1 ]; then bflag=("--marlin-blocks-per-cu=$bpc"); fi
  printf '\n== classic-aligned Marlin B=%d ==\n' "$bpc"
  if ! "$BIN" "${COMMON[@]}" "${bflag[@]}" 2>&1 | tee "$log"; then
    fail "B=$bpc returned nonzero"
  fi
  grep -Fq '[dense-marlin-aligned] scheduler=marlin-only topology=1Mx2Nx4K cta_threads=256 output_cohort_threads=64 warp_k_extent=32 warp_k_cohorts=4 tile=16x128x128 warp=16x64x32 stages=4 bits=4 fold=1 artifact_tile_k=64 artifact_axis=WarpK32' "$log" \
    || fail "B=$bpc did not print the exact compiled topology"
  grep -Eq '^  \[dense marlin aligned artifact\] batch=0 bytes=8388608 placement=WK4 artifact_tile_k=64 roundtrip_bad=0/16777216$' "$log" \
    || fail "B=$bpc did not consume and roundtrip the WK4 artifact"
  grep -Eq '^  \[streamk fixture exactness\] fixture=a0-exact shape=1x4096x4096 .* -> ORDER-INDEPENDENT\+FP16-EXACT$' "$log" \
    || fail "B=$bpc did not use the exact decode fixture"
  grep -Eq "^  \\[dense marlin decomposition\\] real_cu=[0-9]+ occupancy_api=[0-9]+ blocks_per_cu=${bpc} Q=32 Kt=32 G=[0-9]+ I=[0-9]+ active=[0-9]+ idle=[0-9]+ handoffs=[0-9]+ max_peers=[0-9]+ workspace=[0-9]+$" "$log" \
    || fail "B=$bpc did not report its decomposition"
  grep -Eq '^  \[dense scheduler=marlin\] logical_cta=32 cu=[0-9]+ occupancy_api=[0-9]+ grid=\([0-9]+,1,1\) physical_cta=[0-9]+ block_threads=256 warps/cta=8 resident_warps/cu=[0-9]+$' "$log" \
    || fail "B=$bpc did not launch the 256-thread/8-warp kernel"
  grep -Eq '^  \[dense kernel-span-upper\] n=20 median=[0-9.]+ us .*distinct-event-pairs=20 .*includes-launch-idle=1 .*lock-reset-before-start=0$' "$log" \
    || fail "B=$bpc did not report 20 independent kernel spans"
  local repeats
  repeats="$(grep -Ec '^  \[dense marlin lock fingerprint\] repeat=[1-8]/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$' "$log" || true)"
  [ "$repeats" -eq 8 ] || fail "B=$bpc produced $repeats/8 stable lock fingerprints"
  grep -Eq '^  \[dense marlin lock protocol\] fixture_identity=a0-exact shape=1x4096x4096 repeats=8 stable=1 all-bitexact=1 same-workspace=1 external-lock-reset=0$' "$log" \
    || fail "B=$bpc did not close the lock protocol"
}

run_point 1
mapfile -t CAPS < <(sed -nE \
  's/^  \[dense marlin decomposition\] real_cu=[0-9]+ occupancy_api=([0-9]+) blocks_per_cu=1 Q=32 Kt=32 .*/\1/p' \
  "$ARTIFACT_ROOT/bpc1.log")
[ "${#CAPS[@]}" -eq 1 ] || fail 'B=1 did not report exactly one occupancy_api value'
CAP="${CAPS[0]}"
[[ "$CAP" =~ ^[1-9][0-9]*$ ]] || fail "invalid occupancy_api=$CAP"
printf '[marlin-wk4] exact instantiated-kernel B cap=%s\n' "$CAP"

for bpc in 2 4 6; do
  if [ "$bpc" -gt "$CAP" ]; then
    printf '[marlin-wk4] NOT RUN: B=%d exceeds Gemm::maximum_active_blocks()=%d\n' \
      "$bpc" "$CAP" | tee "$ARTIFACT_ROOT/bpc${bpc}.not-run"
    continue
  fi
  run_point "$bpc"
done

# Prove the runtime guard itself, using one value beyond the discovered cap.
# run() rejects this before gemm.initialize()/launch; a successful process or
# the absence of the exact cap in the diagnostic is a fail-open ABI.
ILLEGAL=$((CAP + 1))
set +e
"$BIN" "${COMMON[@]}" "--marlin-blocks-per-cu=$ILLEGAL" \
  >"$ARTIFACT_ROOT/illegal-bpc.log" 2>&1
illegal_rc=$?
set -e
[ "$illegal_rc" -ne 0 ] || fail "illegal B=$ILLEGAL unexpectedly returned success"
grep -Fq -- "--marlin-blocks-per-cu=$ILLEGAL is outside the exact kernel occupancy range 1..$CAP" \
  "$ARTIFACT_ROOT/illegal-bpc.log" \
  || fail "illegal B=$ILLEGAL was not rejected by the exact runtime cap"

printf '\n[marlin-wk4] PASS: classic-aligned WK4 target built; supported B points passed exact output + 8-launch locks; over-cap B stayed NOT RUN\n'
printf '[marlin-wk4] artifacts preserved at %s\n' "$ARTIFACT_ROOT"
