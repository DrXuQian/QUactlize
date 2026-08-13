#!/usr/bin/env bash
# Classic-aligned dense Marlin decode: standalone classic format + 2N x 4K.
# This target is Marlin-only.  It sweeps only scheduler blocks_per_cu; B values
# above Gemm::maximum_active_blocks() are reported NOT RUN and the binary's
# fail-closed upper bound is checked without launching a kernel.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_ARGV=("$0" "$@")
TARGET=test_lowbit_dense_marlin_wk4_ab
ARTIFACT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-dense-marlin-wk4.XXXXXX")"
BUILD_ROOT="$ARTIFACT_ROOT/ppu0010"
BUILD_LOG="$ARTIFACT_ROOT/build.log"
RUNNER_LOG="$ARTIFACT_ROOT/runner.log"
COMMANDS_LOG="$ARTIFACT_ROOT/commands.jsonl"
SUBMODULE_STATUS_FILE="$ARTIFACT_ROOT/submodule-status.txt"
PROVENANCE_TOOL="$ROOT/tools/write_box_run_provenance.py"
IDENTITY_PROBE_TOOL="$ROOT/tools/probe_box_identity.py"
POLICY="$ROOT/dev/fold_derivation/BOX_RUN_PREREGISTRATION.md"
RUN_IDENTITY_FILE="$ARTIFACT_ROOT/run-identity.json"
IDENTITY_PROBE_FILE="$ARTIFACT_ROOT/identity-probe.json"

# Preserve the complete runner stream inside the bundle instead of relying on
# an operator-side tee.  The screen remains a live copy of the same bytes.
exec 3>&1 4>&2
exec > >(tee "$RUNNER_LOG") 2>&1
RUNNER_TEE_PID=$!
RUNNER_STREAM_CLOSED=0

finish_runner_stream() {
  if [ "$RUNNER_STREAM_CLOSED" -eq 0 ]; then
    exec 1>&3 2>&4
    wait "$RUNNER_TEE_PID"
    RUNNER_STREAM_CLOSED=1
  fi
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ] && [ -n "${BIN_SHA:-}" ] && [ -s "$COMMANDS_LOG" ] \
      && [ -s "$RUN_IDENTITY_FILE" ]; then
    printf '[marlin-wk4] runner_exit_status=%d\n' "$rc"
    python3 "$PROVENANCE_TOOL" write \
      --output "$ARTIFACT_ROOT/provenance.json" \
      --root-sha "$ROOT_SHA" --root-status clean \
      --submodule-status-file "$SUBMODULE_STATUS_FILE" \
      --actlize-sha "$ACTLIZE_SHA" --binary-sha256 "$BIN_SHA" \
      --device-model "$QUACTLIZE_BOX_DEVICE_MODEL" \
      --pci-identity "$QUACTLIZE_BOX_PCI_IDENTITY" \
      --driver-version "$QUACTLIZE_BOX_DRIVER_VERSION" \
      --sdk-compiler-identity "$QUACTLIZE_BOX_SDK_COMPILER_IDENTITY" \
      --identity-probe-file "$IDENTITY_PROBE_FILE" \
      --run-identity-file "$RUN_IDENTITY_FILE" \
      --commands-file "$COMMANDS_LOG" --runner-exit-status "$rc" \
      --protocol-sample-count "$SAMPLES" -- "${ORIGINAL_ARGV[@]}" || true
  fi
  finish_runner_stream
  exit "$rc"
}
trap on_exit EXIT

fail() {
  printf '[marlin-wk4] FAIL: %s\n' "$*" >&2
  printf '[marlin-wk4] artifacts preserved at %s\n' "$ARTIFACT_ROOT" >&2
  exit 1
}

record_command() {
  local role="$1" rc="$2"; shift 2
  python3 "$PROVENANCE_TOOL" record --path "$COMMANDS_LOG" \
    --role "$role" --exit-status "$rc" -- "$@"
}

identity_probe_get() {
  python3 "$IDENTITY_PROBE_TOOL" get --file "$IDENTITY_PROBE_FILE" \
    --field "$1" --part "$2"
}

resolve_box_identity() {
  local -a cmd=(python3 "$IDENTITY_PROBE_TOOL" resolve --output "$IDENTITY_PROBE_FILE")
  local rc
  set +e
  "${cmd[@]}"
  rc=$?
  set -e
  record_command box-identity-probe "$rc" "${cmd[@]}"
  [ "$rc" -eq 0 ] || fail 'automatic box identity probe failed; no kernel was built or run'

  QUACTLIZE_BOX_DEVICE_MODEL="$(identity_probe_get device_model value)"
  QUACTLIZE_BOX_PCI_IDENTITY="$(identity_probe_get pci_identity value)"
  QUACTLIZE_BOX_DRIVER_VERSION="$(identity_probe_get driver_version value)"
  QUACTLIZE_BOX_SDK_COMPILER_IDENTITY="$(identity_probe_get sdk_compiler_identity value)"
  QUACTLIZE_BOX_DEVICE_MODEL_SOURCE="$(identity_probe_get device_model source)"
  QUACTLIZE_BOX_PCI_IDENTITY_SOURCE="$(identity_probe_get pci_identity source)"
  QUACTLIZE_BOX_DRIVER_VERSION_SOURCE="$(identity_probe_get driver_version source)"
  QUACTLIZE_BOX_SDK_COMPILER_IDENTITY_SOURCE="$(identity_probe_get sdk_compiler_identity source)"
  export QUACTLIZE_BOX_DEVICE_MODEL QUACTLIZE_BOX_PCI_IDENTITY \
    QUACTLIZE_BOX_DRIVER_VERSION QUACTLIZE_BOX_SDK_COMPILER_IDENTITY
}

verify_source_identity() {
  local current_root current_actlize current_submodules
  if [ -n "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]; then
    fail 'final source identity check found a dirty root or submodule tree'
  fi
  current_root="$(git -C "$ROOT" rev-parse HEAD)"
  current_actlize="$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
  [ "$current_root" = "$ROOT_SHA" ] \
    || fail "root HEAD changed during the run: $ROOT_SHA -> $current_root"
  [ "$current_actlize" = "$ACTLIZE_SHA" ] \
    || fail "actlize HEAD changed during the run: $ACTLIZE_SHA -> $current_actlize"
  current_submodules="$(git -C "$ROOT" submodule status --recursive)"
  cmp -s <(printf '%s\n' "$current_submodules") "$SUBMODULE_STATUS_FILE" \
    || fail 'recursive submodule status changed during the run'
  printf '[marlin-wk4] final-source-identity=EXACT root=%s actlize=%s\n' \
    "$current_root" "$current_actlize"
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

[ "$#" -eq 0 ] || fail 'this runner accepts no positional arguments'

if [ -n "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]; then
  fail 'source tree is dirty; commit/stash every root and submodule change'
fi
git -C "$ROOT" submodule status --recursive >"$SUBMODULE_STATUS_FILE"
if grep -Eq '^[+\-U]' "$SUBMODULE_STATUS_FILE"; then
  cat "$SUBMODULE_STATUS_FILE" >&2
  fail 'recursive submodule checkout differs from the recorded gitlink'
fi

ROOT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
ACTLIZE_SHA="$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
SAMPLES="$(python3 "$PROVENANCE_TOOL" policy-sample-count --policy "$POLICY" --kind dense)"
[ "$SAMPLES" -eq 20 ] || fail "dense harness supports exactly 20 samples, policy requests $SAMPLES"
resolve_box_identity
printf '[marlin-wk4] root-sha=%s\n' "$ROOT_SHA"
printf '[marlin-wk4] root-status=clean\n'
printf '[marlin-wk4] actlize-sha=%s\n' "$ACTLIZE_SHA"
printf '[marlin-wk4] device-model=%s pci=%s driver=%s sdk=%s\n' \
  "$QUACTLIZE_BOX_DEVICE_MODEL" "$QUACTLIZE_BOX_PCI_IDENTITY" \
  "$QUACTLIZE_BOX_DRIVER_VERSION" "$QUACTLIZE_BOX_SDK_COMPILER_IDENTITY"
printf '[marlin-wk4] identity-sources device_model=%s pci_identity=%s driver_version=%s sdk_compiler_identity=%s\n' \
  "$QUACTLIZE_BOX_DEVICE_MODEL_SOURCE" "$QUACTLIZE_BOX_PCI_IDENTITY_SOURCE" \
  "$QUACTLIZE_BOX_DRIVER_VERSION_SOURCE" "$QUACTLIZE_BOX_SDK_COMPILER_IDENTITY_SOURCE"
printf '[marlin-wk4] artifacts=%s\n' "$ARTIFACT_ROOT"

# The standalone local evidence is an admission control, not a timing cell.
# Its host/compile-time oracles cannot all run on this box's nvcc/GCC13 stack,
# so consume the exact seven-line result-SHA evidence instead of pretending it
# was freshly executed here.  ci/check_l143_wk4_committed_evidence.py owns its
# regeneration in the local tier.  Absence of either authority is VOID.
LOCAL_EVIDENCE_LOG="$ARTIFACT_ROOT/local-evidence-admission.log"
LOCAL_GATE=(python3 "$ROOT/ci/check_dense_marlin_wk4_target.py")
set +e
"${LOCAL_GATE[@]}" 2>&1 | tee "$LOCAL_EVIDENCE_LOG"
local_gate_rc=${PIPESTATUS[0]}
set -e
record_command standalone-static-target "$local_gate_rc" "${LOCAL_GATE[@]}"
[ "$local_gate_rc" -eq 0 ] || fail 'standalone static target admission failed'

LOCAL_EVIDENCE_PATH=dev/fold_derivation/l143_standalone_marlin.expected.txt
LOCAL_ORACLE=(git -C "$ROOT" show "$ROOT_SHA:$LOCAL_EVIDENCE_PATH")
printf '[marlin-wk4] local-evidence=committed-standalone-oracle source-sha=%s path=%s fresh-box-execution=0\n' \
  "$ROOT_SHA" "$LOCAL_EVIDENCE_PATH" | tee -a "$LOCAL_EVIDENCE_LOG"
set +e
"${LOCAL_ORACLE[@]}" 2>&1 | tee -a "$LOCAL_EVIDENCE_LOG"
local_oracle_rc=${PIPESTATUS[0]}
set -e
record_command committed-standalone-evidence "$local_oracle_rc" "${LOCAL_ORACLE[@]}"
[ "$local_oracle_rc" -eq 0 ] || fail 'committed standalone evidence is absent from the result SHA'
[ "$(grep -Fxc '[dense-marlin-wk4] PASS: standalone format/collective/scheduler/kernel wired; generic WK4 compatibility absent; ten structural plants rejected' "$LOCAL_EVIDENCE_LOG" || true)" -eq 2 ] \
  || fail 'standalone admission lacks one fresh and one result-SHA static contract PASS'
grep -Fxq '[L167] PASS: independent classic/direct and Awesome-CuTe/permutation anchors agree; asymmetric provider, byte, inverse, and negative controls proved' "$LOCAL_EVIDENCE_LOG" \
  || fail 'standalone admission lacks the independently anchored classic format proof'
grep -Fxq '[l168:runner] positive=PASS negative_controls=3/3_RED result=PASS' "$LOCAL_EVIDENCE_LOG" \
  || fail 'standalone admission lacks the exact pipeline-cadence closure'
grep -Fxq '[l169] PASS: generated-unit shape instantiates standalone Marlin collective/scheduler/kernel; only the two explicit nvcc/PPU environmental diagnostics remain' "$LOCAL_EVIDENCE_LOG" \
  || fail 'standalone admission lacks the generated standalone type closure'
grep -Fxq '[l170:runner] positive=PASS negative_controls=7/7_RED result=PASS' "$LOCAL_EVIDENCE_LOG" \
  || fail 'standalone admission lacks the scheduler lifecycle closure'
grep -Fxq '[classic-156] PASS: exact one-launch shape, source/tool/binary identity and full ACU capture are fail-closed' "$LOCAL_EVIDENCE_LOG" \
  || fail 'standalone admission lacks the profile capture contract'
grep -Fxq '[l143] PASS: standalone Marlin format + cadence + generated type + scheduler lifecycle; generic WK4 compatibility is absent; no device result claimed' "$LOCAL_EVIDENCE_LOG" \
  || fail 'standalone admission lacks the exact aggregate conclusion'

BUILD_CMD=(env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 TARGET="$TARGET"
  QUANT=int4 BENCH_GS=128 "$ROOT/build.sh")
set +e
"${BUILD_CMD[@]}" 2>&1 | tee "$BUILD_LOG"
build_rc=${PIPESTATUS[0]}
set -e
record_command device-build "$build_rc" "${BUILD_CMD[@]}"
if [ "$build_rc" -ne 0 ]; then
  fail 'classic-aligned Marlin target failed to build'
fi
BIN="$(find_one 'classic-aligned Marlin binary' "$BUILD_ROOT" -type f \
    -name "$TARGET" -perm -u+x)"
BIN_SHA="$(sha256sum "$BIN" | awk '{print $1}')"
printf '[marlin-wk4] binary=%s\n' "$BIN"
printf '[marlin-wk4] binary-sha256=%s\n' "$BIN_SHA"
python3 "$PROVENANCE_TOOL" write-identity \
  --output "$RUN_IDENTITY_FILE" \
  --root-sha "$ROOT_SHA" --submodule-status-file "$SUBMODULE_STATUS_FILE" \
  --actlize-sha "$ACTLIZE_SHA" --binary-sha256 "$BIN_SHA" \
  --device-model "$QUACTLIZE_BOX_DEVICE_MODEL" \
  --pci-identity "$QUACTLIZE_BOX_PCI_IDENTITY" \
  --driver-version "$QUACTLIZE_BOX_DRIVER_VERSION" \
  --sdk-compiler-identity "$QUACTLIZE_BOX_SDK_COMPILER_IDENTITY" \
  --identity-probe-file "$IDENTITY_PROBE_FILE" \
  --protocol-sample-count "$SAMPLES" >/dev/null

COMMON=(--marlin --streamk_exact_fixture --m=1 --n=4096 --k=4096
        --l=1 --g=128 --mode=1 --alpha=1 --beta=0 --iterations="$SAMPLES")

run_point() {
  local bpc="$1" label="bpc${1}" log="$ARTIFACT_ROOT/bpc${1}.log"
  local -a bflag=()
  # B=1 is the default-compatibility arm.  Omitting the option is deliberate:
  # a default that merely equals an explicit override is not proof that the
  # historical call path stayed unchanged.
  if [ "$bpc" -ne 1 ]; then bflag=("--marlin-blocks-per-cu=$bpc"); fi
  if [ "$bpc" -eq 1 ]; then
    [ "${#bflag[@]}" -eq 0 ] || fail 'B=1 default arm unexpectedly carries an override'
    printf '[marlin-wk4] B=1 path=historical-default explicit-blocks-per-cu-override=0\n'
  else
    [ "${#bflag[@]}" -eq 1 ] \
      || fail "B=$bpc diagnostic arm lacks its unique override"
  fi
  printf '\n== classic-aligned Marlin B=%d ==\n' "$bpc"
  local -a cmd=("$BIN" "${COMMON[@]}" "${bflag[@]}")
  set +e
  "${cmd[@]}" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  record_command "dense-wk4-bpc${bpc}" "$rc" "${cmd[@]}"
  if [ "$rc" -ne 0 ]; then
    fail "B=$bpc returned nonzero"
  fi
  grep -Fq '[dense-marlin-aligned] scheduler=marlin-only topology=1Mx2Nx4K cta_threads=256 output_cohort_threads=64 warp_k_extent=32 warp_k_cohorts=4 tile=16x128x128 warp=16x64x32 stages=4 bits=4 fold=1 artifact=classic-marlin-u32 scale=classic-gs128-permuted load=cp.async consumer_axis=WarpK32' "$log" \
    || fail "B=$bpc did not print the exact compiled topology"
  grep -Eq '^  \[dense marlin aligned artifact\] batch=0 bytes=8388608 placement=classic-marlin-u32 scale=classic-gs128-permuted consumer_axis=WarpK32 roundtrip_bad=0/16777216$' "$log" \
    || fail "B=$bpc did not consume and roundtrip the standalone classic artifact"
  grep -Eq '^  \[streamk fixture exactness\] fixture=a0-exact shape=1x4096x4096 .* -> ORDER-INDEPENDENT\+FP16-EXACT$' "$log" \
    || fail "B=$bpc did not use the exact decode fixture"
  grep -Eq "^  \\[dense marlin decomposition\\] real_cu=[0-9]+ occupancy_api=[0-9]+ blocks_per_cu=${bpc} Q=32 Kt=32 G=[0-9]+ I=[0-9]+ active=[0-9]+ idle=[0-9]+ handoffs=[0-9]+ max_peers=[0-9]+ workspace=[0-9]+$" "$log" \
    || fail "B=$bpc did not report its decomposition"
  grep -Eq '^  \[dense scheduler=marlin\] logical_cta=32 cu=[0-9]+ occupancy_api=[0-9]+ grid=\([0-9]+,1,1\) physical_cta=[0-9]+ block_threads=256 warps/cta=8 resident_warps/cu=[0-9]+$' "$log" \
    || fail "B=$bpc did not launch the 256-thread/8-warp kernel"
  grep -Eq '^  \[dense kernel-span-upper\] n=20 median=[0-9.]+ us .*distinct-event-pairs=20 .*includes-launch-idle=1 .*lock-reset-before-start=0$' "$log" \
    || fail "B=$bpc did not report 20 independent kernel spans"
  local repeats repeat repeat_count
  repeats="$(grep -Ec '^  \[dense marlin lock fingerprint\] repeat=[1-8]/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$' "$log" || true)"
  [ "$repeats" -eq 8 ] || fail "B=$bpc produced $repeats/8 stable lock fingerprints"
  for repeat in 1 2 3 4 5 6 7 8; do
    repeat_count="$(grep -Ec "^  \\[dense marlin lock fingerprint\\] repeat=${repeat}/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$" "$log" || true)"
    [ "$repeat_count" -eq 1 ] \
      || fail "B=$bpc lock fingerprint repeat=$repeat appeared $repeat_count times (expected exactly once)"
  done
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
ILLEGAL_CMD=("$BIN" "${COMMON[@]}" "--marlin-blocks-per-cu=$ILLEGAL")
"${ILLEGAL_CMD[@]}" >"$ARTIFACT_ROOT/illegal-bpc.log" 2>&1
illegal_rc=$?
set -e
record_command dense-wk4-illegal-bpc "$illegal_rc" "${ILLEGAL_CMD[@]}"
[ "$illegal_rc" -ne 0 ] || fail "illegal B=$ILLEGAL unexpectedly returned success"
grep -Fq -- "--marlin-blocks-per-cu=$ILLEGAL is outside the exact kernel occupancy range 1..$CAP" \
  "$ARTIFACT_ROOT/illegal-bpc.log" \
  || fail "illegal B=$ILLEGAL was not rejected by the exact runtime cap"

verify_source_identity

printf '\n[marlin-wk4] PASS: standalone classic Marlin built on classic u32 bytes; supported B points passed exact output + 8-launch locks; over-cap B stayed NOT RUN\n'
printf '[marlin-wk4] runner_exit_status=0\n'
printf '[marlin-wk4] artifacts preserved at %s\n' "$ARTIFACT_ROOT"

# Close and join the runner stream before publishing provenance.json.  The
# latter is the completion marker, so no observer can see a successful bundle
# while runner.log is still in flight.
finish_runner_stream

python3 "$PROVENANCE_TOOL" write \
  --output "$ARTIFACT_ROOT/provenance.json" \
  --root-sha "$ROOT_SHA" --root-status clean \
  --submodule-status-file "$SUBMODULE_STATUS_FILE" \
  --actlize-sha "$ACTLIZE_SHA" --binary-sha256 "$BIN_SHA" \
  --device-model "$QUACTLIZE_BOX_DEVICE_MODEL" \
  --pci-identity "$QUACTLIZE_BOX_PCI_IDENTITY" \
  --driver-version "$QUACTLIZE_BOX_DRIVER_VERSION" \
  --sdk-compiler-identity "$QUACTLIZE_BOX_SDK_COMPILER_IDENTITY" \
  --identity-probe-file "$IDENTITY_PROBE_FILE" \
  --run-identity-file "$RUN_IDENTITY_FILE" \
  --commands-file "$COMMANDS_LOG" --runner-exit-status 0 \
  --protocol-sample-count "$SAMPLES" -- "${ORIGINAL_ARGV[@]}"
trap - EXIT
