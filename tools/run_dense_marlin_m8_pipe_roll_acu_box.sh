#!/usr/bin/env bash
# Same-SHA BPC1 experiment for INBOX 173.
#
# baseline      : shipping outer+inner unroll spelling
# outer-roll    : only the pointer-indexed outer pipe loop is rolled
# inner-control : both loops rolled; compile/disassembly only, never executed
#
# ACU reports are native .acurep files.  This runner deliberately emits no CSV.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_marlin_m8_ab
ACU="${ACU:-$(command -v acu || true)}"
OUT="${MARLIN_M8_PIPE_ROLL_OUT:-/workspace/quactlize-dense-marlin-m8-pipe-roll-acu}"
REPORTER="$ROOT/dev/fold_derivation/l182_marlin_pipe_roll_report.py"

fail() { printf '[marlin-m8-pipe-roll] FAIL: %s\n' "$*" >&2; exit 1; }
[ "$#" -eq 0 ] || fail 'this runner accepts no positional arguments'
[ -x "$REPORTER" ] || fail "missing executable reporter: $REPORTER"
[ -n "$ACU" ] && [ -x "$ACU" ] || \
  fail 'ACU is unavailable; set ACU to the site acu executable'
[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ] || \
  fail 'source tree must be clean so all three binaries name one exact SHA'
mkdir -p "$(dirname "$OUT")"
exec 9>"${OUT}.lock"
flock -n 9 || fail "another pipe-roll experiment owns $OUT"
if [ -e "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  fail "output directory is not empty: $OUT"
fi
mkdir -p "$OUT"

resolve_executable() {
  local candidate="$1"
  [ -n "$candidate" ] || return 1
  if [[ "$candidate" == */* ]]; then
    [ -x "$candidate" ] && { readlink -f "$candidate"; return 0; }
  else
    command -v "$candidate" 2>/dev/null && return 0
  fi
  return 1
}

sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
hgcc="$(resolve_executable "${HGCC:-${sdk_root:+$sdk_root/bin/hgcc}}" || true)"
hgobjdump="$(resolve_executable "${HGOBJDUMP:-${sdk_root:+$sdk_root/bin/hgobjdump}}" || true)"
[ -n "$hgcc" ] || hgcc="$(resolve_executable "$(command -v hgcc 2>/dev/null || true)" || true)"
[ -n "$hgobjdump" ] || hgobjdump="$(resolve_executable "$(command -v hgobjdump 2>/dev/null || true)" || true)"
[ -n "$hgcc" ] && [ -n "$hgobjdump" ] || \
  fail 'real hgcc and hgobjdump are required for the static-footprint admission gate'
[ -n "$sdk_root" ] || sdk_root="$(cd "$(dirname "$hgcc")/.." && pwd)"
sdk_root="$(cd "$sdk_root" && pwd)"
[ "$(resolve_executable "$sdk_root/bin/hgcc" || true)" = "$hgcc" ] || \
  fail "hgcc is not owned by PPU_SDK=$sdk_root"
[ "$(resolve_executable "$sdk_root/bin/hgobjdump" || true)" = "$hgobjdump" ] || \
  fail "hgobjdump is not owned by PPU_SDK=$sdk_root"
compiler_identity="$($hgcc --version 2>&1 | head -n 1 || true)"
objdump_identity="$($hgobjdump --version 2>&1 | head -n 1 || true)"
[ -n "$compiler_identity" ] && [[ "$compiler_identity" != *stub* ]] || \
  fail 'hgcc identity is empty or a stub'
[ -n "$objdump_identity" ] && [[ "$objdump_identity" != *stub* ]] || \
  fail 'hgobjdump identity is empty or a stub'

acu_real="$(readlink -f "$ACU")"
acu_identity="$($acu_real --version 2>&1 | head -n 1 || true)"
[ -n "$acu_identity" ] || acu_identity='version-unreported'

ROOT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
ACTLIZE_SHA="$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
git -C "$ROOT" submodule status --recursive >"$OUT/submodule-status.txt"
grep -Eq '^[+\-U]' "$OUT/submodule-status.txt" && \
  fail 'a submodule checkout differs from its recorded gitlink'
sha256sum "$hgcc" "$hgobjdump" >"$OUT/sdk-tools.sha256"
sha256sum "$acu_real" >"$OUT/acu.sha256"

SOURCE_PATHS=(
  quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp
  quactlize/include/quactlize_extensions/cutlass/gemm/kernel/marlin_kernel_ppu.hpp
  benchmarks/lowbit_dense_unit.inc
  benchmarks/test_lowbit_dense_bench.cu
  quactlize/csrc/CMakeLists.txt.in
)
(cd "$ROOT" && sha256sum "${SOURCE_PATHS[@]}") >"$OUT/source.before.sha256"

MODES=(baseline outer-roll inner-roll-control)
VALUES=(0 1 2)
BINS=()
LINES=()
RESOURCES=()

for index in "${!MODES[@]}"; do
  mode="${MODES[$index]}"
  value="${VALUES[$index]}"
  mode_dir="$OUT/$mode"
  build_dir="$mode_dir/build"
  mkdir -p "$build_dir"
  BUILD=(env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE -u CC -u CXX
         PPU_SDK="$sdk_root" PPU_HOME= PPU_SDK_SITE_DEFAULT=
         PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 TARGET="$TARGET"
         JOBS=1 QUANT=int4 BENCH_GS=128
         PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS=)
  if [ "$value" -eq 0 ]; then
    BUILD+=(PPU_DEFS=)
  else
    BUILD+=(PPU_DEFS="PPU_MARLIN_PIPE_ROLL=$value")
  fi
  BUILD+=("$ROOT/build.sh")
  printf '%q ' "${BUILD[@]}" >"$mode_dir/build.command"
  printf '\n' >>"$mode_dir/build.command"
  "${BUILD[@]}" >"$mode_dir/build.log" 2>&1 || {
    first="$(grep -m1 -E 'fatal error:|error:|CMake Error|\[FAIL\]' "$mode_dir/build.log" || tail -n 1 "$mode_dir/build.log")"
    fail "$mode build failed: $first"
  }
  if [ "$value" -eq 0 ]; then
    ! grep -Fq 'PPU_DEFS applied:' "$mode_dir/build.log" || \
      fail 'baseline unexpectedly carried a build override'
  else
    grep -Fq "PPU_DEFS verified on $TARGET's compile command: -DPPU_MARLIN_PIPE_ROLL=$value" \
      "$mode_dir/build.log" || fail "$mode did not prove its device compile define"
  fi

  mapfile -t bins < <(find "$build_dir" -type f -name "$TARGET" -perm -u+x -print)
  [ "${#bins[@]}" -eq 1 ] || fail "$mode expected one binary, found ${#bins[@]}"
  bin="${bins[0]}"
  BINS+=("$bin")
  sha256sum "$bin" >"$mode_dir/binary.sha256"
  generated="$(find "$build_dir" -type f -name lowbit_dense_marlin_m8_ab_unit.cu -print -quit)"
  [ -n "$generated" ] || fail "$mode generated unit is absent"
  build_make="$(find "$build_dir" -path '*test_lowbit_dense_marlin_m8_ab.dir/build.make' -print -quit)"
  [ -n "$build_make" ] || fail "$mode build.make is absent"
  python3 - "$build_make" "$generated" "$hgcc" "$value" >"$mode_dir/hgcc-command.txt" <<'PY'
import shlex, sys
from pathlib import Path

build_make, generated, hgcc = map(Path, sys.argv[1:4])
mode = int(sys.argv[4])
lines = [line.strip() for line in build_make.read_text(errors="replace").splitlines()
         if str(generated) in line and str(hgcc) in line]
if len(lines) != 1:
    raise SystemExit(f"L182 command audit: expected one generated-unit hgcc command, got {len(lines)}")
tokens = shlex.split(lines[0])
required = {
    "-arch=ppu_10", "-lineinfo", "-DDENSE_MARLIN_WK4_AB=1",
    "-DDENSE_MARLIN_M8_AB=1", "-DDENSE_MARLIN_AB=1",
    "-DDENSE_STREAMK_AB=1", "-DBENCH_GS=128", "-DBENCH_TSK=64",
    "-DDENSE_AB_BITS=4", "-DDENSE_AB_ARTIFACT_TK=64",
    "-DDENSE_AB_TM=8", "-DDENSE_AB_TN=128", "-DDENSE_AB_TK=128",
    "-DDENSE_AB_WM=8", "-DDENSE_AB_WN=64", "-DDENSE_AB_WARP_K=32",
    "-DDENSE_AB_ST=4", "-DDENSE_AB_BC=0",
    "-DTILE_M=16", "-DTILE_N=128", "-DWARP_M=16", "-DWARP_N=64",
    "-DSTAGES=4",
}
missing = sorted(required - set(tokens))
if missing:
    raise SystemExit("L182 command audit: missing " + ",".join(missing))
pipe = [token for token in tokens if token.startswith("-DPPU_MARLIN_PIPE_ROLL=")]
want = [] if mode == 0 else [f"-DPPU_MARLIN_PIPE_ROLL={mode}"]
if pipe != want:
    raise SystemExit(f"L182 command audit: pipe defines are {pipe}, expected {want}")
print(lines[0])
PY

  "$hgobjdump" -lelf "$bin" >"$mode_dir/list-elf.txt" 2>"$mode_dir/list-elf.err" || \
    fail "$mode hgobjdump -lelf failed"
  mapfile -t all_symbols < <(sed -n 's/^.*Func [0-9][0-9]*:[[:space:]]*\([^[:space:]]*\).*$/\1/p' "$mode_dir/list-elf.txt")
  symbols=()
  pretty_symbols=()
  for candidate in "${all_symbols[@]}"; do
    pretty="$(c++filt "$candidate")"
    if [[ "$pretty" == *device_kernel* && "$pretty" == *MarlinKernelPPU* && "$pretty" == *MarlinCollectivePPU* ]]; then
      symbols+=("$candidate")
      pretty_symbols+=("$pretty")
    fi
  done
  [ "${#symbols[@]}" -eq 1 ] || {
    printf '%s\n' "${pretty_symbols[@]}" >"$mode_dir/symbol-candidates.txt"
    fail "$mode has ${#symbols[@]} standalone device symbols, expected exactly one"
  }
  symbol="${symbols[0]}"
  printf '%s\n' "$symbol" >"$mode_dir/kernel-symbol.txt"
  printf '%s\n' "${pretty_symbols[0]}" >"$mode_dir/kernel-symbol-demangled.txt"
  line="$mode_dir/kernel-line.txt"
  resource="$mode_dir/resource-usage.txt"
  "$hgobjdump" -line "-func=$symbol" "$bin" >"$line" 2>"$mode_dir/kernel-line.err" || \
    fail "$mode exact-symbol line disassembly failed"
  "$hgobjdump" "-res-usage=$symbol" "$bin" >"$resource" 2>"$mode_dir/resource-usage.err" || \
    fail "$mode exact-symbol resource report failed"
  LINES+=("$line")
  RESOURCES+=("$resource")
done

[ "$(sha256sum "${BINS[0]}" | awk '{print $1}')" != \
  "$(sha256sum "${BINS[1]}" | awk '{print $1}')" ] || \
  fail 'baseline and outer-roll binaries are byte-identical; the compiler route was not exercised'

python3 "$REPORTER" \
  --source "$ROOT/quactlize/include/quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp" \
  --baseline-line "${LINES[0]}" --baseline-resource "${RESOURCES[0]}" \
  --outer-roll-line "${LINES[1]}" --outer-roll-resource "${RESOURCES[1]}" \
  --inner-roll-control-line "${LINES[2]}" \
  --inner-roll-control-resource "${RESOURCES[2]}" \
  --output "$OUT/codegen.json" | tee "$OUT/codegen.log"

ARGS=(--marlin --streamk_exact_fixture --m=1 --n=4096 --k=4096
      --l=1 --g=128 --mode=1 --alpha=1 --beta=0 --iterations=0)
REPORTS=()
for index in 0 1; do
  mode="${MODES[$index]}"
  value="${VALUES[$index]}"
  bin="${BINS[$index]}"
  mode_dir="$OUT/$mode"
  CORRECTNESS=("$bin" "${ARGS[@]}")
  printf '%q ' "${CORRECTNESS[@]}" >"$mode_dir/correctness.command"
  printf '\n' >>"$mode_dir/correctness.command"
  "${CORRECTNESS[@]}" 2>&1 | tee "$mode_dir/correctness.log"
  grep -Fq 'family=ppu-m8-extension' "$mode_dir/correctness.log" || \
    fail "$mode correctness did not identify the m8 family"
  grep -Fq 'ORDER-INDEPENDENT+FP16-EXACT' "$mode_dir/correctness.log" || \
    fail "$mode correctness did not establish the exact fixture"
  grep -Fq 'mapping=ppu-m8n16 coverage=exact-once' "$mode_dir/correctness.log" || \
    fail "$mode output-owner map did not close"
  grep -Fq 'Disposition: Passed' "$mode_dir/correctness.log" || \
    fail "$mode exact fixture failed"
  [ "$(grep -Ec '^  \[dense marlin lock fingerprint\] repeat=[1-8]/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$' "$mode_dir/correctness.log" || true)" -eq 8 ] || \
    fail "$mode 8-launch lock fingerprint failed"
  for repeat in 1 2 3 4 5 6 7 8; do
    repeat_count="$(grep -Ec "^  \\[dense marlin lock fingerprint\\] repeat=${repeat}/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$" "$mode_dir/correctness.log" || true)"
    [ "$repeat_count" -eq 1 ] || \
      fail "$mode lock fingerprint repeat=$repeat appeared $repeat_count times (expected exactly once)"
  done

  report_base="$mode_dir/marlin-m8-${mode}-bpc1.report"
  ACU_CMD=("$ACU" -f -o "$report_base" --set full "$bin" "${ARGS[@]}"
           --marlin-profile-subject-only)
  printf '%q ' "${ACU_CMD[@]}" >"$mode_dir/acu.command"
  printf '\n' >>"$mode_dir/acu.command"
  "${ACU_CMD[@]}" 2>&1 | tee "$mode_dir/acu.log"
  report_candidates=()
  [ -s "$report_base" ] && report_candidates+=("$report_base")
  [ -s "${report_base}.acurep" ] && report_candidates+=("${report_base}.acurep")
  [ "${#report_candidates[@]}" -eq 1 ] || \
    fail "$mode ACU produced ${#report_candidates[@]} unambiguous reports"
  report="${report_candidates[0]}"
  REPORTS+=("$report")
  marker="[dense marlin ACU subject-only] instruction=m8n16k16 blocks_per_cu=1 pipe_roll=$value outer_pipe_rolled=$value inner_loop_rolled=0 subject_launches=1 device_reference=0 lock_fingerprints=0"
  grep -Fq "$marker" "$mode_dir/acu.log" || \
    fail "$mode subject-only marker did not close"
  [ "$(grep -Fc '[dense marlin ACU subject-only]' "$mode_dir/acu.log" || true)" -eq 1 ] || \
    fail "$mode subject marker count is not one"
  ! grep -Fq 'Disposition:' "$mode_dir/acu.log" || \
    fail "$mode ACU process unexpectedly ran numerical verification"
  ! grep -Fq '[dense marlin lock fingerprint]' "$mode_dir/acu.log" || \
    fail "$mode ACU process unexpectedly ran lock fingerprints"
done

(cd "$ROOT" && sha256sum "${SOURCE_PATHS[@]}") >"$OUT/source.after.sha256"
cmp "$OUT/source.before.sha256" "$OUT/source.after.sha256" || \
  fail 'source authority changed while building/profiling'
verify_source_identity() {
  local current_root current_actlize current_submodules
  [ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ] || \
    fail 'final source identity check found a dirty root or submodule tree'
  current_root="$(git -C "$ROOT" rev-parse HEAD)"
  current_actlize="$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
  [ "$current_root" = "$ROOT_SHA" ] || \
    fail "root HEAD changed during the run: $ROOT_SHA -> $current_root"
  [ "$current_actlize" = "$ACTLIZE_SHA" ] || \
    fail "actlize HEAD changed during the run: $ACTLIZE_SHA -> $current_actlize"
  current_submodules="$(git -C "$ROOT" submodule status --recursive)"
  cmp -s <(printf '%s\n' "$current_submodules") "$OUT/submodule-status.txt" || \
    fail 'recursive submodule status changed during the run'
  for index in "${!MODES[@]}"; do
    sha256sum -c "${OUT}/${MODES[$index]}/binary.sha256" >/dev/null || \
      fail "${MODES[$index]} binary changed after it was admitted"
  done
  printf '[marlin-m8-pipe-roll] final-source-identity=EXACT root=%s actlize=%s binaries=3/3\n' \
    "$current_root" "$current_actlize"
}
verify_source_identity
{
  printf 'schema=quactlize.marlin-m8-pipe-roll-acu.v1\n'
  printf 'root_sha=%s\nactlize_sha=%s\n' "$ROOT_SHA" "$ACTLIZE_SHA"
  printf 'target=%s\nshape=M1,N4096,K4096,L1,gs128\n' "$TARGET"
  printf 'config=TM8,TN128,TK128,WM8,WN64,WarpK32,S4,BPC1\n'
  printf 'modes=baseline:0,outer-roll:1,inner-roll-control:2\n'
  printf 'profiled_modes=baseline,outer-roll\n'
  printf 'inner_control=compile-disassembly-only-never-executed\n'
  printf 'reports=%s,%s\n' "${REPORTS[0]}" "${REPORTS[1]}"
  printf 'primary_metric=Instruction Fetch share of all stall cycles; dynamic instruction total is not an admission criterion\n'
  printf 'fetch_share_supported=outer-roll drops >=10.0 absolute percentage-points versus same-run baseline\n'
  printf 'fetch_share_unresolved=outer-roll drops >=5.0 and <10.0 absolute percentage-points\n'
  printf 'fetch_share_falsified=outer-roll drops <5.0 absolute percentage-points (including an increase) after static shrink >=3.5x\n'
  printf 'secondary_metrics=Stall Sync,I-cache hit rate,time\n'
  printf 'hypothesis_falsifier=static mainloop shrinks >=3.5x but Instruction Fetch share falls <5.0 absolute percentage-points\n'
  printf 'hgcc_identity=%s\nhgobjdump_identity=%s\n' "$compiler_identity" "$objdump_identity"
  printf 'acu=%s\nacu_identity=%s\nacu_sha256=%s\n' \
    "$acu_real" "$acu_identity" "$(sha256sum "$acu_real" | awk '{print $1}')"
  printf 'device_identity_probe=not-part-of-this-same-process-mechanism-A/B\n'
} >"$OUT/manifest.txt"
find "$OUT" -type f ! -name bundle.sha256 -print0 | sort -z | \
  xargs -0 sha256sum >"$OUT/bundle.sha256"

printf '[marlin-m8-pipe-roll] PASS: same-SHA baseline/outer-roll exact fixtures and BPC1 ACU captured; inner-roll resource control RED\n'
printf '[marlin-m8-pipe-roll] root-sha=%s\n' "$ROOT_SHA"
printf '[marlin-m8-pipe-roll] reports=%s,%s\n' "${REPORTS[0]}" "${REPORTS[1]}"
printf '[marlin-m8-pipe-roll] read codegen.json before interpreting ACU; dynamic instruction totals are diagnostic only\n'
