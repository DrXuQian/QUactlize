#!/usr/bin/env bash
# Device regression for the four formats which still ship the Xplane byte map.
#
# Two libraries are required for every row:
#   * base_so: default/Q4 build, used by generic placement, BC and the
#     independent stored-ScaleFirst producer;
#   * format_so: one PPU_PACKED_FORMAT build, used only by the selected
#     fully-quantized arrangement reader.
#
# Using format_so as QUACTLIZE_PPU_LIB makes the independent arm inherit the
# packed format under test.  This runner therefore names both handles and
# checks their build identities before pytest starts.
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

jobs="${JOBS:-16}"
resume="${RESUME:-0}"
formats="${FORMATS:-Q2_K Q3_K Q5_K Q6_K}"
prepass_arm="${PREPASS_ARM:-cooperative}"
scope="${SCOPE:-full}"
# The broad compatibility board historically enables every packed-scale
# specialization.  A prepass-only production-isomorphism check can instead
# name the model-format build, e.g. QUACTLIZE_DENSE_ONLY=10 for Q2_K.
base_defs="${BASE_DEFS:-PPU_PACKED_SCALE=1}"
prepass_n="${PREPASS_N:-256}"
prepass_k="${PREPASS_K:-512}"
sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
sha="$(git rev-parse HEAD)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="${OUT:-/workspace/quactlize-nonq4-xplane-${sha:0:8}-${stamp}}"

fail() {
  printf '[nonq4-xplane] FAIL: %s\n' "$*" >&2
  return 2
}

case "$resume" in
  0|1) ;;
  *) fail "RESUME must be 0 or 1, got $resume" ;;
esac
case "$prepass_arm" in
  cooperative) prepass_defs='' ;;
  serial|ladder) prepass_defs='PPU_PACKED_UNIT_PREPASS_SERIAL=1' ;;
  *) fail "PREPASS_ARM must be cooperative, serial or ladder, got $prepass_arm" ;;
esac
case "$scope" in
  full|prepass) ;;
  *) fail "SCOPE must be full or prepass, got $scope" ;;
esac
if [ "$scope" = prepass ]; then
  [[ "$prepass_n" =~ ^[1-9][0-9]*$ ]] && ((prepass_n % 256 == 0)) || \
    fail "PREPASS_N must be a positive multiple of 256, got $prepass_n"
  [[ "$prepass_k" =~ ^[1-9][0-9]*$ ]] && ((prepass_k % 256 == 0)) || \
    fail "PREPASS_K must be a positive multiple of 256, got $prepass_k"
fi

if [ "$prepass_arm" = ladder ]; then
  python3 - <<'PY'
from pathlib import Path
import re

header = Path("quactlize/include/gguf_scale_prepass.hpp")
device = Path("quactlize/csrc/device/ppu_backend.cu")

def body(path, needle):
    source = path.read_text()
    begin = source.index(needle)
    begin = source.index("{", begin)
    depth = 0
    for end in range(begin, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[begin + 1:end]
    raise AssertionError(f"unterminated body: {needle}")

def tokens(source):
    source = re.sub(r"//.*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"\s+", "", source)

pairs = (
    ("production-clone", "__global__ void prepass_unit_kernel(UnitPrepassKernelArgs args)",
     "__global__ void prepass_unit_kernel_cu_clone", ("lane&3", "lane&7")),
    ("bootstrap", "__global__ void prepass_header_bootstrap_kernel",
     "__global__ void prepass_cu_bootstrap_kernel", ("0x3c00u", "0x3c01u")),
    ("template-bootstrap", "__global__ void prepass_header_template_bootstrap_kernel",
     "__global__ void prepass_cu_template_bootstrap_kernel", ("0x3c00u", "0x3c01u")),
)
for name, header_needle, device_needle, mutation in pairs:
    left = tokens(body(header, header_needle))
    right = tokens(body(device, device_needle))
    assert left == right, f"{name} changed more than source location"
    assert mutation[0] in right, f"{name} negative plant target is absent"
    assert left != right.replace(*mutation), f"{name} negative plant did not turn RED"

production_tail = tokens(body(header, pairs[0][1])).split("usingU=", 1)[1]
scalar_tail = tokens(body(device, "__global__ void prepass_unit_kernel_cu_scalar")).split("usingU=", 1)[1]
assert production_tail == scalar_tail, "flat-scalar arm changed the cooperative body"
assert production_tail != scalar_tail.replace("lane&3", "lane&7"), \
    "flat-scalar body negative plant did not turn RED"

source = device.read_text()
for marker in ("half_scale[o] = sz.scale", "half_zero[o] = sz.zero",
               "prepass_device_arch_marker_kernel", "header_serial=[scale:%zu,zero:%zu]",
               "header_production=[scale:%zu,zero:%zu]", "cu_clone=[scale:%zu,zero:%zu]",
               "cu_scalar=[scale:%zu,zero:%zu]", "dequant_control=[bad:%zu,sentinel:%zu]",
               "launch=[%d,%d,%d", "prepass_cu_bootstrap_kernel<<<1, 128>>>",
               "dequant_kernel<T><<<1, 128>>>"):
    assert marker in source, f"missing ladder seam: {marker}"
print("[nonq4-xplane:ladder-source] PASS exact header/.cu bodies; mutation and missing-seam negatives RED")
PY
fi

if [ "$resume" -eq 0 ] && [ -e "$out" ]; then
  fail "OUT already exists; choose a fresh path or use RESUME=1: $out"
fi
mkdir -p "$out/results"
trap 'rc=$?; printf "[nonq4-xplane] DONE rc=%d artifacts=%s\n" "$rc" "$out"' EXIT

[ -x "$sdk_root/bin/hgcc" ] || fail "real PPU hgcc is absent; set PPU_SDK"
[ -x "$sdk_root/bin/hgobjdump" ] || fail "real PPU hgobjdump is absent; set PPU_SDK"
compiler_identity="$($sdk_root/bin/hgcc --version 2>&1 | head -n 1 || true)"
objdump_identity="$($sdk_root/bin/hgobjdump --version 2>&1 | head -n 1 || true)"
[ -n "$compiler_identity" ] && [[ "$compiler_identity" != *stub* ]] || \
  fail "hgcc identity is empty or a stub"
[ -n "$objdump_identity" ] && [[ "$objdump_identity" != *stub* ]] || \
  fail "hgobjdump identity is empty or a stub"
printf 'source_sha=%s\nhgcc=%s\nhgobjdump=%s\n' \
  "$sha" "$compiler_identity" "$objdump_identity" >"$out/results/authority.txt"
git submodule status --recursive >"$out/results/submodule-status.txt"
! grep -Eq '^[+U-]' "$out/results/submodule-status.txt" || \
  fail "a submodule differs from the recorded gitlink"

# The host extension owns the dlopen split.  Build it with the system compiler,
# never an inherited CUDA/PPU CMake toolchain from an earlier experiment.
env -u CC -u CXX -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
  CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
  python3 setup.py build_ext --inplace >"$out/results/host-build.log" 2>&1 || {
    tail -30 "$out/results/host-build.log" >&2
    fail "host extension build failed"
  }

build_device() {
  local label="$1" defs="$2"
  local build="$out/build-$label" log="$out/results/build-$label.log"
  local build_make so

  printf '[nonq4-xplane] build label=%s defs=%s\n' "$label" "$defs"
  env -i \
    HOME="$HOME" USER="${USER:-root}" PATH="$PATH" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LANG="${LANG:-C.UTF-8}" \
    PPU_SDK="$sdk_root" PPU_ARCHS=ppu0010 \
    PPU_BUILD_DIR="$build" PPU_BUILD_RESUME="$resume" \
    PPU_DEFS="$defs" TARGET=quactlize_ppu JOBS="$jobs" \
    CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
    "$root/build.sh" >"$log" 2>&1 || {
      grep -n -B4 -A8 -E \
        'error:|fatal error:|undefined reference|ld\.lld:|LLVM ERROR|Killed|timed out|Segmentation|PLEASE submit' \
        "$log" | head -120 >&2 || true
      tail -40 "$log" >&2
      fail "$label device build failed"
    }

  grep -qF '[build.sh] CUTLASS_PPU_ARCHS=ppu0010' "$log" || \
    fail "$label did not bind ppu0010"
  grep -qF "PPU hgcc        : $sdk_root/bin/hgcc" "$build/cmake.log" || \
    fail "$label CMake did not bind the selected hgcc"
  grep -qF 'PPU device archs: ppu0010' "$build/cmake.log" || \
    fail "$label CMake did not bind the ppu0010 device architecture"
  local def
  for def in $defs; do
    grep -qF "PPU_DEFS verified on quactlize_ppu's compile command: -D$def" "$log" || \
      fail "$label did not compile with -D$def"
  done

  build_make="$(find "$build" -path '*quactlize_ppu.dir/build.make' -print -quit)"
  [ -n "$build_make" ] || fail "$label has no quactlize_ppu build.make"
  grep -qF "$sdk_root/bin/hgcc" "$build_make" || \
    fail "$label device objects were not assigned to hgcc"
  grep -q -- '-arch=ppu_10' "$build_make" || fail "$label lacks -arch=ppu_10"
  grep -q -- '-x hg' "$build_make" || fail "$label lacks the PPU device-language flag"

  so="$(grep -m1 '^built: ' "$log" | cut -d' ' -f2-)"
  [ -f "$so" ] || fail "$label build reported no shared library"
  "$sdk_root/bin/hgobjdump" -lelf "$so" \
    >"$out/results/$label.elf.txt" 2>"$out/results/$label.elf.err" || \
    fail "$label shared library is not parseable by PPU hgobjdump"
  grep -q 'Func ' "$out/results/$label.elf.txt" || \
    fail "$label shared library exposes no PPU device functions"
  if [ "$prepass_arm" = ladder ]; then
    for symbol in \
      prepass_header_bootstrap_kernel prepass_cu_bootstrap_kernel \
      prepass_header_template_bootstrap_kernel prepass_cu_template_bootstrap_kernel \
      prepass_device_arch_marker_kernel prepass_unit_kernel_cu_clone prepass_unit_kernel_cu_scalar
    do
      # Identical-code folding may retain only one symbol for the token-identical pairs.  Runtime launches are the
      # authority; record the census without rejecting a valid alias before it can run.
      printf 'FQ_PACKED_UNIT_PREPASS_ELF label=%s symbol=%s count=%s\n' \
        "$label" "$symbol" "$(grep -c "$symbol" "$out/results/$label.elf.txt" || true)"
    done
  fi
  sha256sum "$so" >"$out/results/$label.so.sha256"
  printf '%s\n' "$so" >"$out/results/$label.so.path"
}

# A failed/diagnostic build may be resumed across a change to the ONE translation unit that owns the probe.  hgcc's
# custom command tracks that .cu as MAIN_DEPENDENCY, so make will rebuild it; included headers are deliberately not
# allowed because the generated command has no header depfile.  The runner itself may differ because it is not an
# input to any device object.  This is narrower than build.sh's normal source-authority rule and exists only for the
# prepass ladder, where throwing away an otherwise complete base build would add no evidence.
if [ "$resume" -eq 1 ] && [ "$scope" = prepass ] && [ "$prepass_arm" = ladder ]; then
  authority="$out/build-base/.quactlize-source-head"
  if [ -f "$authority" ]; then
    read -r prior_sha <"$authority"
    if [ "$prior_sha" != "$sha" ]; then
      git diff --quiet HEAD -- || fail "ladder resume refuses tracked working-tree changes"
      changed="$(git diff --name-only "$prior_sha..$sha")" || fail "cannot compare ladder resume authorities"
      saw_device=0
      while IFS= read -r path; do
        [ -n "$path" ] || continue
        case "$path" in
          quactlize/csrc/device/ppu_backend.cu) saw_device=1 ;;
          tools/run_nonq4_xplane_correctness_box.sh) ;;
          *) fail "ladder resume source change is not a directly tracked probe TU: $path" ;;
        esac
      done <<<"$changed"
      [ "$saw_device" -eq 1 ] || fail "ladder resume changed no device probe translation unit"
      printf '%s\n' "$sha" >"$authority"
      printf '[nonq4-xplane] resume authority advanced %s -> %s for direct probe TU only\n' \
        "$prior_sha" "$sha"
    fi
  fi
fi

# The base has packed-unit support but deliberately has no selected
# PPU_PACKED_FORMAT.  It remains the independent producer/oracle arm.  Keep
# its production build identity caller-selectable for the prepass isolate;
# the default preserves the complete compatibility board.
build_device base "$base_defs $prepass_defs"
base_so="$(cat "$out/results/base.so.path")"
base_make="$(find "$out/build-base" -path '*quactlize_ppu.dir/build.make' -print -quit)"
! grep -qE -- '(^|[[:space:]])-DPPU_PACKED_FORMAT(=|[[:space:]])' "$base_make" || \
  fail "base library unexpectedly selected PPU_PACKED_FORMAT"

format_spec() {
  case "$1" in
    Q2_K) printf '10 2\n' ;;
    Q3_K) printf '11 3\n' ;;
    # Q4_K remains outside this runner's product denominator.  It is admitted
    # only as the same-binary/same-shape packed-unit prepass control: SCOPE=prepass
    # does not build or invoke a format-selected tensor-core reader.
    Q4_K)
      [ "$scope" = prepass ] || fail "Q4_K is only a prepass control in the non-Q4 runner"
      printf '12 0\n'
      ;;
    Q5_K) printf '13 1\n' ;;
    Q6_K) printf '14 4\n' ;;
    *) fail "unknown non-Q4 format $1" ;;
  esac
}

oracle_nodes=(
  test_packed_unit_scale_derivation_matches_the_scale_first_planes
  test_bc_dequant_all_matches_official_gguf
  test_bc_gemv_matches_dequant_first_and_rejects_fault
  test_bc_gemv_moe_matches_dequant_first_and_rejects_fault
  test_fully_quantized_grouped_matches_dequant_first_and_rejects_fault
  test_fully_quantized_dense_matches_dequant_first_and_rejects_fault
)
if [ "$scope" = prepass ]; then
  oracle_nodes=(test_packed_unit_scale_derivation_matches_the_scale_first_planes)
fi

for label in $formats; do
  read -r qtype fmt < <(format_spec "$label")
  test_log="$out/results/$label.test.log"
  format_so=''

  if [ "$scope" = full ]; then
    build_device "$label" "PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=$fmt"
    format_so="$(cat "$out/results/$label.so.path")"
    if cmp -s "$base_so" "$format_so"; then
      fail "$label format library is byte-identical to the base despite a different compile identity"
    fi
  fi

  # KEEP THESE TWO HANDLES DIFFERENT.  load() owns generic placement, BC and
  # the independent ScaleFirst arm; load_format(fmt) owns the selected FQ
  # reader.  Reversing this assignment invalidates the oracle itself.
  : >"$test_log"
  passed=0
  for oracle in "${oracle_nodes[@]}"; do
    oracle_log="$out/results/$label.$oracle.log"
    oracle_env=(QUACTLIZE_PPU_LIB="$base_so" QUACTLIZE_PACKED_FORMAT="$qtype" PYTHONPATH="$root")
    if [ "$scope" = prepass ]; then
      oracle_env+=("QUACTLIZE_PREPASS_TEST_N=$prepass_n" "QUACTLIZE_PREPASS_TEST_K=$prepass_k")
    fi
    if [ "$scope" = full ]; then
      oracle_env+=("QUACTLIZE_PPU_LIB_FMT${fmt}=$format_so")
    fi
    if ! env "${oracle_env[@]}" \
        python3 -m pytest -q -rs -s tests/test_gguf_routes.py \
          -m fully_quantized_dense -k "$label and $oracle" >"$oracle_log" 2>&1; then
      printf 'NONQ4_XPLANE_ORACLE format=%s oracle=%s verdict=FAIL\n' "$label" "$oracle" | tee -a "$test_log"
      grep -E '^FQ_PACKED_UNIT_PREPASS_LADDER ' "$oracle_log" >&2 || true
      tail -80 "$oracle_log" >&2
      fail "$label oracle $oracle failed"
    fi
    grep -Eq '(^| )1 passed' "$oracle_log" || fail "$label oracle $oracle did not run exactly one passing test"
    ! grep -qi 'skipped' "$oracle_log" || fail "$label oracle $oracle unexpectedly skipped"
    cat "$oracle_log" >>"$test_log"
    printf 'NONQ4_XPLANE_ORACLE format=%s oracle=%s verdict=PASS\n' "$label" "$oracle" | tee -a "$test_log"
    passed=$((passed + 1))
  done
  expected="${#oracle_nodes[@]}"
  [ "$passed" -eq "$expected" ] || fail "$label ran $passed/$expected isolated oracles"
  printf 'NONQ4_XPLANE format=%s verdict=PASS tests=%s scope=%s prepass_arm=%s N=%s K=%s base_defs=%s\n' \
    "$label" "$expected" "$scope" "$prepass_arm" "$prepass_n" "$prepass_k" "${base_defs// /,}"
done

printf 'NONQ4_XPLANE_ALL verdict=PASS formats=%s\n' "${formats// /,}"
