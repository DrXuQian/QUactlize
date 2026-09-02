#!/usr/bin/env bash
# Local-only resumable builder for the isolated A02 typed diagnostics.
set -euo pipefail

fail() {
  printf '[fq-a02-build] FAIL: %s\n' "$*" >&2
  exit 2
}

generate_once() {
  local output="$1"
  shift
  if [[ -d "$output" && ! -L "$output" ]]; then
    [[ -f "$output/manifest.json" && ! -L "$output/manifest.json" ]] ||
      fail "preserving incomplete generated directory: $output"
    return
  fi
  [[ ! -e "$output" && ! -L "$output" ]] ||
    fail "generated path is not a regular directory: $output"
  "$@"
}

ensure_directory() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    [[ -d "$path" && ! -L "$path" ]] ||
      fail "expected regular directory, preserving unexpected path: $path"
  else
    mkdir "$path"
  fi
}

assert_replaceable_regular_file() {
  local path="$1"
  [[ ! -e "$path" && ! -L "$path" ]] ||
    [[ -f "$path" && ! -L "$path" ]] ||
    fail "refusing to overwrite non-regular stage path: $path"
}

assert_build_identity() {
  local arm="$1" log="$2" cmake_log="$3" build_make="$4"
  [[ -f "$log" && ! -L "$log" && -f "$cmake_log" && ! -L "$cmake_log" &&
     -f "$build_make" && ! -L "$build_make" ]] ||
    fail "$arm build evidence is missing or symlinked"
  if [[ "$arm" == q4 ]]; then
    grep -F '[build.sh] FQ_SWEEP_QTYPE=12' "$log" >/dev/null &&
      grep -F '[build.sh] FQ_SWEEP_ARTIFACT_TK=64' "$log" >/dev/null &&
      grep -F '[build.sh] FQ_SWEEP_BCHUNK=0' "$log" >/dev/null &&
      grep -F 'FullyQuantized internal sweep: q=12 A=64 bc=0 format=0' "$cmake_log" >/dev/null &&
      grep -F -- '-DFQ_SWEEP_QTYPE=12' "$build_make" >/dev/null &&
      grep -F -- '-DFQ_SWEEP_ARTIFACT_TK=64' "$build_make" >/dev/null &&
      grep -F -- '-DFQ_SWEEP_BCHUNK=0' "$build_make" >/dev/null &&
      grep -F -- '-DPPU_PACKED_FORMAT=0' "$build_make" >/dev/null ||
      fail 'Q4 target/definition identity did not reach build/CMake/build.make'
  else
    grep -F '[build.sh] FQ_A02_Q3_GENERATED_DIR=' "$log" >/dev/null &&
      grep -F 'A02 Q3 aggregate: q=11 A=64 effective-bchunk=0+1 units=2 NONPRODUCT' "$cmake_log" >/dev/null &&
      grep -F -- '-DFQ_SWEEP_QTYPE=11' "$build_make" >/dev/null &&
      grep -F -- '-DFQ_SWEEP_ARTIFACT_TK=64' "$build_make" >/dev/null &&
      grep -F -- '-DFQ_SWEEP_BCHUNK=-1' "$build_make" >/dev/null &&
      grep -F -- '-DFQ_SWEEP_MIXED_BCHUNK_DIAGNOSTIC=1' "$build_make" >/dev/null &&
      grep -F -- '-DPPU_PACKED_FORMAT=3' "$build_make" >/dev/null ||
      fail 'Q3 target/definition identity did not reach build/CMake/build.make'
  fi
}

main() {
  [[ $# -le 1 ]] || fail "usage: $0 [BUNDLE_DIR]"
  local root artifact_root source_short out work authority sdk jobs resume
  local generated stage publish q4_tree q3_tree q4_resume q3_resume
  local q4_binary q3_binary q4_make q3_make
  local -a q4_makes q3_makes

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  artifact_root="${FQ_A02_ROOT:-/root/autodl-tmp}"
  mkdir -p "$artifact_root"
  artifact_root="$(realpath -e -- "$artifact_root")"
  source_short="$(git -C "$root" rev-parse --short=8 HEAD)"
  out="$(realpath -m -- "${1:-$artifact_root/fq-a02-prebuilt-$source_short}")"
  case "$out" in "$artifact_root"/*) ;; *) fail 'bundle is outside artifact root' ;; esac
  work="$out.work"
  authority="$work/build-authority.json"
  generated="$work/generated"
  stage="$work/stage"
  publish="$work/publish"
  q4_tree="$work/q4-build"
  q3_tree="$work/q3-build"
  resume="${RESUME:-0}"
  [[ "$resume" == 0 || "$resume" == 1 ]] || fail 'RESUME must be 0 or 1'
  [[ -z "${PPU_DEFS:-}${PPU_EXTRA_DEFS:-}" ]] ||
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS are forbidden'

  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" ||
    fail 'real PPU SDK is required'
  [[ -x "$sdk/bin/hgcc" && ! -L "$sdk/bin/hgcc" &&
     -x "$sdk/bin/hgobjdump" && ! -L "$sdk/bin/hgobjdump" ]] ||
    fail 'regular SDK compiler and inspector are required'
  jobs="${JOBS:-2}"
  [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail 'JOBS must be positive'
  git -C "$root" diff --quiet --ignore-submodules=none HEAD -- ||
    fail 'tracked source or submodule state is dirty'
  [[ -z "$(git -C "$root" status --porcelain -- \
    build.sh quactlize/csrc/CMakeLists.txt.in \
    quactlize/csrc/fq_internal_sweep.cmake.in \
    benchmarks/test_fully_quantized_internal_sweep.cu \
    benchmarks/fully_quantized_splitk_producer_bench.hpp \
    benchmarks/fully_quantized_splitk_producer_unit.inc \
    tools/fully_quantized_internal_matrix.py \
    tools/gen_fully_quantized_splitk_producer_units.py \
    tools/select_fq_a02_typed_diagnostics.py \
    tools/check_fq_a02_typed_diagnostics.py tools/fq_a02_prebuilt.py \
    tools/build_fq_a02_prebuilt.sh tools/run_fq_a02_prebuilt_box.sh)" ]] ||
    fail 'A02 authority is dirty or untracked'

  if [[ -e "$out" || -L "$out" ]]; then
    [[ "$resume" == 1 && -d "$out" && ! -L "$out" ]] ||
      fail 'completed bundle exists; use RESUME=1 to verify it'
    [[ -d "$work" && ! -L "$work" ]] ||
      fail 'completed bundle lost its deterministic build authority work'
    python3 -B "$root/tools/fq_a02_prebuilt.py" verify-build-authority \
      --file "$authority" --source-root "$root" --sdk "$sdk"
    python3 -B "$root/tools/fq_a02_prebuilt.py" verify \
      --bundle "$out" --source-root "$root" --sdk "$sdk"
    printf '[fq-a02-build] PASS completed bundle strictly reverified=%s\n' "$out"
    return
  fi
  if [[ -e "$work" || -L "$work" ]]; then
    [[ "$resume" == 1 && -d "$work" && ! -L "$work" ]] ||
      fail 'deterministic work exists; use RESUME=1 or inspect it'
    python3 -B "$root/tools/fq_a02_prebuilt.py" verify-build-authority \
      --file "$authority" --source-root "$root" --sdk "$sdk"
  else
    mkdir "$work"
    python3 -B "$root/tools/fq_a02_prebuilt.py" create-build-authority \
      --output "$authority" --source-root "$root" --sdk "$sdk"
  fi

  if [[ -e "$publish" || -L "$publish" ]]; then
    [[ "$resume" == 1 && -d "$publish" && ! -L "$publish" ]] ||
      fail 'publish stage exists; use RESUME=1 or inspect it'
    python3 -B "$root/tools/fq_a02_prebuilt.py" verify \
      --bundle "$publish" --source-root "$root" --sdk "$sdk"
    mv -- "$publish" "$out"
    printf '[fq-a02-build] PASS recovered atomic publish=%s\n' "$out"
    return
  fi

  ensure_directory "$generated"
  ensure_directory "$stage"
  python3 -B "$root/tools/select_fq_a02_typed_diagnostics.py" --self-test
  python3 -B "$root/tools/check_fq_a02_typed_diagnostics.py" --self-test
  python3 -B "$root/tools/fq_a02_prebuilt.py" self-test
  generate_once "$generated/q4-full" \
    python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --per-unit 1 \
    --out-dir "$generated/q4-full"
  generate_once "$generated/q3-bc0-full" \
    python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 11 --artifact-tk 64 --bchunk 0 --per-unit 1 \
    --out-dir "$generated/q3-bc0-full"
  generate_once "$generated/q3-bc1-full" \
    python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 11 --artifact-tk 64 --bchunk 1 --per-unit 1 \
    --out-dir "$generated/q3-bc1-full"
  if [[ -d "$generated/exact" && ! -L "$generated/exact" ]]; then
    [[ -f "$generated/exact/q4/manifest.json" &&
       -f "$generated/exact/q3/manifest.json" ]] ||
      fail "preserving incomplete exact selection: $generated/exact"
  else
    [[ ! -e "$generated/exact" && ! -L "$generated/exact" ]] ||
      fail 'exact selection path is not a regular directory'
    python3 -B "$root/tools/select_fq_a02_typed_diagnostics.py" \
      --q4-source "$generated/q4-full" --q3-bc0 "$generated/q3-bc0-full" \
      --q3-bc1 "$generated/q3-bc1-full" --output "$generated/exact"
  fi

  if [[ -e "$q4_tree" || -L "$q4_tree" ]]; then
    [[ -d "$q4_tree" && ! -L "$q4_tree" ]] ||
      fail "Q4 build tree has an unexpected type: $q4_tree"
  fi
  q4_resume=0; [[ -f "$q4_tree/CMakeCache.txt" ]] && q4_resume=1
  env -u CC -u CXX -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
    -u PPU_DEFS -u PPU_EXTRA_DEFS \
    PPU_SDK="$sdk" PPU_BUILD_DIR="$q4_tree" PPU_BUILD_RESUME="$q4_resume" \
    PPU_ARCHS=ppu0010 JOBS="$jobs" TARGET=test_fully_quantized_internal_sweep \
    FQ_SWEEP_GENERATED_DIR="$generated/exact/q4" FQ_SWEEP_QTYPE=12 \
    FQ_SWEEP_ARTIFACT_TK=64 FQ_SWEEP_BCHUNK=0 FQ_SWEEP_PACKED_FORMAT=0 \
    bash "$root/build.sh" >"$work/q4-build.log" 2>&1 || {
      tail -n 120 "$work/q4-build.log" >&2
      fail 'Q4 build failed; deterministic work preserved'
    }
  if [[ -e "$q3_tree" || -L "$q3_tree" ]]; then
    [[ -d "$q3_tree" && ! -L "$q3_tree" ]] ||
      fail "Q3 build tree has an unexpected type: $q3_tree"
  fi
  q3_resume=0; [[ -f "$q3_tree/CMakeCache.txt" ]] && q3_resume=1
  env -u CC -u CXX -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
    -u PPU_DEFS -u PPU_EXTRA_DEFS \
    PPU_SDK="$sdk" PPU_BUILD_DIR="$q3_tree" PPU_BUILD_RESUME="$q3_resume" \
    PPU_ARCHS=ppu0010 JOBS="$jobs" TARGET=test_fq_a02_q3_bchunk_aggregate \
    FQ_A02_Q3_GENERATED_DIR="$generated/exact/q3" \
    bash "$root/build.sh" >"$work/q3-build.log" 2>&1 || {
      tail -n 120 "$work/q3-build.log" >&2
      fail 'Q3 aggregate build failed; deterministic work preserved'
    }

  mapfile -t q4_makes < <(find -P "$q4_tree" -type f \
    -path '*test_fully_quantized_internal_sweep.dir/build.make')
  mapfile -t q3_makes < <(find -P "$q3_tree" -type f \
    -path '*test_fq_a02_q3_bchunk_aggregate.dir/build.make')
  [[ ${#q4_makes[@]} -eq 1 && ${#q3_makes[@]} -eq 1 ]] ||
    fail 'target build.make evidence is not unique'
  q4_make="${q4_makes[0]}"; q3_make="${q3_makes[0]}"
  assert_build_identity q4 "$work/q4-build.log" "$q4_tree/cmake.log" "$q4_make"
  assert_build_identity q3 "$work/q3-build.log" "$q3_tree/cmake.log" "$q3_make"

  q4_binary="$(find -P "$q4_tree" -type f \
    -name test_fully_quantized_internal_sweep -perm -u+x -print -quit)"
  q3_binary="$(find -P "$q3_tree" -type f \
    -name test_fq_a02_q3_bchunk_aggregate -perm -u+x -print -quit)"
  [[ -f "$q4_binary" && ! -L "$q4_binary" &&
     -f "$q3_binary" && ! -L "$q3_binary" ]] || fail 'exact binaries are missing'
  for target in test_fully_quantized_internal_sweep \
      test_fq_a02_q3_bchunk_aggregate q4.isa.txt q3.isa.txt; do
    assert_replaceable_regular_file "$stage/$target"
  done
  cp -- "$q4_binary" "$stage/test_fully_quantized_internal_sweep"
  cp -- "$q3_binary" "$stage/test_fq_a02_q3_bchunk_aggregate"
  "$sdk/bin/hgobjdump" --dump-isa "$q4_binary" >"$stage/q4.isa.txt"
  "$sdk/bin/hgobjdump" --dump-isa "$q3_binary" >"$stage/q3.isa.txt"
  [[ -s "$stage/q4.isa.txt" && -s "$stage/q3.isa.txt" ]] ||
    fail 'ISA evidence is empty'
  python3 -B "$root/tools/fq_a02_prebuilt.py" create \
    --bundle "$publish" --stage "$stage" --sdk "$sdk" \
    --logs "$work/q4-build.log" "$work/q3-build.log" \
    --cmake-logs "$q4_tree/cmake.log" "$q3_tree/cmake.log" \
    --build-makes "$q4_make" "$q3_make"
  mv -- "$publish" "$out"
  printf '[fq-a02-build] PASS bundle=%s work=%s\n' "$out" "$work"
}

main "$@"
