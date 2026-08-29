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
# packed format under test.  Q2 exposed that invalid topology as 7686/8192
# differing scale values.  This runner therefore names both handles and checks
# their build identities before pytest starts.
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

jobs="${JOBS:-16}"
resume="${RESUME:-0}"
formats="${FORMATS:-Q2_K Q3_K Q5_K Q6_K}"
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
  sha256sum "$so" >"$out/results/$label.so.sha256"
  printf '%s\n' "$so" >"$out/results/$label.so.path"
}

# The base has packed-unit support but deliberately has no selected
# PPU_PACKED_FORMAT.  It remains the independent producer/oracle arm.
build_device base 'PPU_PACKED_SCALE=1'
base_so="$(cat "$out/results/base.so.path")"
base_make="$(find "$out/build-base" -path '*quactlize_ppu.dir/build.make' -print -quit)"
! grep -qE -- '(^|[[:space:]])-DPPU_PACKED_FORMAT(=|[[:space:]])' "$base_make" || \
  fail "base library unexpectedly selected PPU_PACKED_FORMAT"

format_spec() {
  case "$1" in
    Q2_K) printf '10 2\n' ;;
    Q3_K) printf '11 3\n' ;;
    Q5_K) printf '13 1\n' ;;
    Q6_K) printf '14 4\n' ;;
    *) fail "unknown non-Q4 format $1" ;;
  esac
}

for label in $formats; do
  read -r qtype fmt < <(format_spec "$label")
  build_device "$label" "PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=$fmt"
  format_so="$(cat "$out/results/$label.so.path")"
  test_log="$out/results/$label.test.log"

  if cmp -s "$base_so" "$format_so"; then
    fail "$label format library is byte-identical to the base despite a different compile identity"
  fi

  # KEEP THESE TWO HANDLES DIFFERENT.  load() owns generic placement, BC and
  # the independent ScaleFirst arm; load_format(fmt) owns the selected FQ
  # reader.  Reversing this assignment invalidates the oracle itself.
  if ! env \
      QUACTLIZE_PPU_LIB="$base_so" \
      "QUACTLIZE_PPU_LIB_FMT${fmt}=$format_so" \
      QUACTLIZE_PACKED_FORMAT="$qtype" PYTHONPATH="$root" \
      python3 -m pytest -q -rs -s tests/test_gguf_routes.py \
        -m fully_quantized_dense -k "$label" >"$test_log" 2>&1; then
    tail -80 "$test_log" >&2
    fail "$label device oracle failed"
  fi
  grep -Eq '(^| )6 passed' "$test_log" || fail "$label did not run six passing oracles"
  ! grep -qi 'skipped' "$test_log" || fail "$label unexpectedly skipped an oracle"
  printf 'NONQ4_XPLANE format=%s verdict=PASS tests=6\n' "$label"
done

printf 'NONQ4_XPLANE_ALL verdict=PASS formats=%s\n' "${formats// /,}"
