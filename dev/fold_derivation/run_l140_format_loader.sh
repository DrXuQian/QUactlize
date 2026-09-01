#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cxx=${CXX:-c++}

"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$repo/quactlize/include" -I"$repo/quactlize/csrc/preprocess" \
  "$repo/dev/fold_derivation/l140_format_loader.cpp" \
  "$repo/quactlize/csrc/preprocess/thop/ppu_backend.cpp" \
  -ldl -pthread -o "$tmp/l140"

fake_so() {
  local marker=$1
  local output=$2
  shift 2
  local packed_format=${1:--1}
  if (($#)); then shift; fi
  "$cxx" -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC \
    -I"$repo/quactlize/include" \
    -DL140_BACKEND_MARKER="$marker" \
    -DL140_PACKED_FORMAT="$packed_format" \
    "$@" \
    "$repo/dev/fold_derivation/l140_fake_ppu_backend.cpp" -o "$output"
}

unset_formats() {
  unset QUACTLIZE_PPU_LIB_FMT0 QUACTLIZE_PPU_LIB_FMT1 QUACTLIZE_PPU_LIB_FMT2 \
        QUACTLIZE_PPU_LIB_FMT3 QUACTLIZE_PPU_LIB_FMT4
}

# BASE SPLICE: a .so suffix becomes _fmtN.so.  Every qtype is checked against
# an independent wire mapping inside l140, while the opened binary reports N.
for fmt in 0 1 2 3 4; do fake_so "$((100 + fmt))" "$tmp/base_fmt${fmt}.so" "$fmt"; done
unset_formats
QUACTLIZE_PPU_LIB="$tmp/base.so" "$tmp/l140" 100 >"$tmp/base.log"
grep -q 'qtype=10 format=2 marker=102 want=102' "$tmp/base.log"
grep -q 'qtype=11 format=3 marker=103 want=103' "$tmp/base.log"
grep -q 'qtype=12 format=0 marker=100 want=100' "$tmp/base.log"
grep -q 'qtype=13 format=1 marker=101 want=101' "$tmp/base.log"
grep -q 'qtype=14 format=4 marker=104 want=104' "$tmp/base.log"

# EXPLICIT OVERRIDE wins over those five valid base-derived libraries.
for fmt in 0 1 2 3 4; do fake_so "$((200 + fmt))" "$tmp/override${fmt}.so" "$fmt"; done
QUACTLIZE_PPU_LIB="$tmp/base.so" \
QUACTLIZE_PPU_LIB_FMT0="$tmp/override0.so" \
QUACTLIZE_PPU_LIB_FMT1="$tmp/override1.so" \
QUACTLIZE_PPU_LIB_FMT2="$tmp/override2.so" \
QUACTLIZE_PPU_LIB_FMT3="$tmp/override3.so" \
QUACTLIZE_PPU_LIB_FMT4="$tmp/override4.so" \
  "$tmp/l140" 200 >"$tmp/override.log"
grep -q 'FORMAT_MAP_AND_PATHS PASS' "$tmp/override.log"

# An explicitly present but empty override does not mask the base-derived path.
QUACTLIZE_PPU_LIB="$tmp/base.so" \
QUACTLIZE_PPU_LIB_FMT0= QUACTLIZE_PPU_LIB_FMT1= QUACTLIZE_PPU_LIB_FMT2= \
QUACTLIZE_PPU_LIB_FMT3= QUACTLIZE_PPU_LIB_FMT4= \
  "$tmp/l140" 100 >"$tmp/empty.log"
grep -q 'FORMAT_MAP_AND_PATHS PASS' "$tmp/empty.log"

# A base without .so gets _fmtN appended rather than suffix-spliced.
for fmt in 0 1 2 3 4; do fake_so "$((300 + fmt))" "$tmp/no_suffix_fmt${fmt}" "$fmt"; done
unset_formats
QUACTLIZE_PPU_LIB="$tmp/no_suffix" "$tmp/l140" 300 >"$tmp/no_suffix.log"
grep -q 'FORMAT_MAP_AND_PATHS PASS' "$tmp/no_suffix.log"

# With no environment override, dlopen resolves the documented bare names.
mkdir "$tmp/bare"
for fmt in 0 1 2 3 4; do
  fake_so "$((400 + fmt))" "$tmp/bare/libquactlize_ppu_fmt${fmt}.so" "$fmt"
done
unset_formats
env -u QUACTLIZE_PPU_LIB LD_LIBRARY_PATH="$tmp/bare${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$tmp/l140" 400 >"$tmp/bare.log"
grep -q 'FORMAT_MAP_AND_PATHS PASS' "$tmp/bare.log"

# One installed bundle directory names all six libraries. Explicit paths are
# still stronger, but a deployment no longer needs six path variables or an
# LD_LIBRARY_PATH mutation just to select the canonical set.
mkdir "$tmp/bundle"
fake_so 550 "$tmp/bundle/libquactlize_ppu.so"
for fmt in 0 1 2 3 4; do
  fake_so "$((500 + fmt))" "$tmp/bundle/libquactlize_ppu_fmt${fmt}.so" "$fmt"
done
unset_formats
env -u QUACTLIZE_PPU_LIB QUACTLIZE_PPU_BUNDLE="$tmp/bundle" \
  "$tmp/l140" 500 >"$tmp/bundle-formats.log"
grep -q 'FORMAT_MAP_AND_PATHS PASS' "$tmp/bundle-formats.log"
env -u QUACTLIZE_PPU_LIB QUACTLIZE_PPU_BUNDLE="$tmp/bundle" \
  "$tmp/l140" --default 550 >"$tmp/bundle-default.log"
grep -q 'DEFAULT_PATH PASS' "$tmp/bundle-default.log"

# An explicit base continues to outrank the installed bundle.
QUACTLIZE_PPU_BUNDLE="$tmp/bundle" QUACTLIZE_PPU_LIB="$tmp/base.so" \
  "$tmp/l140" 100 >"$tmp/bundle-precedence.log"
grep -q 'FORMAT_MAP_AND_PATHS PASS' "$tmp/bundle-precedence.log"

# load() is deliberately not format-selected: the base path is used verbatim.
fake_so 777 "$tmp/default.so"
unset_formats
QUACTLIZE_PPU_LIB="$tmp/default.so" "$tmp/l140" --default 777 >"$tmp/default.log"
grep -q 'DEFAULT_PATH PASS' "$tmp/default.log"

# NEGATIVE 1: route Q3_K through Q6_K's real binary.  The independent qtype
# anchor must detect it even though both libraries are complete and loadable.
if L140_PLANT_WRONG_MAP=1 QUACTLIZE_PPU_LIB="$tmp/base.so" "$tmp/l140" 100 \
     >"$tmp/wrong_map.log" 2>&1; then
  echo 'L140 FAIL: planted qtype -> packed-format error stayed green' >&2
  exit 1
fi
grep -q 'qtype=11 format=4 marker=104 want=103' "$tmp/wrong_map.log"
grep -q 'EXPECTED_CONTRACT_RED' "$tmp/wrong_map.log"

# NEGATIVE 2: an explicit FMT3 path points at FMT4's binary.  Precedence must
# make that planted path observable, and the per-format marker must reject it.
if QUACTLIZE_PPU_LIB="$tmp/base.so" QUACTLIZE_PPU_LIB_FMT3="$tmp/override4.so" \
     "$tmp/l140" 100 >"$tmp/wrong_path.log" 2>&1; then
  echo 'L140 FAIL: planted per-format path error stayed green' >&2
  exit 1
fi
grep -q 'qtype=11 format=3 marker=-1 want=103' "$tmp/wrong_path.log"
grep -q 'packed-format identity mismatch' "$tmp/wrong_path.log"
grep -q 'EXPECTED_CONTRACT_RED' "$tmp/wrong_path.log"

# NEGATIVE 3: the build identity is mandatory for both the default and a FMT
# slot. A complete legacy-shaped library must not become callable merely
# because all of the historical operation symbols happen to be present.
fake_so 600 "$tmp/missing_identity.so" -1 -DL140_OMIT_BUILD_IDENTITY=1
if QUACTLIZE_PPU_LIB="$tmp/missing_identity.so" "$tmp/l140" --default 600 \
     >"$tmp/missing_identity_default.log" 2>&1; then
  echo 'L140 FAIL: default library without build identity stayed green' >&2
  exit 1
fi
grep -q 'missing symbol quactlize_ppu_build_packed_format_v1' "$tmp/missing_identity_default.log"
grep -q 'DEFAULT_PATH_RED' "$tmp/missing_identity_default.log"

if QUACTLIZE_PPU_LIB="$tmp/base.so" QUACTLIZE_PPU_LIB_FMT3="$tmp/missing_identity.so" \
     "$tmp/l140" 100 >"$tmp/missing_identity_fmt.log" 2>&1; then
  echo 'L140 FAIL: FMT library without build identity stayed green' >&2
  exit 1
fi
grep -q 'qtype=11 format=3 marker=-1 want=103' "$tmp/missing_identity_fmt.log"
grep -q 'missing symbol quactlize_ppu_build_packed_format_v1' "$tmp/missing_identity_fmt.log"
grep -q 'EXPECTED_CONTRACT_RED' "$tmp/missing_identity_fmt.log"

# NEGATIVE 4: load() owns the -1 slot. A valid FMT0 library at the default
# path is still the wrong binary and must fail before exposing its operations.
fake_so 700 "$tmp/wrong_default.so" 0
if QUACTLIZE_PPU_LIB="$tmp/wrong_default.so" "$tmp/l140" --default 700 \
     >"$tmp/wrong_default.log" 2>&1; then
  echo 'L140 FAIL: FMT0 library admitted as the default build' >&2
  exit 1
fi
grep -q 'packed-format identity mismatch' "$tmp/wrong_default.log"
grep -q 'requested fmt-1, library reports fmt0' "$tmp/wrong_default.log"
grep -q 'DEFAULT_PATH_RED' "$tmp/wrong_default.log"

echo 'L140 format-loader oracle PASS: qtype map=5/5; exact default/FMT identity; explicit/base/bundle/bare/default precedence; four EXPECTED_RED classes'
