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
  "$cxx" -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC \
    -I"$repo/quactlize/include" \
    -DL140_BACKEND_MARKER="$marker" \
    "$repo/dev/fold_derivation/l140_fake_ppu_backend.cpp" -o "$output"
}

unset_formats() {
  unset QUACTLIZE_PPU_LIB_FMT0 QUACTLIZE_PPU_LIB_FMT1 QUACTLIZE_PPU_LIB_FMT2 \
        QUACTLIZE_PPU_LIB_FMT3 QUACTLIZE_PPU_LIB_FMT4
}

# BASE SPLICE: a .so suffix becomes _fmtN.so.  Every qtype is checked against
# an independent wire mapping inside l140, while the opened binary reports N.
for fmt in 0 1 2 3 4; do fake_so "$((100 + fmt))" "$tmp/base_fmt${fmt}.so"; done
unset_formats
QUACTLIZE_PPU_LIB="$tmp/base.so" "$tmp/l140" 100 >"$tmp/base.log"
grep -q 'qtype=10 format=2 marker=102 want=102' "$tmp/base.log"
grep -q 'qtype=11 format=3 marker=103 want=103' "$tmp/base.log"
grep -q 'qtype=12 format=0 marker=100 want=100' "$tmp/base.log"
grep -q 'qtype=13 format=1 marker=101 want=101' "$tmp/base.log"
grep -q 'qtype=14 format=4 marker=104 want=104' "$tmp/base.log"

# EXPLICIT OVERRIDE wins over those five valid base-derived libraries.
for fmt in 0 1 2 3 4; do fake_so "$((200 + fmt))" "$tmp/override${fmt}.so"; done
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
for fmt in 0 1 2 3 4; do fake_so "$((300 + fmt))" "$tmp/no_suffix_fmt${fmt}"; done
unset_formats
QUACTLIZE_PPU_LIB="$tmp/no_suffix" "$tmp/l140" 300 >"$tmp/no_suffix.log"
grep -q 'FORMAT_MAP_AND_PATHS PASS' "$tmp/no_suffix.log"

# With no environment override, dlopen resolves the documented bare names.
mkdir "$tmp/bare"
for fmt in 0 1 2 3 4; do
  fake_so "$((400 + fmt))" "$tmp/bare/libquactlize_ppu_fmt${fmt}.so"
done
unset_formats
env -u QUACTLIZE_PPU_LIB LD_LIBRARY_PATH="$tmp/bare${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$tmp/l140" 400 >"$tmp/bare.log"
grep -q 'FORMAT_MAP_AND_PATHS PASS' "$tmp/bare.log"

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
grep -q 'qtype=11 format=3 marker=204 want=103' "$tmp/wrong_path.log"
grep -q 'EXPECTED_CONTRACT_RED' "$tmp/wrong_path.log"

echo 'L140 format-loader oracle PASS: qtype map=5/5; explicit/base/bare/default precedence; wrong-map+wrong-path EXPECTED_RED'
