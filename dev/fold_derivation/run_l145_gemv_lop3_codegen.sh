#!/usr/bin/env bash
# Compile and disassemble the REAL shipping int4 GEMV specialization on NVIDIA sm_120.
#
# This is a code-generation gate, not a benchmark and not a PPU claim. It answers whether nvcc has already
# collapsed gemv_converter.hpp's `(w >> shift) & mask | magic` into one ternary operation per half2 pair. Seeing
# the word LOP3 is insufficient: NVIDIA also uses LOP3 for an ordinary AND or OR, and that is exactly what the
# current code does twice per pair.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="${L145_SOURCE:-$HERE/l145_gemv_lop3_codegen.cu}"
BACKEND="${L145_BACKEND:-$ROOT/quactlize/csrc/device/ppu_backend.cu}"
NVCC="${NVCC:-$(command -v nvcc || true)}"
NVDISASM="${NVDISASM:-$(command -v nvdisasm || true)}"
CUOBJDUMP="${CUOBJDUMP:-$(command -v cuobjdump || true)}"

if [[ -z "$NVCC" || -z "$NVDISASM" ]]; then
  echo "L145 SKIP: nvcc and nvdisasm are required for the sm_120 codegen audit"
  exit 3
fi

# Bind the probe to the shipping qtype=12 route. A standalone type with the same spelling would only prove
# itself; these anchors fail closed if production changes StepK/threads or ceases to launch CtaN=8/Chunk=2.
[[ $(grep -Fc 'case 12: return RUN(Int4,  16, 128);' "$BACKEND") -eq 1 ]] || {
  echo "L145 FAIL: production qtype=12 device route is no longer exactly Int4/s16/t128" >&2
  exit 1
}
[[ $(grep -Fc 'bool const launched = ppu_gemv::launch_gemv<D, 8, 2>(p, stream);' "$BACKEND") -eq 1 ]] || {
  echo "L145 FAIL: production lowbit_device no longer launches CtaN=8/Chunk=2" >&2
  exit 1
}

OUT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-l145.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT
mkdir -p "$OUT/keep"

set +e
"$NVCC" -std=c++17 -O3 -arch=sm_120 -lineinfo \
  -I"$ROOT/quactlize/include" -I"$ROOT/third_party/cutlass/include" \
  --keep --keep-dir "$OUT/keep" -cubin "$SRC" -o "$OUT/l145.cubin" \
  >"$OUT/nvcc.log" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  first="$(grep -m1 -E 'fatal error:|error:' "$OUT/nvcc.log" || tail -n 1 "$OUT/nvcc.log")"
  echo "L145 SKIP: nvcc cannot build the sm_120 production probe: $first"
  exit 3
fi

"$NVDISASM" -gi -c "$OUT/l145.cubin" > "$OUT/l145.sass"

count() { grep -Ec "$1" "$OUT/l145.sass" || true; }
mask_lop3="$(count 'LOP3\.LUT .*0xf000f, RZ, 0xc0')"
magic_lop3="$(count 'LOP3\.LUT .*0x64006400, RZ, 0xfc')"
shift4="$(count 'SHF\.R\.U32\.HI .*0x4,')"
shift8="$(count 'SHF\.R\.U32\.HI .*0x8,')"
shift12="$(count 'SHF\.R\.U32\.HI .*0xc,')"
offset_hadd2="$(count 'HADD2 .* -1024, -1024')"
source_marks="$(count 'File .*gemv_converter\.hpp.*, line 70')"

pairs=64
shifts=$((shift4 + shift8 + shift12))
lop3_total=$((mask_lop3 + magic_lop3))
extract_total=$((lop3_total + shifts))

[[ $source_marks -gt 0 ]] || {
  echo "L145 FAIL: disassembly lost the gemv_converter.hpp:70 source binding" >&2
  exit 1
}
[[ $mask_lop3 -eq 64 && $magic_lop3 -eq 64 ]] || {
  echo "L145 FAIL: expected separate mask/magic LOP3 counts 64/64, got $mask_lop3/$magic_lop3" >&2
  exit 1
}
[[ $shift4 -eq 16 && $shift8 -eq 16 && $shift12 -eq 16 ]] || {
  echo "L145 FAIL: expected nonzero shifts 4/8/12 each 16 times, got $shift4/$shift8/$shift12" >&2
  exit 1
}
[[ $offset_hadd2 -eq 64 ]] || {
  echo "L145 FAIL: expected 64 separate offset HADD2 instructions, got $offset_hadd2" >&2
  exit 1
}

# A naive check that reports "fused" merely because LOP3 exists would see 128 LOP3 for 64 pairs. Make that
# false-positive shape explicit: fused means ONE ternary instruction and ZERO separate shifts per pair, neither
# of which is true here.
[[ $lop3_total -ne $pairs || $shifts -ne 0 ]] || {
  echo "L145 FAIL: converter unexpectedly became one-LOP3/no-shift; update TODO #28 and this pinned verdict" >&2
  exit 1
}

regs="unknown"
if [[ -n "$CUOBJDUMP" ]]; then
  regs="$("$CUOBJDUMP" --dump-resource-usage "$OUT/l145.cubin" 2>/dev/null |
            sed -n 's/.*REG:\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  [[ -n "$regs" ]] || regs="unknown"
fi

printf 'L145 shipping=int4/native s16/t128 artifact256 M1/CtaN8/Chunk2/gs32 pairs=%d\n' "$pairs"
printf 'L145 SASS mask_lop3=%d magic_lop3=%d shifts=%d [%d,%d,%d] offset_hadd2=%d regs=%s\n' \
       "$mask_lop3" "$magic_lop3" "$shifts" "$shift4" "$shift8" "$shift12" "$offset_hadd2" "$regs"
printf 'L145 extraction=%d/%d = %.2f integer instructions/pair; p0=2, p1..3=3; fused target=1\n' \
       "$extract_total" "$pairs" "$(awk -v n="$extract_total" -v d="$pairs" 'BEGIN{print n/d}')"
echo "L145 PASS: sm_120 nvcc leaves mask LOP3 + magic LOP3 + 3/4-pair shift; #28 premise holds on RTX 5090 only"
