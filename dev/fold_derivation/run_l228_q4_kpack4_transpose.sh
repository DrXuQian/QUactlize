#!/usr/bin/env bash
set -euo pipefail

main() {
  local repo base out compiler
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  base="${QUACTLIZE_L228_OUT:-/workspace/quactlize-l228-q4-kpack4-transpose}"
  out="${base}/run-$$"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l228-runner] FAIL: nvcc is unavailable; K-pack4 proof did not run\n' >&2
    return 2
  fi

  # Bind the host composition to the actual PPU0010 transport pair.  The host
  # build deliberately uses stub_inc first: it must not rediscover the known
  # hggc_fp8.h host-toolchain failure while proving architecture-independent
  # CuTe layouts.
  python3 - "$repo" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
copy = (root / "third_party/actlize/include/cute/arch/copy_ppu0010_aiu.hpp").read_text()
operand = (root / "third_party/actlize/include/cutlass/gemm/config/gemm_operands.hpp").read_text()
tests = (root / "third_party/actlize/test/unit/gemm_ppu/device/ppu0010_gemm_f16_f16_f32_tensor_op_f32_aiu.cu").read_text()

required_copy = (
    "PPU0010_AIU_LOAD<NumBitsPerTMA, Element, true, Swzl",
    "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16",
    "ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16",
)
for token in required_copy:
    if token not in copy:
        raise SystemExit(f"[l228-source] RED: missing PPU0010 transposed transport token: {token}")
if copy.count("ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16") != 2:
    raise SystemExit("[l228-source] RED: fp16/bf16 transposed reader denominator changed")

required_operand = (
    "struct DefaultGemm_AIU_Operand<",
    "static constexpr int BlockContSize = Block_MN{} * sizeof(Element);",
    "Layout<Shape<Int<CUBE_W>, Int<CUBE_H>>, Stride<_1, Int<CUBE_W>>>",
)
for token in required_operand:
    if token not in operand:
        raise SystemExit(f"[l228-source] RED: transposed operand contract drifted: {token}")

# The existing fp16 GEMM suite is the device authority that the matched b16
# write/read pair presents B RowMajor bytes to the ordinary MMA B fragment.
if tests.count("cutlass::half_t, cutlass::layout::RowMajor") < 4:
    raise SystemExit("[l228-source] RED: transposed fp16 GEMM authority disappeared")
print("[l228-source] PASS matched-b16-transpose=BOUND fp16-rowmajor-device-authority=BOUND")
PY

  local -a common=(
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
    -I "$repo/dev/fold_derivation/stub_inc"
    -I "$repo/third_party/actlize/include"
    -I "$repo/quactlize/include"
  )
  local source="$repo/dev/fold_derivation/l228_q4_kpack4_transpose.cu"
  "$compiler" "${common[@]}" "$source" -o "$out/l228" \
    >"$out/build.log" 2>&1 || {
      printf '[l228-runner] FAIL: canonical K-pack4 proof did not compile\n' >&2
      tail -n 120 "$out/build.log" >&2
      return 2
    }
  "$out/l228" | tee "$out/run.log"
  grep -Fqx \
    'L228 KPACK4_CANONICAL PASS roundtrip_bad=0 traversal_bad=0 transports=44 tk32_shared_transports=16 tail_bad=0 tail_valid=1728 tail_padded=2368 metadata_bad=0 mma_bad=0 existing_converter_exact=1024/1024 converter_bad=0 converter_words_same_n=256/256 native_kquartet=256/256 same_gs32=256/256 source_dup=0 destination_dup=0 negative_controls=RED bytes_per_n16k64=512' \
    "$out/run.log"

  local name define
  while read -r name define; do
    "$compiler" "${common[@]}" "$define" "$source" -o "$out/red-$name" \
      >"$out/red-$name.build.log" 2>&1 || {
        printf '[l228-runner] FAIL: RED plant %s did not compile\n' "$name" >&2
        return 2
      }
    if "$out/red-$name" >"$out/red-$name.run.log" 2>&1; then
      printf '[l228-runner] FAIL: RED plant escaped: %s\n' "$name" >&2
      return 1
    fi
    grep -F 'L228 KPACK4_CANONICAL FAIL' "$out/red-$name.run.log" >/dev/null
    printf '[l228-red] PASS plant=%s result=RED\n' "$name"
  done <<'EOF'
naive-consecutive-pack -DL228_NAIVE_CONSECUTIVE_PACK=1
rotated-converter-destination -DL228_ROTATE_CONVERTER_DESTINATION=1
shifted-metadata-atom -DL228_SHIFT_METADATA_ATOM=1
EOF

  sha256sum "$source" >"$out/source.sha256"
  printf '[l228-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
