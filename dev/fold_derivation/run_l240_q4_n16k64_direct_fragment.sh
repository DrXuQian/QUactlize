#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out compiler source
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L240_OUT:-/tmp/quactlize-l240-q4-n16k64-direct-fragment}"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l240-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi

  python3 - "$root" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
collective = (root / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp").read_text()
converter = (root / "quactlize/include/actlize_extensions/cutlass/quactlize_mix_gemm_convert.h").read_text()
for token in (
    "class TensorLayoutIn,\n            class TensorLayoutOut",
    "using Converter = cutlass::MixGemmNumericArrayConverter<DstType, SrcType, ConversionVectorWidth>;",
    "*dst_array_ptr = Converter::convert(*src_array_ptr);",
):
    if token not in collective:
        raise SystemExit(f"[l240-source] RED: production convert_tensor seam changed: {token!r}")
pin = "mixgemm_int4_composed(c, v) != MixGemmEmit<4>::index(c, v)"
if pin not in converter:
    raise SystemExit("[l240-source] RED: int4 fast converter is no longer pinned to MixGemmEmit")
print("[l240-source] PASS convert_tensor=BOUND separate-layouts=BOUND int4-emission=BOUND")
PY

  source="$root/dev/fold_derivation/l240_q4_n16k64_direct_fragment.cu"
  local -a common=(
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
    -I "$root/dev/fold_derivation/stub_inc"
    -I "$root/third_party/actlize/include"
    -I "$root/quactlize/include"
  )

  if ! "$compiler" "${common[@]}" -DL240_COMPILER_PROBE=1 "$source" \
       -o "$out/compiler-probe" >"$out/compiler-probe.log" 2>&1; then
    printf '[l240-runner] SKIP: nvcc cannot compile the CUDA host-oracle probe\n'
    return 0
  fi

  "$compiler" "${common[@]}" "$source" -o "$out/l240" \
    >"$out/build.log" 2>&1 || {
      if grep -F 'hggc_fp8.h' "$out/build.log" >/dev/null; then
        printf '[l240-runner] SKIP: nvcc delegates to the PPU frontend; use committed host evidence\n'
        return 0
      fi
      printf '[l240-runner] FAIL: direct-fragment oracle did not build\n' >&2
      tail -n 160 "$out/build.log" >&2
      return 2
    }
  "$out/l240" | tee "$out/run.log"
  grep -E '^L240 (FRAGMENT|MAP_ANCHOR|REVERSIBLE_MAP|OFFLINE_ABI|Q4_N16K64_DIRECT_FRAGMENT)' \
    "$out/run.log" >"$out/canonical.log"
  diff -u \
    "$root/dev/fold_derivation/l240_q4_n16k64_direct_fragment.expected.txt" \
    "$out/canonical.log"

  local macro label
  while read -r macro label; do
    "$compiler" "${common[@]}" -D"$macro"=1 "$source" \
      -o "$out/red-$label" >"$out/red-$label.build.log" 2>&1
    if "$out/red-$label" >"$out/red-$label.run.log" 2>&1; then
      printf '[l240-runner] FAIL: %s negative escaped\n' "$label" >&2
      return 1
    fi
    grep -E '^L240 Q4_N16K64_DIRECT_FRAGMENT FAIL ' \
      "$out/red-$label.run.log" >/dev/null
    printf '[l240-red] PASS plant=%s result=RED\n' "$label"
  done <<'EOF'
L240_PLANT_SOURCE_EQUALS_DEST source-equals-destination
L240_PLANT_ROTATE_COHORT rotated-cohort
EOF
  printf '[l240-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
