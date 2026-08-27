#!/usr/bin/env bash
set -euo pipefail

main() {
  local repo base out compiler
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  base="${QUACTLIZE_L231_OUT:-/workspace/quactlize-l231-q4-kpack4-production-fragment}"
  out="${base}/run-$$"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l231-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi

  python3 - "$repo" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
source = (root / "quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp").read_text()
required = (
    "if constexpr (kQ4KPack4Transpose)",
    "PPU_Q4_KPACK4_LEGACY_LOADER_OUTPUT_LAYOUT",
    "compact_col_major(\n            shape<1>(cvt_in.layout()), stride<1>(tCrB_mma.layout()))",
    "class TensorLayoutIn,\n            class TensorLayoutOut",
    "static_assert(N == NOut",
)
for token in required:
    if token not in source:
        raise SystemExit(f"[l231-source] RED: missing production destination seam: {token!r}")
wrong = "shape<1>(cvt_in.layout()), stride<1>(cvt_in.layout()))"
if wrong in source:
    raise SystemExit("[l231-source] RED: production destination still borrows the loader N stride")
planted = source.replace(
    "shape<1>(cvt_in.layout()), stride<1>(tCrB_mma.layout()))",
    wrong,
    1,
)
if planted == source or wrong not in planted:
    raise SystemExit("[l231-source] RED: wrong-stride plant did not fire")
builder = (root / "quactlize/include/actlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl").read_text()
n16_required = (
    "PPU_Q4_KPACK4_N16_DELIVERY",
    "using DeliveryN = Int<q4_kpack4::kTransportN>;",
    "static constexpr int InstNum = Block_N{} / DeliveryN{};",
    "DeliveryN{} * PhysicalBlockK{} * cutlass::sizeof_bits<TransportElement>::value",
    "TransportElement, PhysicalBlockK{}, DeliveryN{}, Swap, true, InstNum",
    "Shape<DeliveryN, PhysicalBlockK>, Stride<_1, DeliveryN>",
)
for index, token in enumerate(n16_required):
    expected_count = 2 if index == 0 else 1
    if builder.count(token) != expected_count:
        raise SystemExit(f"[l231-source] RED: N16 production seam differs: {token!r}")
planted_builder = builder.replace(
    "static constexpr int InstNum = Block_N{} / DeliveryN{};",
    "static constexpr int InstNum = 1;", 1)
if planted_builder == builder or n16_required[2] in planted_builder:
    raise SystemExit("[l231-source] RED: N16 one-cube plant did not fire")
print("[l231-source] PASS compute-N-stride=BOUND legacy-negative=BOUND "
      "separate-layout-converter=BOUND n16-production-type=BOUND")
PY

  local -a common=(
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
    -I "$repo/dev/fold_derivation/stub_inc"
    -I "$repo/third_party/actlize/include"
    -I "$repo/quactlize/include"
  )
  local source="$repo/dev/fold_derivation/l231_q4_kpack4_production_fragment.cu"
  "$compiler" "${common[@]}" "$source" -o "$out/l231" >"$out/build.log" 2>&1 || {
    if grep -F 'hggc_fp8.h' "$out/build.log" >/dev/null; then
      printf '[l231-runner] SKIP: nvcc delegates to the PPU frontend; use committed host evidence\n'
      return 0
    fi
    printf '[l231-runner] FAIL: production-fragment oracle did not compile\n' >&2
    tail -n 120 "$out/build.log" >&2
    return 2
  }
  "$out/l231" | tee "$out/run.log"
  grep -Fqx 'L231 KPACK4_PRODUCTION_FRAGMENT PASS' "$out/run.log"
  grep -E '^L231 (GEOMETRY|KPACK4_PRODUCTION_FRAGMENT)' "$out/run.log" \
    >"$out/canonical.log"
  diff -u \
    "$repo/dev/fold_derivation/l231_q4_kpack4_production_fragment.expected.txt" \
    "$out/canonical.log"

  "$compiler" "${common[@]}" -DL231_ROTATE_DESTINATION=1 "$source" \
    -o "$out/red-rotate" >"$out/red-rotate.build.log" 2>&1
  if "$out/red-rotate" >"$out/red-rotate.run.log" 2>&1; then
    printf '[l231-runner] FAIL: rotated-destination negative escaped\n' >&2
    return 1
  fi
  grep -Fqx 'L231 KPACK4_PRODUCTION_FRAGMENT FAIL' "$out/red-rotate.run.log"
  printf '[l231-red] PASS plant=rotated-destination result=RED\n'

  "$compiler" "${common[@]}" -DL231_LEGACY_CANDIDATE=1 "$source" \
    -o "$out/red-loader-stride" >"$out/red-loader-stride.build.log" 2>&1
  if "$out/red-loader-stride" >"$out/red-loader-stride.run.log" 2>&1; then
    printf '[l231-runner] FAIL: legacy-loader-stride negative escaped\n' >&2
    return 1
  fi
  grep -Fqx 'L231 KPACK4_PRODUCTION_FRAGMENT FAIL' "$out/red-loader-stride.run.log"
  printf '[l231-red] PASS plant=legacy-loader-stride result=RED\n'

  # Same offline bytes and logical fragment, but four N16 delivery cubes in
  # place of one N64 cube.  Every geometry becomes directly identity-mapped;
  # this is the host CuTe admission gate for the device cadence A/B.
  "$compiler" "${common[@]}" -DL231_KPACK4_CUBE_N=16 "$source" \
    -o "$out/n16-delivery" >"$out/n16-delivery.build.log" 2>&1
  "$out/n16-delivery" >"$out/n16-delivery.run.log"
  grep -Fqx 'L231 KPACK4_PRODUCTION_FRAGMENT PASS' "$out/n16-delivery.run.log"
  if [ "$(grep -Ec '^L231 GEOMETRY .*current=IDENTITY .*candidate=IDENTITY .*result=PASS$' \
          "$out/n16-delivery.run.log")" -ne 12 ]; then
    printf '[l231-runner] FAIL: N16 delivery identity denominator differs\n' >&2
    grep '^L231 GEOMETRY ' "$out/n16-delivery.run.log" >&2
    return 1
  fi
  printf '[l231-n16] PASS geometries=12/12 current=IDENTITY candidate=IDENTITY\n'
  printf '[l231-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
