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
    # Q4 now shares the one-plane b16 transport implementation with Q2 while
    # retaining its historical schedule/type aliases.  The destination seam
    # is therefore guarded by the generic physical-provider fact.
    "if constexpr (kKPackTranspose)",
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
print("[l231-source] PASS compute-N-stride=BOUND legacy-negative=BOUND separate-layout-converter=BOUND")
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

  for delivery_n in 32 16; do
    "$compiler" "${common[@]}" \
      -DL231_KPACK4_DELIVERY_N="$delivery_n" "$source" \
      -o "$out/l231-d$delivery_n" \
      >"$out/build-d$delivery_n.log" 2>&1
    "$out/l231-d$delivery_n" | tee "$out/run-d$delivery_n.log"
    grep -Fqx 'L231 KPACK4_PRODUCTION_FRAGMENT PASS' \
      "$out/run-d$delivery_n.log"
    if [ "$(grep -c 'candidate=IDENTITY' "$out/run-d$delivery_n.log")" -ne 12 ]; then
      printf '[l231-runner] FAIL: D%s candidate identity denominator differs\n' \
        "$delivery_n" >&2
      return 2
    fi
    printf '[l231-delivery] PASS D=%s geometries=12 candidate=IDENTITY\n' \
      "$delivery_n"
  done

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
  printf '[l231-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
