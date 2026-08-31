#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out nvcc arch source route
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L245_OUT:-/tmp/quactlize-l245-q4-n16k64-grouped}"
  nvcc="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  arch="${L245_CUDA_ARCH:-sm_80}"
  source="$root/dev/fold_derivation/l245_q4_n16k64_grouped_offline.cu"
  route="$root/quactlize/csrc/preprocess/thop/gguf_prepass_ops.cpp"
  mkdir -p "$out"
  if [[ -z "$nvcc" ]]; then
    printf '[l245-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi

  # Bind this standalone C-ABI oracle to the grouped loops that own expert
  # offsets in the real torch route.  The executable below then exercises the
  # same production layout/unit symbols without requiring a torch rebuild.
  python3 - "$route" <<'PY'
from pathlib import Path
import re
import sys

s = Path(sys.argv[1]).read_text()
checks = {
    "forward-low-stride": r"low_expert_bytes\s*=\s*size_t\(n\)\s*\*\s*k\s*\*\s*kLowBits\s*/\s*8",
    "forward-native-base": r"native_low\s*\+\s*size_t\(e\)\s*\*\s*low_expert_bytes",
    "forward-placed-base": r"placed_low\s*\+\s*size_t\(e\)\s*\*\s*low_expert_bytes",
    "grouped-units": r"prepare_units_grouped\s*\(",
    "inverse-source-base": r"auto const\* src_lo\s*=\s*placed_low\s*\+\s*size_t\(e\)\s*\*\s*low_expert_bytes",
    "inverse-destination-base": r"auto\* dst_lo\s*=\s*recovered_low\s*\+\s*size_t\(e\)\s*\*\s*low_expert_bytes",
}
missing = [name for name, pattern in checks.items() if not re.search(pattern, s)]
if missing:
    raise SystemExit("L245 source audit FAIL: " + ",".join(missing))
print("[l245-source] PASS grouped expert bases, code stride and packed-unit seam")
PY

  local -a common=(
    "$nvcc" -std=c++17 -O2 "-arch=$arch" --expt-relaxed-constexpr
    -Xcompiler=-fPIC
    "-I$root/quactlize/include"
    "-I$root/dev/fold_derivation/stub_inc"
    "-I$root/third_party/actlize/include"
    "-I$root/third_party/cutlass/include"
  )

  "${common[@]}" -c -o "$out/layout.o" \
    "$root/quactlize/csrc/device/ppu_dense_layout.cu" \
    >"$out/layout.build.log" 2>&1
  "${common[@]}" -x cu -c -o "$out/units.o" \
    "$root/quactlize/csrc/device/ppu_unit_pack.cpp" \
    >"$out/units.build.log" 2>&1

  build_one() {
    local label="$1" define="$2"
    local -a cmd=("${common[@]}")
    if [[ -n "$define" ]]; then cmd+=("-D$define=1"); fi
    cmd+=("$source" "$out/layout.o" "$out/units.o" -o "$out/$label")
    "${cmd[@]}" >"$out/$label.build.log" 2>&1
  }

  build_one l245 ""
  "$out/l245" | tee "$out/run.log"
  grep '^L245 Q4_N16K64_GROUPED_OFFLINE ' "$out/run.log" \
    >"$out/canonical.log"
  diff -u \
    "$root/dev/fold_derivation/l245_q4_n16k64_grouped_offline.expected.txt" \
    "$out/canonical.log"

  local macro label
  while read -r macro label; do
    build_one "red-$label" "$macro"
    if "$out/red-$label" >"$out/red-$label.run.log" 2>&1; then
      printf '[l245-runner] FAIL: %s negative escaped\n' "$label" >&2
      return 1
    fi
    grep '^L245 Q4_N16K64_GROUPED_OFFLINE FAIL ' \
      "$out/red-$label.run.log" >/dev/null
    printf '[l245-red] PASS plant=%s result=RED\n' "$label"
  done <<'EOF'
L245_PLANT_EXPERT_BASE_REUSE expert-base-reuse
L245_PLANT_METADATA_MUTATION metadata-mutation
EOF

  printf '[l245-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
