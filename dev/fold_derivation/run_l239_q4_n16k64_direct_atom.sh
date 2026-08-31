#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out compiler source
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L239_OUT:-/tmp/quactlize-l239-q4-n16k64-direct}"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l239-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi

  source="$root/dev/fold_derivation/l239_q4_n16k64_direct_atom.cu"
  local -a common=(
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
    -I "$root/dev/fold_derivation/stub_inc"
    -I "$root/third_party/actlize/include"
    -I "$root/quactlize/include"
  )

  # A real compiler-boundary probe.  `which nvcc` is insufficient on a PPU
  # box because that executable may delegate device preprocessing to hgcc.
  if ! "$compiler" "${common[@]}" -DL239_COMPILER_PROBE=1 "$source" \
       -o "$out/compiler-probe" >"$out/compiler-probe.log" 2>&1; then
    printf '[l239-runner] SKIP: nvcc cannot compile the CUDA host-oracle probe\n'
    return 0
  fi

  "$compiler" "${common[@]}" "$source" -o "$out/l239" \
    >"$out/build.log" 2>&1 || {
      if grep -F 'hggc_fp8.h' "$out/build.log" >/dev/null; then
        printf '[l239-runner] SKIP: nvcc delegates to the PPU frontend; use committed host evidence\n'
        return 0
      fi
      printf '[l239-runner] FAIL: N16xK64 direct-atom oracle did not build\n' >&2
      tail -n 120 "$out/build.log" >&2
      return 2
    }

  "$out/l239" | tee "$out/run.log"
  grep -Fqx \
    'L239 N16_K64_DIRECT_ATOM PASS chains=2 lanes=32 vectors=32 words=128 cta_threads=128 reds=6' \
    "$out/run.log"
  diff -u \
    "$root/dev/fold_derivation/l239_q4_n16k64_direct_atom.expected.txt" \
    "$out/run.log"

  local macro label sentinel
  while read -r macro label sentinel; do
    if "$compiler" "${common[@]}" -D"$macro"=1 "$source" \
         -o "$out/red-$label" >"$out/red-$label.build.log" 2>&1; then
      printf '[l239-runner] FAIL: %s negative escaped\n' "$label" >&2
      return 1
    fi
    grep -F "$sentinel" "$out/red-$label.build.log" >/dev/null
    printf '[l239-red] PASS plant=%s result=RED\n' "$label"
  done <<'EOF'
L239_PLANT_STAGE_BYTES_MISMATCH stage-bytes QUACTLIZE_B_DELIVERY_STAGE_BYTES_MISMATCH
L239_PLANT_N_ATOM_MISMATCH n-atom QUACTLIZE_B_DELIVERY_N_ATOM_MISMATCH
L239_PLANT_K_ATOM_MISMATCH k-atom QUACTLIZE_B_DELIVERY_K_ATOM_MISMATCH
L239_PLANT_ALIGNMENT_MISMATCH alignment QUACTLIZE_B_DELIVERY_ALIGNMENT_MISMATCH
L239_PLANT_SOURCE_DEST_LAYOUT_MIX source-dest-layout QUACTLIZE_B_DELIVERY_LAYOUT_MISMATCH
L239_PLANT_THREAD_COUNT_MISMATCH thread-count Q4 cp.async CTA thread count must exactly partition stage vectors
EOF
  printf '[l239-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
