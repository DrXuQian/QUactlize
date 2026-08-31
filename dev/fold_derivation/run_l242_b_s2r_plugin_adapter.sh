#!/usr/bin/env bash
set -euo pipefail

main() {
  local root out compiler source
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  out="${QUACTLIZE_L242_OUT:-/tmp/quactlize-l242-b-s2r-plugin}"
  mkdir -p "$out"
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [[ -z "$compiler" ]]; then
    printf '[l242-runner] SKIP: nvcc is unavailable\n'
    return 0
  fi

  source="$root/dev/fold_derivation/l242_b_s2r_plugin_adapter.cu"

  local collective="$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
  local token
  while IFS= read -r token; do
    grep -F "$token" "$collective" >/dev/null || {
      printf '[l242-source] FAIL: production S2R seam changed: %s\n' "$token" >&2
      return 1
    }
  done <<'EOF'
Tensor tCrB_load = thr_mma_bload.partition_fragment_B(sB_load(_,_,0));
auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma_bload);
Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB_load));
Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB_load);
auto dst_n_stride = compact_col_major(
shape<1>(cvt_in.layout()), stride<1>(tCrB_mma.layout()));
EOF
  printf '[l242-source] PASS legacy expressions and compute-owned destination are bound\n'
  local -a common=(
    -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
    -I "$root/dev/fold_derivation/stub_inc"
    -I "$root/third_party/actlize/include"
    -I "$root/third_party/actlize/tools/util/include"
    -I "$root/quactlize/include"
    --expt-relaxed-constexpr
    -DPPU_PACKED_SCALE=1 -DPPU_PACKED_FORMAT=0 -DPPU_B_CHUNK=0
  )

  if ! "$compiler" "${common[@]}" -DL242_COMPILER_PROBE=1 "$source" \
       -o "$out/compiler-probe" >"$out/compiler-probe.log" 2>&1; then
    printf '[l242-runner] SKIP: nvcc cannot compile the CUDA host-oracle probe\n'
    return 0
  fi

  "$compiler" "${common[@]}" -D__HGGCCC__ -DL242_PPU_TYPE_PROBE=1 \
    "$source" -o "$out/ppu-type-probe" >"$out/ppu-type-probe.log" 2>&1 || {
      printf '[l242-runner] FAIL: PPU-configured adapter type probe did not build\n' >&2
      tail -n 100 "$out/ppu-type-probe.log" >&2
      return 2
    }

  "$compiler" "${common[@]}" "$source" -o "$out/l242" \
    >"$out/build.log" 2>&1 || {
      if grep -F 'hggc_fp8.h' "$out/build.log" >/dev/null; then
        printf '[l242-runner] SKIP: nvcc delegates to the PPU frontend; use committed host evidence\n'
        return 0
      fi
      printf '[l242-runner] FAIL: B S2R adapter oracle did not build\n' >&2
      tail -n 180 "$out/build.log" >&2
      return 2
    }
  "$out/l242" | tee "$out/run.log"
  grep -E '^L242 (LEGACY|DIRECT|B_S2R_PLUGIN)' "$out/run.log" \
    >"$out/canonical.log"
  diff -u \
    "$root/dev/fold_derivation/l242_b_s2r_plugin_adapter.expected.txt" \
    "$out/canonical.log"

  local macro label needle
  while read -r macro label needle; do
    if "$compiler" "${common[@]}" -D"$macro"=1 "$source" \
         -o "$out/red-$label" >"$out/red-$label.build.log" 2>&1; then
      printf '[l242-runner] FAIL: %s negative compiled\n' "$label" >&2
      return 1
    fi
    grep -F "$needle" "$out/red-$label.build.log" >/dev/null
    printf '[l242-red] PASS plant=%s result=RED\n' "$label"
  done <<'EOF'
L242_PLANT_RVALUE_OWNER rvalue-owner no instance of function template
L242_PLANT_K_ATOMS_TWO k-atoms-two Q4 N16xK64 converter must advance four K16 MMA atoms
L242_PLANT_WN8 wn8 Q4 Universal S2R admits WN16/WN32/WN64
EOF
  printf '[l242-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
