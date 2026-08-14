#!/usr/bin/env bash
set -u

repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
out=/workspace/quactlize-l187-bc-q4-fast-reader
mkdir -p "$out"

nvcc=${NVCC:-$(command -v nvcc 2>/dev/null || true)}
nvdisasm=${NVDISASM:-$(command -v nvdisasm 2>/dev/null || true)}
if [[ -z "$nvcc" || -z "$nvdisasm" ]]; then
  echo "L187 SKIP: nvcc and nvdisasm are required for the NVIDIA target-dispatch gate"
  exit 3
fi

common=(
  -std=c++17 -O3 -arch=sm_120 --expt-relaxed-constexpr -lineinfo
  -I "$repo_dir/dev/fold_derivation/stub_inc"
  -I "$repo_dir/quactlize/include"
  -I "$repo_dir/third_party/actlize/include"
)
source="$repo_dir/dev/fold_derivation/l187_bc_q4_fast_reader.cu"
exe="$out/l187_bc_q4_fast_reader"
cubin="$out/l187_bc_q4_fast_reader.cubin"

"$nvcc" "${common[@]}" "$source" -o "$exe" >"$out/build.log" 2>&1
build_rc=$?
if [[ $build_rc -ne 0 ]]; then
  if grep -Eq 'Unsupported gpu architecture|not found' "$out/build.log"; then
    echo "L187 SKIP: nvcc cannot build the sm_120 shipping-reader probe"
    head -n 1 "$out/build.log"
    exit 3
  fi
  echo "L187 FAIL: host/device probe did not compile"
  tail -n 30 "$out/build.log"
  exit 1
fi

"$exe" live | tee "$out/live.log"
live_rc=${PIPESTATUS[0]}
if [[ $live_rc -ne 0 ]]; then
  echo "L187 FAIL: exhaustive live oracle is red"
  exit 1
fi

"$exe" wrong-permutation >"$out/wrong-permutation.log" 2>&1
wrong_rc=$?
if [[ $wrong_rc -ne 1 ]] || ! grep -q 'PLANTED_RED wrong-permutation DETECTED' "$out/wrong-permutation.log"; then
  cat "$out/wrong-permutation.log"
  echo "L187 FAIL: the one-bit within-word permutation plant did not red"
  exit 1
fi

"$exe" missing-denominator >"$out/missing-denominator.log" 2>&1
missing_rc=$?
if [[ $missing_rc -ne 1 ]] || ! grep -q 'PLANTED_RED missing-denominator DETECTED' "$out/missing-denominator.log"; then
  cat "$out/missing-denominator.log"
  echo "L187 FAIL: the missing supported-arrangement denominator plant did not red"
  exit 1
fi

"$nvcc" "${common[@]}" -cubin "$source" -o "$cubin" >"$out/cubin-build.log" 2>&1
cubin_rc=$?
if [[ $cubin_rc -ne 0 ]]; then
  echo "L187 FAIL: exact sm_120 device probe did not produce a cubin"
  tail -n 30 "$out/cubin-build.log"
  exit 1
fi
"$nvdisasm" -gi -c "$cubin" >"$out/l187.sass" 2>"$out/nvdisasm.log"
disasm_rc=$?
if [[ $disasm_rc -ne 0 ]]; then
  echo "L187 FAIL: nvdisasm rejected the exact sm_120 cubin"
  cat "$out/nvdisasm.log"
  exit 1
fi
if grep -Eqi 'ppu\.' "$out/l187.sass"; then
  echo "L187 FAIL: NVIDIA device image contains a PPU-prefixed mnemonic"
  grep -Ein 'ppu\.' "$out/l187.sass" | head
  exit 1
fi
symbols=$(grep -c 'device_binding_probe' "$out/l187.sass" || true)
if [[ $symbols -lt 4 ]]; then
  echo "L187 FAIL: not all four Q4 ArtifactTileK device bindings reached sm_120 codegen (markers=$symbols)"
  exit 1
fi
mask_low=$(grep -Ec 'LOP3.LUT.*0xf000f,' "$out/l187.sass" || true)
mask_high=$(grep -Ec 'LOP3.LUT.*0xf000f0' "$out/l187.sass" || true)
fma_1024=$(grep -Ec 'HFMA2.*-1024, -1024' "$out/l187.sass" || true)
fma_64=$(grep -Ec 'HFMA2.*-64, -64' "$out/l187.sass" || true)
if [[ $mask_low -lt 4 || $mask_low -ne $mask_high ||
      $mask_low -ne $fma_1024 || $mask_low -ne $fma_64 ]]; then
  echo "L187 FAIL: NVIDIA fast-dequant codegen lost a mask/magic arithmetic level "\
       "(lop3-low=$mask_low lop3-high=$mask_high fma-1024=$fma_1024 fma-64=$fma_64)"
  exit 1
fi

echo "L187 PASS: 4/4 Q4 arrangements, 1048576 coordinates, two semantic plants RED; "\
     "sm_120 target branch has balanced LOP3/HFMA2 and no ppu.* mnemonic"
echo "L187 artifacts: $out"
