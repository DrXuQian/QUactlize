#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work_dir=${Q4K_PDF_VS_BC_WORK_DIR:-/workspace/quactlize-q4k-pdf-vs-bc-5090}
samples=${Q4K_PDF_VS_BC_SAMPLES:-11}
mkdir -p "$work_dir"

binary="$work_dir/q4k_pdf_vs_bc_5090"
bad_pdf_binary="$work_dir/q4k_pdf_vs_bc_5090_bad_pdf_magic"
bad_shipping_binary="$work_dir/q4k_pdf_vs_bc_5090_bad_shipping_magic"
build_log="$work_dir/build.log"
bad_pdf_build_log="$work_dir/build-bad-pdf-magic.log"
bad_shipping_build_log="$work_dir/build-bad-shipping-magic.log"
result_log="$work_dir/result.log"

common=(
  nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 --expt-relaxed-constexpr
  --ptxas-options=-v
  -I"$repo_dir/benchmarks"
  -I"$repo_dir/quactlize/include"
  -I"$repo_dir/quactlize/include/gemv_lowbit"
  -I"$repo_dir/third_party/actlize/include"
  -I"$repo_dir/dev/fold_derivation/stub_inc"
  "$repo_dir/benchmarks/q4k_pdf_vs_bc_5090.cu"
)

(set -o pipefail; "${common[@]}" -o "$binary" 2>&1 | tee "$build_log")
(set -o pipefail; "${common[@]}" -DQ4K_PDF_PLANT_WRONG_MAGIC=1 \
  -o "$bad_pdf_binary" 2>&1 | tee "$bad_pdf_build_log")
(set -o pipefail; "${common[@]}" -DQ4K_BC_PLANT_WRONG_MAGIC=1 \
  -o "$bad_shipping_binary" 2>&1 | tee "$bad_shipping_build_log")

{
  echo "git_sha=$(git -C "$repo_dir" rev-parse HEAD)"
  echo "git_status_sha256=$(git -C "$repo_dir" status --porcelain=v1 | sha256sum | awk '{print $1}')"
  echo "binary_sha256=$(sha256sum "$binary" | awk '{print $1}')"
  echo "bad_pdf_binary_sha256=$(sha256sum "$bad_pdf_binary" | awk '{print $1}')"
  echo "bad_shipping_binary_sha256=$(sha256sum "$bad_shipping_binary" | awk '{print $1}')"
  echo "source_sha256=$(sha256sum "$repo_dir/benchmarks/q4k_pdf_vs_bc_5090.cu" | awk '{print $1}')"
  echo "reference_sha256=$(sha256sum "$repo_dir/benchmarks/q4k_pdf_vs_bc_reference_nv.cuh" | awk '{print $1}')"
  echo "reader_sha256=$(sha256sum "$repo_dir/quactlize/include/gguf_bc_q4_reader.hpp" | awk '{print $1}')"
  echo "kernel_sha256=$(sha256sum "$repo_dir/quactlize/include/gguf_bc_q4_gemv.hpp" | awk '{print $1}')"
  echo "dispatch_sha256=$(sha256sum "$repo_dir/quactlize/include/gguf_bc_vecdot.hpp" | awk '{print $1}')"
  nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap,driver_version,memory.total \
    --format=csv,noheader
  nvcc --version
} > "$work_dir/identity.txt"

"$bad_pdf_binary" --arm pdf --correctness-only \
  --expect-correctness-fail pdf > "$work_dir/control-bad-pdf-magic.log" 2>&1
grep -q 'NEGATIVE-CONTROL target=pdf checks=32 red=32 configs=16 red_configs=16 verdict=EXPECTED-RED/PASS' \
  "$work_dir/control-bad-pdf-magic.log"
test "$(grep -c '^ACCURACY arm=pdf/.* verdict=FAIL$' \
  "$work_dir/control-bad-pdf-magic.log")" -eq 32

"$binary" --arm bc --correctness-only --plant-bad-bc-artifact \
  --expect-correctness-fail bc > "$work_dir/control-bad-bc-artifact.log" 2>&1
grep -q 'NEGATIVE-CONTROL target=bc checks=24 red=12 configs=12 red_configs=12 verdict=EXPECTED-RED/PASS' \
  "$work_dir/control-bad-bc-artifact.log"
test "$(grep -c '^ACCURACY arm=bc/.* verdict=FAIL$' \
  "$work_dir/control-bad-bc-artifact.log")" -eq 12

"$bad_shipping_binary" --arm shipping --correctness-only \
  --expect-correctness-fail shipping > "$work_dir/control-bad-shipping-magic.log" 2>&1
grep -q 'NEGATIVE-CONTROL target=shipping checks=24 red=24 configs=12 red_configs=12 verdict=EXPECTED-RED/PASS' \
  "$work_dir/control-bad-shipping-magic.log"
test "$(grep -c '^ACCURACY arm=shipping/.* verdict=FAIL$' \
  "$work_dir/control-bad-shipping-magic.log")" -eq 24

"$binary" --arm pdf --correctness-only > "$work_dir/control-missing-bc.log" 2>&1
grep -q 'SKIP arm=bc reason=operator-selected-pdf-only' "$work_dir/control-missing-bc.log"
grep -q 'SKIP arm=shipping reason=operator-selected-pdf-only' "$work_dir/control-missing-bc.log"
grep -q 'CORRECTNESS-ONLY verdict=PASS' "$work_dir/control-missing-bc.log"

(set -o pipefail; "$binary" --samples "$samples" 2>&1 | tee "$result_log")

test "$(grep -c '^RESULT arm=' "$result_log")" -eq 40
test "$(grep -c '^RESULT arm=pdf/' "$result_log")" -eq 16
test "$(grep -c '^RESULT arm=bc/' "$result_log")" -eq 12
test "$(grep -c '^RESULT arm=shipping/' "$result_log")" -eq 12
test "$(grep -c '^WINNER family=' "$result_log")" -eq 3
grep -q '^WINNER family=shipping ' "$result_log"
grep -q '^VERDICT legacy_bc_vs_pdf ' "$result_log"
grep -q '^VERDICT production_bc_vs_pdf ' "$result_log"
grep -q 'ARTIFACT .*pack_outside_timing=1 .*roundtrip=exact .*byte_neutral=1' "$result_log"
grep -q '^ACCURACY arm=pdf/.* input=positive .* verdict=PASS' "$result_log"
grep -q '^ACCURACY arm=pdf/.* input=signed .* verdict=PASS' "$result_log"
grep -q '^ACCURACY arm=bc/.* input=positive .* verdict=PASS' "$result_log"
grep -q '^ACCURACY arm=bc/.* input=signed .* verdict=PASS' "$result_log"
test "$(grep -c '^ACCURACY arm=pdf/.* verdict=PASS$' "$result_log")" -eq 32
test "$(grep -c '^ACCURACY arm=bc/.* verdict=PASS$' "$result_log")" -eq 24
test "$(grep -c '^ACCURACY arm=shipping/.* verdict=PASS$' "$result_log")" -eq 24
grep -q '^ACCURACY arm=shipping/A64-CtaN2-Wn4-Wk1 input=positive .* verdict=PASS$' "$result_log"
grep -q '^ACCURACY arm=shipping/A64-CtaN2-Wn4-Wk1 input=signed .* verdict=PASS$' "$result_log"

echo "[q4k-pdf-vs-bc] PASS: exact PDF reference + shipping BC artifact; 4/4 controls RED/explicit-SKIP"
echo "[q4k-pdf-vs-bc] artifacts: $work_dir"
