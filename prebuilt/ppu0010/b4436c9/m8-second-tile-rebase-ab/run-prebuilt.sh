#!/usr/bin/env bash
# Run a compile-free A/B for the TM8 second-tile A descriptor.  This is a
# diagnostic: every terminal condition prints a verdict and returns zero so a
# failed probe cannot terminate a caller's interactive container shell.
set -u -o pipefail

EXPECTED_SOURCE=b4436c9befcc3387e371e38c58852c8f8ce768cf
EXPECTED_ACTLIZE=8d46b758c8931807df840a6ed87d272d74a8fdf4
EXPECTED_CUTLASS=f94ec46f4f63f96003d6cfdf2014731e7672c281
EXPECTED_MANIFEST=1d390f8c55040e65114845b6c52112faf3c1f2d042510798beef0ffbc3743d08
EXPECTED_ROWS=3fe38aef2bed75423524e49dd0e8f2913a982cec515d0c0e28d4a3b72345800d
EXPECTED_GENERATED=eb19f1fb2e2140796a76d63b06acb0d2d313dc37e580ccaf2b1648faf8dd8afb
EXPECTED_SDK='PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e sha256=63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd'
EXPECTED_COMPILER='hgcc Release version 2.1.1-a5c56e built=2026-07-25T09:15:42 sha256=fa62c590c67411c23fa4028f15fa562b39ce0cf830830d038a1ec04c59d8c76e'

usage() {
  printf 'usage: CUDA_VISIBLE_DEVICES=N bash %s --ppu-sdk PATH --output NEW_DIR\n' "$0"
}

diagnostic_fail() {
  printf 'M8_SECOND_TILE_REBASE_AB verdict=INFRASTRUCTURE_FAIL reason=%s logs=%s\n' "$1" "${OUTPUT_DIR:-NONE}"
  exit 0
}

PPU_SDK_ROOT=
OUTPUT_DIR=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --ppu-sdk) [ "$#" -ge 2 ] || { usage; exit 0; }; PPU_SDK_ROOT=$2; shift 2 ;;
    --output) [ "$#" -ge 2 ] || { usage; exit 0; }; OUTPUT_DIR=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; diagnostic_fail "unknown_argument:$1" ;;
  esac
done

[ -n "$PPU_SDK_ROOT" ] || diagnostic_fail missing_ppu_sdk
[ -n "$OUTPUT_DIR" ] || diagnostic_fail missing_output
case "${CUDA_VISIBLE_DEVICES:-}" in
  ''|*,*|*[!0-9]*) diagnostic_fail CUDA_VISIBLE_DEVICES_must_be_one_numeric_ordinal ;;
esac
[ ! -e "$OUTPUT_DIR" ] || diagnostic_fail output_already_exists
mkdir -p "$OUTPUT_DIR" || diagnostic_fail cannot_create_output

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || diagnostic_fail cannot_resolve_bundle
SOURCE_ROOT="$(git -C "$BUNDLE_DIR" rev-parse --show-toplevel 2>/dev/null)" || diagnostic_fail bundle_not_in_git
MANIFEST="$BUNDLE_DIR/manifest.json"
ROWS="$BUNDLE_DIR/inputs/tilem-boundary.txt"
GENERATED="$BUNDLE_DIR/inputs/generated-manifest.json"
VERIFIER="$SOURCE_ROOT/tools/verify_prebuilt_ppu_bundle.py"

[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD^ 2>/dev/null)" = "$EXPECTED_SOURCE" ] || diagnostic_fail wrong_artifact_parent
[ "$(git -C "$SOURCE_ROOT" rev-parse "$EXPECTED_SOURCE:third_party/actlize" 2>/dev/null)" = "$EXPECTED_ACTLIZE" ] || diagnostic_fail wrong_actlize
[ "$(git -C "$SOURCE_ROOT" rev-parse "$EXPECTED_SOURCE:third_party/cutlass" 2>/dev/null)" = "$EXPECTED_CUTLASS" ] || diagnostic_fail wrong_cutlass
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$EXPECTED_MANIFEST" ] || diagnostic_fail wrong_manifest_hash
[ "$(sha256sum "$ROWS" | awk '{print $1}')" = "$EXPECTED_ROWS" ] || diagnostic_fail wrong_rows_hash
[ "$(sha256sum "$GENERATED" | awk '{print $1}')" = "$EXPECTED_GENERATED" ] || diagnostic_fail wrong_generated_manifest_hash

verify_common=(
  "$MANIFEST" --qtype 12 --expect-source "$EXPECTED_SOURCE"
  --expect-submodule "third_party/actlize=$EXPECTED_ACTLIZE"
  --expect-submodule "third_party/cutlass=$EXPECTED_CUTLASS"
  --expect-sdk "$EXPECTED_SDK" --expect-compiler "$EXPECTED_COMPILER"
  --expect-arch ppu0010
)
XPLANE_BASE="$(python3 -B "$VERIFIER" "${verify_common[@]}" --role m8-xplane-baseline --target test_ppu_m8n16_collective)" || diagnostic_fail verify_xplane_baseline
XPLANE_REBASE="$(python3 -B "$VERIFIER" "${verify_common[@]}" --role m8-xplane-rebase --target test_ppu_m8n16_collective --expect-ppu-def PPU_M8_A_GMEM_TILE_REBASE=1)" || diagnostic_fail verify_xplane_rebase
KPACK_BASE="$(python3 -B "$VERIFIER" "${verify_common[@]}" --role m8-kpack-baseline --target test_fully_quantized_grouped_kpack_discovery)" || diagnostic_fail verify_kpack_baseline
KPACK_REBASE="$(python3 -B "$VERIFIER" "${verify_common[@]}" --role m8-kpack-rebase --target test_fully_quantized_grouped_kpack_discovery --expect-ppu-def PPU_M8_A_GMEM_TILE_REBASE=1)" || diagnostic_fail verify_kpack_rebase

HGOBJDUMP="$PPU_SDK_ROOT/bin/hgobjdump"
[ -x "$HGOBJDUMP" ] || diagnostic_fail sdk_lacks_hgobjdump
sdk_receipt=''
for receipt in "$PPU_SDK_ROOT/release.yaml" "$PPU_SDK_ROOT/VERSION.txt"; do
  [ ! -r "$receipt" ] || sdk_receipt="$sdk_receipt $(tr '\n' ' ' < "$receipt")"
done
case "$sdk_receipt" in *2.1.1-a5c56e*) ;; *) diagnostic_fail wrong_sdk_release ;; esac

RUNTIME_DIR=
for candidate in "$PPU_SDK_ROOT/lib" "$PPU_SDK_ROOT/lib64" \
    "$PPU_SDK_ROOT/CUDA_SDK/lib64" "$PPU_SDK_ROOT/targets/x86_64-linux/lib"; do
  if [ -f "$candidate/libhggc_wrapper.so" ]; then RUNTIME_DIR=$candidate; break; fi
done
[ -n "$RUNTIME_DIR" ] || diagnostic_fail sdk_lacks_hggc_wrapper
export LD_LIBRARY_PATH="$RUNTIME_DIR:$PPU_SDK_ROOT/lib:$PPU_SDK_ROOT/lib64:$PPU_SDK_ROOT/CUDA_SDK/lib64:$PPU_SDK_ROOT/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

inspect() {
  binary=$1; images=$2; functions=$3; label=$4
  log="$OUTPUT_DIR/$label.elf.log"
  "$HGOBJDUMP" --list-elf "$binary" > "$log" 2>&1 || return 1
  [ "$(grep -c '^ELF FILE [0-9].*(PPU 1\.0)' "$log" || true)" -eq "$images" ] || return 1
  [ "$(grep -c '^Func [0-9]' "$log" || true)" -eq "$functions" ] || return 1
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ldd "$binary" > "$OUTPUT_DIR/$label.ldd.log" 2>&1 || return 1
  ! grep -q 'not found' "$OUTPUT_DIR/$label.ldd.log"
}
inspect "$XPLANE_BASE" 1 10 xplane-baseline || diagnostic_fail inspect_xplane_baseline
inspect "$XPLANE_REBASE" 1 10 xplane-rebase || diagnostic_fail inspect_xplane_rebase
inspect "$KPACK_BASE" 1 5 kpack-baseline || diagnostic_fail inspect_kpack_baseline
inspect "$KPACK_REBASE" 1 5 kpack-rebase || diagnostic_fail inspect_kpack_rebase

run_one() {
  label=$1; shift
  log="$OUTPUT_DIR/$label.run.log"
  "$@" > "$log" 2>&1
  rc=$?
  printf 'M8_SECOND_TILE_REBASE_ARM arm=%s rc=%d\n' "$label" "$rc"
  grep -E '^(  G3 A-TAG|  G4 m8p? +M=(9|15|16|17)|== \[112:SECOND-TILE\]|FQ_GROUPED_KPACK_(SHARD|CELL|COMPLETE))' "$log" || true
  return "$rc"
}

run_one xplane-baseline "$XPLANE_BASE" --second-tile-only; xplane_base_rc=$?
run_one xplane-rebase "$XPLANE_REBASE" --second-tile-only; xplane_rebase_rc=$?
kpack_args=(
  --rows-file="$ROWS" --experts=256 --n=3072 --k=512
  --iterations=1 --warmups=1 --correctness-repeats=1
  --workload-key=grouped_control_tilem-boundary_n3072_k512_e256
  --router-profile=tilem-boundary
)
run_one kpack-baseline "$KPACK_BASE" "${kpack_args[@]}"; kpack_base_rc=$?
run_one kpack-rebase "$KPACK_REBASE" "${kpack_args[@]}"; kpack_rebase_rc=$?

xplane_base_a_red=0
grep -Eq '^  G3 A-TAG M=(9|15|16|17) raw-bitdiff=[1-9][0-9]*/' "$OUTPUT_DIR/xplane-baseline.run.log" && xplane_base_a_red=1
xplane_rebase_clean=0
[ "$xplane_rebase_rc" -eq 0 ] && grep -Fq '== [112:SECOND-TILE] PASS: errors=0 M=9/15/16/17 seams=mainloop-A+ptr-array-epilogue+nonpersistent+persistent ==' "$OUTPUT_DIR/xplane-rebase.run.log" && xplane_rebase_clean=1
kpack_base_red=0
[ "$kpack_base_rc" -ne 0 ] && grep -Eq 'state=RAW_FP16_MISMATCH raw_bad=[1-9]' "$OUTPUT_DIR/kpack-baseline.run.log" && kpack_base_red=1
kpack_rebase_clean=0
[ "$kpack_rebase_rc" -eq 0 ] && grep -Eq 'state=MEASURED raw_bad=0 ' "$OUTPUT_DIR/kpack-rebase.run.log" && grep -Eq '^FQ_GROUPED_KPACK_COMPLETE q=12 status=PASS ' "$OUTPUT_DIR/kpack-rebase.run.log" && kpack_rebase_clean=1

if [ "$xplane_base_a_red" -eq 1 ] && [ "$xplane_rebase_clean" -eq 1 ] && \
   [ "$kpack_base_red" -eq 1 ] && [ "$kpack_rebase_clean" -eq 1 ]; then
  verdict=ROOT_CAUSE_CONFIRMED_A_DESCRIPTOR_REBASE_CLOSES_XPLANE_AND_KPACK
elif [ "$xplane_rebase_clean" -ne 1 ] || [ "$kpack_rebase_clean" -ne 1 ]; then
  verdict=HYPOTHESIS_REJECTED_CANDIDATE_REMAINS_DIRTY
elif [ "$kpack_base_red" -eq 1 ] && [ "$kpack_rebase_clean" -eq 1 ]; then
  verdict=KPACK_CLOSED_XPLANE_BASELINE_DID_NOT_LOCALIZE_A
else
  verdict=BASELINE_DID_NOT_REPRODUCE
fi

printf 'M8_SECOND_TILE_REBASE_AB verdict=%s xplane=[base_rc:%d,a_red:%d,rebase_rc:%d,clean:%d] kpack=[base_rc:%d,red:%d,rebase_rc:%d,clean:%d] source=%s logs=%s\n' \
  "$verdict" "$xplane_base_rc" "$xplane_base_a_red" "$xplane_rebase_rc" "$xplane_rebase_clean" \
  "$kpack_base_rc" "$kpack_base_red" "$kpack_rebase_rc" "$kpack_rebase_clean" "$EXPECTED_SOURCE" "$OUTPUT_DIR"
exit 0
