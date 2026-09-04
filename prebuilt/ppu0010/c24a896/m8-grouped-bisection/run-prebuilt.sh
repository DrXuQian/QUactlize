#!/usr/bin/env bash
# Compile-free raw-bit bisection for grouped Q4 K-pack TM8 failures.
# Every terminal path returns zero so a diagnostic cannot terminate the caller's container.
set -u -o pipefail

EXPECTED_SOURCE=c24a896b3a5fd184a7f0d647a7b00e1e7812c670
EXPECTED_ACTLIZE=8d46b758c8931807df840a6ed87d272d74a8fdf4
EXPECTED_CUTLASS=f94ec46f4f63f96003d6cfdf2014731e7672c281
EXPECTED_MANIFEST=5360b78426a3b290f2aabda5ac01374added92e653958233d55348d39115ac69
EXPECTED_TM8_GENERATED=53546d4ab7659cb36a7df508dc33b289d38453d81fb996eec3c8ce0cf8c82bbc
EXPECTED_TM16_GENERATED=5720ba2ef7f5b96483c19eca8e45f063cfdd9b20680b9c2618a756d13c11cd67
EXPECTED_SDK='PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e sha256=63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd'
EXPECTED_COMPILER='hgcc Release version 2.1.1-a5c56e built=2026-07-25T09:15:42 sha256=fa62c590c67411c23fa4028f15fa562b39ce0cf830830d038a1ec04c59d8c76e'

usage() {
  printf 'usage: CUDA_VISIBLE_DEVICES=N bash %s --ppu-sdk PATH --output DIR\n' "$0"
}

diagnostic_fail() {
  printf 'M8_GROUPED_BISECTION verdict=INFRASTRUCTURE_FAIL reason=%s logs=%s\n' \
    "$1" "${RUN_DIR:-${OUTPUT_BASE:-NONE}}"
  exit 0
}

PPU_SDK_ROOT=
OUTPUT_BASE=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --ppu-sdk) [ "$#" -ge 2 ] || { usage; exit 0; }; PPU_SDK_ROOT=$2; shift 2 ;;
    --output) [ "$#" -ge 2 ] || { usage; exit 0; }; OUTPUT_BASE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; diagnostic_fail "unknown_argument:$1" ;;
  esac
done

[ -n "$PPU_SDK_ROOT" ] || diagnostic_fail missing_ppu_sdk
[ -n "$OUTPUT_BASE" ] || diagnostic_fail missing_output
case "${CUDA_VISIBLE_DEVICES:-}" in
  ''|*,*|*[!0-9]*) diagnostic_fail CUDA_VISIBLE_DEVICES_must_be_one_numeric_ordinal ;;
esac

stamp=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$OUTPUT_BASE/run-$stamp-$$"
mkdir -p "$RUN_DIR" || diagnostic_fail cannot_create_output

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || diagnostic_fail cannot_resolve_bundle
SOURCE_ROOT="$(git -C "$BUNDLE_DIR" rev-parse --show-toplevel 2>/dev/null)" || diagnostic_fail bundle_not_in_git
MANIFEST="$BUNDLE_DIR/manifest.json"
TM8_GENERATED="$BUNDLE_DIR/inputs/tm8-generated-manifest.json"
TM16_GENERATED="$BUNDLE_DIR/inputs/tm16-generated-manifest.json"
VERIFIER="$SOURCE_ROOT/tools/verify_prebuilt_ppu_bundle.py"

[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD^ 2>/dev/null)" = "$EXPECTED_SOURCE" ] || diagnostic_fail wrong_artifact_parent
[ "$(git -C "$SOURCE_ROOT" rev-parse "$EXPECTED_SOURCE:third_party/actlize" 2>/dev/null)" = "$EXPECTED_ACTLIZE" ] || diagnostic_fail wrong_actlize
[ "$(git -C "$SOURCE_ROOT" rev-parse "$EXPECTED_SOURCE:third_party/cutlass" 2>/dev/null)" = "$EXPECTED_CUTLASS" ] || diagnostic_fail wrong_cutlass
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$EXPECTED_MANIFEST" ] || diagnostic_fail wrong_manifest_hash
[ "$(sha256sum "$TM8_GENERATED" | awk '{print $1}')" = "$EXPECTED_TM8_GENERATED" ] || diagnostic_fail wrong_tm8_generated_manifest
[ "$(sha256sum "$TM16_GENERATED" | awk '{print $1}')" = "$EXPECTED_TM16_GENERATED" ] || diagnostic_fail wrong_tm16_generated_manifest

verify_common=(
  "$MANIFEST" --qtype 12 --expect-source "$EXPECTED_SOURCE"
  --expect-submodule "third_party/actlize=$EXPECTED_ACTLIZE"
  --expect-submodule "third_party/cutlass=$EXPECTED_CUTLASS"
  --expect-sdk "$EXPECTED_SDK" --expect-compiler "$EXPECTED_COMPILER"
  --expect-arch ppu0010
)
TM8_BIN="$(python3 -B "$VERIFIER" "${verify_common[@]}" \
  --role grouped-q4-kpack-tm8-np-p \
  --target test_fully_quantized_grouped_kpack_discovery)" || diagnostic_fail verify_tm8
TM16_BIN="$(python3 -B "$VERIFIER" "${verify_common[@]}" \
  --role grouped-q4-kpack-tm16-np-p \
  --target test_fully_quantized_grouped_kpack_discovery)" || diagnostic_fail verify_tm16

HGOBJDUMP="$PPU_SDK_ROOT/bin/hgobjdump"
[ -x "$HGOBJDUMP" ] || diagnostic_fail sdk_lacks_hgobjdump
sdk_receipt=
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
  binary=$1; label=$2
  "$HGOBJDUMP" --list-elf "$binary" > "$RUN_DIR/$label.elf.log" 2>&1 || return 1
  [ "$(grep -c '^ELF FILE [0-9].*(PPU 1\.0)' "$RUN_DIR/$label.elf.log" || true)" -eq 1 ] || return 1
  [ "$(grep -c '^Func [0-9]' "$RUN_DIR/$label.elf.log" || true)" -eq 6 ] || return 1
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ldd "$binary" > "$RUN_DIR/$label.ldd.log" 2>&1 || return 1
  ! grep -q 'not found' "$RUN_DIR/$label.ldd.log"
}
inspect "$TM8_BIN" tm8 || diagnostic_fail inspect_tm8
inspect "$TM16_BIN" tm16 || diagnostic_fail inspect_tm16

make_single_expert_rows() {
  rows=$1; path=$2
  printf '%d\n0\n' "$rows" > "$path"
}

make_boundary_rows() {
  path=$1
  i=0
  while [ "$i" -lt 256 ]; do
    case "$i" in
      0) value=15 ;; 1) value=16 ;; 2) value=17 ;;
      3) value=31 ;; 4) value=32 ;; 5) value=33 ;;
      6) value=127 ;; 7) value=128 ;; 8) value=129 ;;
      *) value=0 ;;
    esac
    printf '%d\n' "$value"
    i=$((i + 1))
  done > "$path"
}

run_one() {
  family=$1; binary=$2; profile=$3; experts=$4; rows_file=$5; n=$6
  label="$family-$profile-n$n"
  log="$RUN_DIR/$label.run.log"
  "$binary" --rows-file="$rows_file" --experts="$experts" --n="$n" --k=512 \
    --iterations=1 --warmups=1 --correctness-repeats=1 \
    --workload-key="$label" --router-profile="$profile" > "$log" 2>&1
  rc=$?
  printf 'M8_GROUPED_BISECTION_ARM family=%s profile=%s experts=%s n=%s rc=%d\n' \
    "$family" "$profile" "$experts" "$n" "$rc"
  grep -E '^FQ_GROUPED_KPACK_(SHARD|CELL|MISMATCH_MAP|COMPLETE)' "$log" || true
}

for rows in 7 8 9; do
  rows_file="$RUN_DIR/e0-$rows.txt"
  make_single_expert_rows "$rows" "$rows_file"
  for n in 64 3072; do
    run_one tm8 "$TM8_BIN" "e0-$rows" 2 "$rows_file" "$n"
    run_one tm16 "$TM16_BIN" "e0-$rows" 2 "$rows_file" "$n"
  done
done

boundary="$RUN_DIR/tilem-boundary.txt"
make_boundary_rows "$boundary"
run_one tm8 "$TM8_BIN" tilem-boundary 256 "$boundary" 3072
run_one tm16 "$TM16_BIN" tilem-boundary 256 "$boundary" 3072

arms=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*.run.log' | wc -l)
maps=$(grep -hEc '^FQ_GROUPED_KPACK_MISMATCH_MAP' "$RUN_DIR"/*.run.log 2>/dev/null | awk '{s += $1} END {print s + 0}')
printf 'M8_GROUPED_BISECTION verdict=DIAGNOSTIC_COMPLETE arms=%s mismatch_maps=%s source=%s logs=%s\n' \
  "$arms" "$maps" "$EXPECTED_SOURCE" "$RUN_DIR"
exit 0
