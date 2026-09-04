#!/usr/bin/env bash
# Verify and execute the prebuilt ppu0010 m8n16 prerequisite and second-tile
# numerical probes.  This runner never configures or builds source code.
set -Eeuo pipefail

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  printf 'run-prebuilt.sh must be executed with bash, not sourced\n' >&2
  return 2
fi

EXPECTED_SOURCE=c2f9980d52c6b32c955ed75b9bebba302f619aec
EXPECTED_ACTLIZE=8d46b758c8931807df840a6ed87d272d74a8fdf4
EXPECTED_CUTLASS=f94ec46f4f63f96003d6cfdf2014731e7672c281
EXPECTED_MANIFEST_SHA256=cf4af68031b1829c1a52784ca2ea23c98deccab632d23e1a06439f9b324547da
EXPECTED_GATES_SHA256=465ddbac3b6d86af6b9eaa09e06bae1d8ea825d3ec086eeccd04494b3a789a34
EXPECTED_COLLECTIVE_SHA256=9906e92fa2403d598f537db7625d0b34a64bae1ae13ccc0bbc70d5c78afa9d85
EXPECTED_SDK='PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e sha256=63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd'
EXPECTED_COMPILER='hgcc Release version 2.1.1-a5c56e built=2026-07-25T09:15:42 sha256=fa62c590c67411c23fa4028f15fa562b39ce0cf830830d038a1ec04c59d8c76e'

usage() {
  printf 'usage: CUDA_VISIBLE_DEVICES=N bash %s --ppu-sdk PATH --output NEW_DIR\n' "$0" >&2
}

fail() {
  printf '[m8n16-prebuilt] FAIL: %s\n' "$*" >&2
  if [ -n "${OUTPUT_DIR:-}" ]; then
    printf '[m8n16-prebuilt] logs=%s\n' "$OUTPUT_DIR" >&2
  fi
  exit 1
}

PPU_SDK_ROOT=
OUTPUT_DIR=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --ppu-sdk)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      PPU_SDK_ROOT="$2"
      shift 2
      ;;
    --output)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

[ -n "$PPU_SDK_ROOT" ] && [ -n "$OUTPUT_DIR" ] || { usage; exit 2; }
case "${CUDA_VISIBLE_DEVICES:-}" in
  ''|*,*|*[!0-9]*) fail 'CUDA_VISIBLE_DEVICES must name exactly one numeric device ordinal' ;;
esac
[ ! -e "$OUTPUT_DIR" ] || fail "output already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_ROOT="$(git -C "$BUNDLE_DIR" rev-parse --show-toplevel 2>/dev/null)" \
  || fail 'bundle is not inside its artifact Git worktree'
MANIFEST="$BUNDLE_DIR/manifest.json"
VERIFIER="$SOURCE_ROOT/tools/verify_prebuilt_ppu_bundle.py"

artifact_parent="$(git -C "$SOURCE_ROOT" rev-parse HEAD^ 2>/dev/null)" \
  || fail 'artifact commit has no source parent'
[ "$artifact_parent" = "$EXPECTED_SOURCE" ] \
  || fail "artifact parent differs: $artifact_parent"
[ "$(git -C "$SOURCE_ROOT" rev-parse "$EXPECTED_SOURCE:third_party/actlize")" = "$EXPECTED_ACTLIZE" ] \
  || fail 'actlize gitlink differs'
[ "$(git -C "$SOURCE_ROOT" rev-parse "$EXPECTED_SOURCE:third_party/cutlass")" = "$EXPECTED_CUTLASS" ] \
  || fail 'cutlass gitlink differs'

actual_manifest_sha="$(sha256sum "$MANIFEST" | awk '{print $1}')" \
  || fail 'cannot hash manifest'
[ "$actual_manifest_sha" = "$EXPECTED_MANIFEST_SHA256" ] \
  || fail "manifest SHA-256 differs: $actual_manifest_sha"

common_verify=(
  "$MANIFEST"
  --qtype 12
  --expect-source "$EXPECTED_SOURCE"
  --expect-submodule "third_party/actlize=$EXPECTED_ACTLIZE"
  --expect-submodule "third_party/cutlass=$EXPECTED_CUTLASS"
  --expect-sdk "$EXPECTED_SDK"
  --expect-compiler "$EXPECTED_COMPILER"
  --expect-arch ppu0010
)
GATES_BIN="$(python3 -B "$VERIFIER" "${common_verify[@]}" \
  --role m8n16-prerequisite --target test_ppu_m8n16_gates)" \
  || fail 'prerequisite payload verification failed'
COLLECTIVE_BIN="$(python3 -B "$VERIFIER" "${common_verify[@]}" \
  --role m8n16-second-tile --target test_ppu_m8n16_collective)" \
  || fail 'second-tile payload verification failed'
[ "$(sha256sum "$GATES_BIN" | awk '{print $1}')" = "$EXPECTED_GATES_SHA256" ] \
  || fail 'prerequisite binary authority differs'
[ "$(sha256sum "$COLLECTIVE_BIN" | awk '{print $1}')" = "$EXPECTED_COLLECTIVE_SHA256" ] \
  || fail 'second-tile binary authority differs'

HGOBJDUMP="$PPU_SDK_ROOT/bin/hgobjdump"
[ -x "$HGOBJDUMP" ] || fail "PPU SDK lacks bin/hgobjdump: $PPU_SDK_ROOT"
sdk_identity=''
for receipt in "$PPU_SDK_ROOT/release.yaml" "$PPU_SDK_ROOT/VERSION.txt"; do
  if [ -r "$receipt" ]; then
    sdk_identity+="$(tr '\n' ' ' < "$receipt") "
  fi
done
case "$sdk_identity" in
  *2.1.1-a5c56e*) ;;
  *) fail 'PPU SDK is not release 2.1.1-a5c56e' ;;
esac

RUNTIME_DIR=
for candidate in \
  "$PPU_SDK_ROOT/lib" \
  "$PPU_SDK_ROOT/lib64" \
  "$PPU_SDK_ROOT/CUDA_SDK/lib64" \
  "$PPU_SDK_ROOT/targets/x86_64-linux/lib"; do
  if [ -f "$candidate/libhggc_wrapper.so" ]; then
    RUNTIME_DIR="$candidate"
    break
  fi
done
[ -n "$RUNTIME_DIR" ] || fail 'PPU SDK lacks libhggc_wrapper.so'
export LD_LIBRARY_PATH="$RUNTIME_DIR:$PPU_SDK_ROOT/lib:$PPU_SDK_ROOT/lib64:$PPU_SDK_ROOT/CUDA_SDK/lib64:$PPU_SDK_ROOT/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

inspect_ppu_image() {
  local binary="$1" expected_images="$2" expected_functions="$3" label="$4"
  local log="$OUTPUT_DIR/$label.elf.log"
  "$HGOBJDUMP" -lelf "$binary" > "$log" 2>&1 \
    || fail "$label live ELF inspection failed"
  grep -q 'file format elf64-x86-64' "$log" \
    || fail "$label host carrier identity is absent"
  local all_images ppu10_images functions
  all_images="$(grep -c '^ELF FILE [0-9]' "$log" || true)"
  ppu10_images="$(grep -c '^ELF FILE [0-9].*(PPU 1\.0)' "$log" || true)"
  functions="$(grep -c '^Func [0-9]' "$log" || true)"
  [ "$all_images" -eq "$expected_images" ] && [ "$ppu10_images" -eq "$expected_images" ] \
    || fail "$label PPU image census differs: all=$all_images ppu0010=$ppu10_images expected=$expected_images"
  [ "$functions" -eq "$expected_functions" ] \
    || fail "$label PPU function census differs: got=$functions expected=$expected_functions"
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ldd "$binary" > "$OUTPUT_DIR/$label.ldd.log" 2>&1 \
    || fail "$label runtime linkage is not loadable on this host"
  ! grep -q 'not found' "$OUTPUT_DIR/$label.ldd.log" \
    || fail "$label runtime linkage contains an unresolved library"
  grep -Fq "$RUNTIME_DIR/libhggc_wrapper.so" "$OUTPUT_DIR/$label.ldd.log" \
    || fail "$label did not bind the selected SDK runtime"
}

inspect_ppu_image "$GATES_BIN" 2 2 prerequisite
inspect_ppu_image "$COLLECTIVE_BIN" 1 10 second-tile
python3 -B "$SOURCE_ROOT/ci/check_m8n16_g2_contract.py" \
  > "$OUTPUT_DIR/g2-contract.log" 2>&1 \
  || fail 'G2 source contract failed'
python3 -B "$SOURCE_ROOT/ci/check_m8n16_second_tile_contract.py" \
  > "$OUTPUT_DIR/second-tile-contract.log" 2>&1 \
  || fail 'second-tile source contract failed'

require_once() {
  local needle="$1" log="$2" description="$3"
  local count
  count="$(grep -Fxc "$needle" "$log" || true)"
  [ "$count" -eq 1 ] || fail "$description marker count is $count, expected 1"
}

GATES_LOG="$OUTPUT_DIR/prerequisite.run.log"
if ! "$GATES_BIN" 2>&1 | tee "$GATES_LOG"; then
  fail 'positive G0/G1/G2 prerequisite returned nonzero'
fi
require_once '[G1] PASS: total_bad=0' "$GATES_LOG" 'G1'
require_once '[G2-control-path] same-payload=production-x4 cube=16x64 coords=(0,0) green=get_i/get_j red=historical-nvidia-x2-provider-map' "$GATES_LOG" 'G2 control'
require_once '[G2-green-detail] x4_values=512 x4_bad=0 projected_changed=0/128 lower_poison_changed=128/128' "$GATES_LOG" 'G2 green detail'
require_once '[G2-green] mismatches=0 PASS' "$GATES_LOG" 'G2 green'
require_once '[G2-negative-detail] same_payload=x4-swzl geometry=16x64 bad_map_values=128 bad_map_bad=0 coincident_words=2/64 red_expected=124/128' "$GATES_LOG" 'G2 negative detail'
require_once '[G2-negative] mismatches=124 EXPECTED_RED/PASS' "$GATES_LOG" 'G2 negative'
GATES_FINAL='== [111] PASS: G1=0 G2=0 =='
require_once "$GATES_FINAL" "$GATES_LOG" 'G1/G2 aggregate'
[ "$(awk 'NF {line=$0} END {print line}' "$GATES_LOG")" = "$GATES_FINAL" ] \
  || fail 'G1/G2 aggregate marker is not the final nonempty line'
! grep -Eq 'MISMATCH|UNEXPECTED_GREEN|== \[[^]]+\] FAIL:' "$GATES_LOG" \
  || fail 'G1/G2 log contains a failure marker'

COLLECTIVE_LOG="$OUTPUT_DIR/second-tile.run.log"
if ! "$COLLECTIVE_BIN" --second-tile-only 2>&1 | tee "$COLLECTIVE_LOG"; then
  fail 'G3/G4 second-tile numerical probe returned nonzero'
fi
require_once '[offline] m8/m16 B artifacts byte-identical: 4096 physical bytes (4096 logical); roundtrip=0/8192' "$COLLECTIVE_LOG" 'offline identity'
grep -Eq '^  G3 raw FP32 accum +bad=0/256 max_abs=[^ ]+ MATCH$' "$COLLECTIVE_LOG" \
  || fail 'G3 raw FP32 accumulator marker is absent'

for m in 1 2 3 7 8 9 15 16 17; do
  count=$((m * 32))
  grep -Eq "^  G4 m8 +M=${m} golden bad=0/${count} max_abs=[^ ]+ MATCH$" "$COLLECTIVE_LOG" \
    || fail "G4 m8 M=$m golden marker is absent"
  grep -Eq "^  G4 m16 +M=${m} golden bad=0/${count} max_abs=[^ ]+ MATCH$" "$COLLECTIVE_LOG" \
    || fail "G4 m16 M=$m golden marker is absent"
  require_once "  G4 m8-vs-m16 M=${m} bitdiff=0/${count} MATCH" "$COLLECTIVE_LOG" "G4 m8/m16 M=$m"
done

for m in 9 15 16 17; do
  count=$((m * 32))
  red=$(((m - 8) * 32))
  require_once "  G3 A-TAG M=${m} raw-bitdiff=0/${count} MATCH" "$COLLECTIVE_LOG" "G3 A-tag M=$m"
  require_once "  G3 A-TAG-NEGATIVE M=${m} replay-oracle-bitdiff=0/${count} observed-red=${red} expected-red=${red} EXPECTED_RED" "$COLLECTIVE_LOG" "G3 A-tag negative M=$m"
  require_once "  G4 EPILOGUE-TAG M=${m} raw-bitdiff=0/${count} MATCH" "$COLLECTIVE_LOG" "G4 epilogue tag M=$m"
  row16=-1
  if [ "$m" -gt 16 ]; then row16=32; fi
  require_once "  G4 EPILOGUE-TAG-NEGATIVE M=${m} observed-red=${red} expected-red=${red} row8-red=32 row16-red=${row16} EXPECTED_RED" "$COLLECTIVE_LOG" "G4 epilogue negative M=$m"
  grep -Eq "^  G4 m8p +M=${m} golden bad=0/${count} max_abs=[^ ]+ MATCH$" "$COLLECTIVE_LOG" \
    || fail "G4 persistent m8 M=$m golden marker is absent"
  require_once "  G4 m8p-vs-m8 M=${m} bitdiff=0/${count} MATCH" "$COLLECTIVE_LOG" "G4 persistent/nonpersistent M=$m"
done

COLLECTIVE_FINAL='== [112:SECOND-TILE] PASS: errors=0 M=9/15/16/17 seams=mainloop-A+ptr-array-epilogue+nonpersistent+persistent =='
require_once "$COLLECTIVE_FINAL" "$COLLECTIVE_LOG" 'second-tile aggregate'
[ "$(awk 'NF {line=$0} END {print line}' "$COLLECTIVE_LOG")" = "$COLLECTIVE_FINAL" ] \
  || fail 'second-tile aggregate marker is not the final nonempty line'
! grep -Eq 'MISMATCH|UNEXPECTED_GREEN|== \[[^]]+\] FAIL:' "$COLLECTIVE_LOG" \
  || fail 'second-tile log contains a failure marker'

printf 'M8N16_SECOND_TILE_PREBUILT_DEVICE PASS source=%s manifest=%s logs=%s\n' \
  "$EXPECTED_SOURCE" "$EXPECTED_MANIFEST_SHA256" "$OUTPUT_DIR"
