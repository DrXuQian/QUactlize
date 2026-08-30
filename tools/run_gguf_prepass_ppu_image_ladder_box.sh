#!/usr/bin/env bash
# Identify the first source addition that turns a minimal ppu0010 image into hggcErrorInvalidKernelImage.
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

jobs="${JOBS:-16}"
sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
sha="$(git rev-parse HEAD)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="${OUT:-/workspace/quactlize-gguf-prepass-image-${sha:0:8}-${stamp}}"
resume="${RESUME:-0}"

case "$resume" in
  0|1) ;;
  *) printf '[gguf-prepass-image] FAIL: RESUME must be 0 or 1, got %q\n' "$resume" >&2; exit 2 ;;
esac

fail() {
  printf '[gguf-prepass-image] FAIL: %s\n' "$*" >&2
  return 2
}

python3 - <<'PY'
from pathlib import Path

body = Path("tests/gguf_prepass_ppu_image_arm.inc").read_text()
cmake = Path("quactlize/csrc/CMakeLists.txt.in").read_text()
runner = Path("tools/run_gguf_prepass_ppu_image_ladder_box.sh").read_text()
for arm, value in (("marker", 0), ("header", 1), ("raw", 2), ("packed", 3)):
    wrapper = Path(f"tests/test_gguf_prepass_ppu_image_{arm}.cu").read_text()
    assert wrapper == (f'#define PPU_PREPASS_IMAGE_ARM {value}\n'
                       '#include "gguf_prepass_ppu_image_arm.inc"\n')
    assert f"test_gguf_prepass_ppu_image_${{_arm}}" in cmake
required = (
    "__global__ void prepass_image_marker",
    "prepass_kernel<gguf_scale::KType::Q2_K, 0><<<1, 256>>>(args)",
    "prepass_unit_kernel<gguf_scale::KType::Q2_K, 0><<<1, 256>>>(args)",
    "args.num_cols",  # zero-column raw guard is inside the exact production body
    "FQ_PREPASS_PPU_IMAGE_VERDICT",
)
production = Path("quactlize/include/gguf_scale_prepass.hpp").read_text()
for marker in required[:3] + required[4:]:
    assert marker in body, f"missing image rung: {marker}"
assert "if (n >= args.num_cols) return;" in production
assert "if (e >= args.num_experts) return;" in production
assert body.replace("PPU_PREPASS_IMAGE_ARM == 2", "PPU_PREPASS_IMAGE_ARM == 9") != body
assert body.replace("marker_launch.immediate == 0", "marker_launch.immediate != 0") != body
bad_copy = 'cp "$configure_' + 'log" "$build_log"'
assert bad_copy not in runner, "marker log must not be copied onto itself"
assert 'RESUME=1 reusing' in runner
print("[gguf-prepass-image:self-test] PASS four distinct objects, exact raw/packed symbols, inert guards, two negatives and resumable marker build")
PY

[ -x "$sdk_root/bin/hgcc" ] || fail "real PPU hgcc is absent; set PPU_SDK"
[ -x "$sdk_root/bin/hgobjdump" ] || fail "real PPU hgobjdump is absent; set PPU_SDK"
if [ "$resume" = 0 ]; then
  [ ! -e "$out" ] || fail "OUT already exists: $out"
else
  [ -f "$out/build/CMakeCache.txt" ] || fail "RESUME=1 needs the configured build in $out/build"
  [ -s "$out/results/build-marker.log" ] || fail "RESUME=1 needs the completed marker build log"
  authority_file="$out/build/.quactlize-source-head"
  [ -s "$authority_file" ] || fail "RESUME=1 artifact has no source authority"
  read -r authority <"$authority_file"
  git cat-file -e "${authority}^{commit}" 2>/dev/null || fail "RESUME=1 source authority is not in this checkout"
  # The resume commit may differ only by this runner fix. hgcc custom rules have no header depfiles, so explicitly
  # prove every CMake/device input to the four arms is byte-identical before reusing even one object.
  git diff --quiet "$authority" HEAD -- \
    quactlize/csrc/CMakeLists.txt.in \
    tests/gguf_prepass_ppu_image_arm.inc \
    tests/test_gguf_prepass_ppu_image_marker.cu \
    tests/test_gguf_prepass_ppu_image_header.cu \
    tests/test_gguf_prepass_ppu_image_raw.cu \
    tests/test_gguf_prepass_ppu_image_packed.cu || \
    fail "RESUME=1 refused changed CMake/device inputs since $authority"
  printf '[gguf-prepass-image] RESUME=1 reusing marker build authority=%s artifacts=%s\n' "$authority" "$out"
fi
mkdir -p "$out/results"
trap 'rc=$?; printf "[gguf-prepass-image] DONE rc=%d artifacts=%s\n" "$rc" "$out"' EXIT

build="$out/build"
configure_log="$out/results/build-marker.log"
if [ "$resume" = 0 ]; then
  env -i \
    HOME="$HOME" USER="${USER:-root}" PATH="$PATH" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LANG="${LANG:-C.UTF-8}" \
    PPU_SDK="$sdk_root" PPU_ARCHS=ppu0010 \
    PPU_BUILD_DIR="$build" TARGET=test_gguf_prepass_ppu_image_marker JOBS="$jobs" \
    CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
    "$root/build.sh" >"$configure_log" 2>&1 || {
      grep -n -B5 -A10 -E \
        'error:|fatal error:|undefined reference|ld\.lld:|LLVM ERROR|Killed|timed out|Segmentation|PLEASE submit' \
        "$configure_log" | head -120 >&2 || true
      tail -40 "$configure_log" >&2
      fail "marker build failed"
    }
fi
grep -qF '[build.sh] CUTLASS_PPU_ARCHS=ppu0010' "$configure_log" || fail "configured tree lost ppu0010"

build_and_run() {
  local arm="$1" target="test_gguf_prepass_ppu_image_${1}"
  local build_log="$out/results/build-${arm}.log"
  if [ "$arm" = marker ]; then
    : # build-marker.log is configure_log itself; do not copy a path onto itself.
  elif ! cmake --build "$build" --target "$target" -- -j"$jobs" >"$build_log" 2>&1; then
      grep -n -B5 -A10 -E \
        'error:|fatal error:|undefined reference|ld\.lld:|LLVM ERROR|Killed|timed out|Segmentation|PLEASE submit' \
        "$build_log" | head -120 >&2 || true
      tail -40 "$build_log" >&2
      fail "$arm build failed"
  fi

  local build_make object binary
  build_make="$(find "$build" -path "*${target}.dir/build.make" -print -quit)"
  [ -n "$build_make" ] || fail "$arm build.make is absent"
  grep -qF "$sdk_root/bin/hgcc" "$build_make" || fail "$arm was not compiled by the real hgcc"
  grep -q -- '-arch=ppu_10' "$build_make" || fail "$arm lacks -arch=ppu_10"
  grep -qF "test_gguf_prepass_ppu_image_${arm}.cu" "$build_make" || fail "$arm source identity is absent"
  object="$(find "$build" -type f -name "test_gguf_prepass_ppu_image_${arm}_*.o" -print -quit)"
  [ -s "$object" ] || fail "$arm hgcc object is absent"
  printf '%s\t%s\t%s\n' "$arm" "$object" "$(sha256sum "$object" | awk '{print $1}')" \
    >>"$out/results/objects.tsv"

  binary="$(find "$build" -type f -name "$target" -perm -u+x -print -quit)"
  [ -x "$binary" ] || fail "$arm executable is absent"
  "$sdk_root/bin/hgobjdump" -lelf "$binary" >"$out/results/elf-${arm}.txt" \
    2>"$out/results/elf-${arm}.err" || fail "$arm is not parseable by hgobjdump"
  grep -q 'prepass_image_marker' "$out/results/elf-${arm}.txt" || fail "$arm lacks its marker symbol"
  case "$arm" in
    marker|header)
      ! grep -q 'prepass_kernelILNS_5KTypeE0ELi0' "$out/results/elf-${arm}.txt" || \
        fail "$arm unexpectedly instantiated raw prepass"
      ! grep -q 'prepass_unit_kernelILNS_5KTypeE0ELi0' "$out/results/elf-${arm}.txt" || \
        fail "$arm unexpectedly instantiated packed prepass"
      ;;
    raw)
      grep -q 'prepass_kernelILNS_5KTypeE0ELi0' "$out/results/elf-${arm}.txt" || \
        fail "raw specialization is absent"
      ;;
    packed)
      grep -q 'prepass_unit_kernelILNS_5KTypeE0ELi0' "$out/results/elf-${arm}.txt" || \
        fail "packed specialization is absent"
      ;;
  esac

  local log="$out/results/device-${arm}.log"
  if "$binary" >"$log" 2>&1; then :; else :; fi
  cat "$log"
}

printf 'arm\tobject\tsha256\n' >"$out/results/objects.tsv"
for arm in marker header raw packed; do
  build_and_run "$arm"
done

python3 - "$out/results" <<'PY' | tee "$out/results/verdict.log"
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
arms = ("marker", "header", "raw", "packed")
rows = {}
for arm in arms:
    text = (root / f"device-{arm}.log").read_text(errors="replace")
    match = re.search(
        rf"^FQ_PREPASS_PPU_IMAGE arm={arm} marker_bad=(\d+) marker_value=(0x[0-9a-f]+) "
        r"marker=\[before:(\d+),immediate:(\d+),sync:(\d+)\] "
        r"subject=\[before:(\d+),immediate:(\d+),sync:(\d+)\]$", text, re.M)
    if not match:
        raise SystemExit(f"{arm}: expected exactly one parseable device row")
    rows[arm] = tuple(int(x, 0) for x in match.groups())
    verdicts = re.findall(rf"^FQ_PREPASS_PPU_IMAGE_VERDICT arm={arm} verdict=(PASS|FAIL)$", text, re.M)
    if len(verdicts) != 1:
        raise SystemExit(f"{arm}: expected one verdict, got {len(verdicts)}")

def clean(arm):
    marker_bad, marker_value, mb, mi, ms, sb, si, ss = rows[arm]
    return marker_bad == 0 and marker_value == 0x3c00 and (mb, mi, ms, sb, si, ss) == (0, 0, 0, 0, 0, 0)

states = {arm: int(clean(arm)) for arm in arms}
if not states["marker"]:
    verdict = "STANDALONE_BUILD_OR_RUNTIME_INVALID"
elif not states["header"]:
    verdict = "HEADER_INCLUSION_POISONS_IMAGE"
elif not states["raw"] and not states["packed"]:
    verdict = "RAW_AND_PACKED_SPECIALIZATIONS_POISON_IMAGES"
elif not states["raw"]:
    verdict = "RAW_SPECIALIZATION_POISONS_IMAGE"
elif not states["packed"]:
    verdict = "PACKED_SPECIALIZATION_POISONS_IMAGE"
else:
    verdict = "SPECIALIZATIONS_ADMIT_IN_SEPARATE_IMAGES"
print("FQ_PREPASS_PPU_IMAGE_ROOT " +
      " ".join(f"{arm}_clean={states[arm]}" for arm in arms) + f" verdict={verdict}")
PY

# A marker-only failure is not assigned to our source until an already-established PPU executable built in the same
# environment also fails or passes. Run this control only on that branch; successful marker arms need no extra build.
if ! grep -q 'marker_clean=1' "$out/results/verdict.log"; then
  control_log="$out/results/build-known-control.log"
  cmake --build "$build" --target test_ppu_f16x2_probe -- -j"$jobs" >"$control_log" 2>&1 || \
    fail "known PPU control build failed"
  control_binary="$(find "$build" -type f -name test_ppu_f16x2_probe -perm -u+x -print -quit)"
  [ -x "$control_binary" ] || fail "known PPU control executable is absent"
  if "$control_binary" >"$out/results/device-known-control.log" 2>&1; then control_rc=0; else control_rc=$?; fi
  grep -E '^== probe (PASS|FAIL)' "$out/results/device-known-control.log" || true
  printf 'FQ_PREPASS_PPU_IMAGE_CONTROL target=test_ppu_f16x2_probe rc=%d\n' "$control_rc" \
    | tee -a "$out/results/verdict.log"
fi

printf '[gguf-prepass-image] DIAGNOSTIC_COMPLETE sha=%s artifacts=%s\n' "$sha" "$out"
