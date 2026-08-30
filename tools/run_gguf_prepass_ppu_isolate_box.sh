#!/usr/bin/env bash
# One-TU HGCC carrier isolate for the production raw and packed GGUF metadata prepass kernels.
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

jobs="${JOBS:-16}"
sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
sha="$(git rev-parse HEAD)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="${OUT:-/workspace/quactlize-gguf-prepass-ppu-isolate-${sha:0:8}-${stamp}}"

fail() {
  printf '[gguf-prepass-isolate] FAIL: %s\n' "$*" >&2
  return 2
}

python3 - <<'PY'
from pathlib import Path

source = Path("tests/test_gguf_prepass_ppu_isolate.cu").read_text()
cmake = Path("quactlize/csrc/CMakeLists.txt.in").read_text()
required = (
    '#include "gguf_scale_prepass.hpp"',
    "prepass_kernel<KType::Q2_K, 0><<<raw_grid, 256>>>(raw_args)",
    "prepass_unit_kernel<KType::Q2_K, 0><<<packed_grid, 256>>>(packed_args)",
    "prepass_host<KType::Q2_K, 0>",
    "prepass_unit_host<KType::Q2_K, 0>",
    "raw() ^ 1u",
    "FQ_PREPASS_PPU_ISOLATE_VERDICT",
)
for marker in required:
    assert marker in source, f"missing isolate seam: {marker}"
assert "__global__ void carrier_marker" in source
assert "__global__ void prepass_kernel" not in source
assert "__global__ void prepass_unit_kernel" not in source
assert "test_gguf_prepass_ppu_isolate" in cmake
assert source.replace("raw() ^ 1u", "raw()") != source, "value-negative plant target is absent"
assert source.replace("packed_bad == 0", "packed_bad != 0") != source, "verdict-negative plant target is absent"
print("[gguf-prepass-isolate:self-test] PASS production header bodies, nonzero oracle, poison and two negatives")
PY

[ -x "$sdk_root/bin/hgcc" ] || fail "real PPU hgcc is absent; set PPU_SDK"
[ -x "$sdk_root/bin/hgobjdump" ] || fail "real PPU hgobjdump is absent; set PPU_SDK"
[ ! -e "$out" ] || fail "OUT already exists: $out"
mkdir -p "$out/results"
trap 'rc=$?; printf "[gguf-prepass-isolate] DONE rc=%d artifacts=%s\n" "$rc" "$out"' EXIT

build_log="$out/results/build.log"
env -i \
  HOME="$HOME" USER="${USER:-root}" PATH="$PATH" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LANG="${LANG:-C.UTF-8}" \
  PPU_SDK="$sdk_root" PPU_ARCHS=ppu0010 \
  PPU_BUILD_DIR="$out/build" TARGET=test_gguf_prepass_ppu_isolate JOBS="$jobs" \
  CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
  "$root/build.sh" >"$build_log" 2>&1 || {
    grep -n -B5 -A10 -E \
      'error:|fatal error:|undefined reference|ld\.lld:|LLVM ERROR|Killed|timed out|Segmentation|PLEASE submit' \
      "$build_log" | head -160 >&2 || true
    tail -50 "$build_log" >&2
    fail "standalone carrier build failed"
  }

grep -qF '[build.sh] CUTLASS_PPU_ARCHS=ppu0010' "$build_log" || fail "build did not bind ppu0010"
build_make="$(find "$out/build" -path '*test_gguf_prepass_ppu_isolate.dir/build.make' -print -quit)"
[ -n "$build_make" ] || fail "target build.make is absent"
grep -qF "$sdk_root/bin/hgcc" "$build_make" || fail "target was not assigned to real hgcc"
grep -q -- '-arch=ppu_10' "$build_make" || fail "target lacks -arch=ppu_10"
grep -qF 'test_gguf_prepass_ppu_isolate.cu' "$build_make" || fail "target lost its one source"
! grep -qF 'ppu_backend.cu' "$build_make" || fail "carrier accidentally inherited the monolithic backend"

binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
[ -x "$binary" ] || fail "build reported no executable"
"$sdk_root/bin/hgobjdump" -lelf "$binary" >"$out/results/elf.txt" 2>"$out/results/elf.err" || \
  fail "carrier is not parseable by PPU hgobjdump"
for symbol in carrier_marker prepass_kernel prepass_unit_kernel; do
  grep -q "$symbol" "$out/results/elf.txt" || fail "carrier ELF lacks $symbol"
done

device_log="$out/results/device.log"
if "$binary" >"$device_log" 2>&1; then
  device_rc=0
else
  device_rc=$?
fi
cat "$device_log"

python3 - "$device_log" "$device_rc" <<'PY' | tee "$out/results/verdict.log"
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(errors="replace")
device_rc = int(sys.argv[2])
rows = re.findall(r"^FQ_PREPASS_PPU_ISOLATE (.+)$", text, re.M)
if len(rows) != 1:
    raise SystemExit(f"expected one isolate row, got {len(rows)}")
fields = dict(re.findall(r"([a-z_]+)=([0-9]+)", rows[0]))
needed = ("marker_bad", "raw_bad", "raw_sentinel", "raw_red_bad", "raw_gold_nonzero",
          "packed_bad", "packed_sentinel", "packed_red_bad", "packed_gold_nonzero")
missing = [key for key in needed if key not in fields]
if missing:
    raise SystemExit(f"isolate row lacks fields: {missing}")
v = {key: int(fields[key]) for key in needed}
if v["marker_bad"]:
    verdict = "STANDALONE_CARRIER_IMAGE_INVALID"
elif v["raw_bad"] or v["raw_sentinel"] or v["packed_bad"] or v["packed_sentinel"]:
    verdict = "PREPASS_KERNEL_CODEGEN_OR_ABI_FAILS_STANDALONE"
elif not v["raw_red_bad"] or not v["packed_red_bad"] or not v["raw_gold_nonzero"] or not v["packed_gold_nonzero"]:
    verdict = "ORACLE_DENOMINATOR_INVALID"
elif device_rc != 0:
    verdict = "DEVICE_EXIT_CONTRADICTS_EXACT_CELLS"
else:
    verdict = "PRODUCTION_PREPASS_BODIES_VALID_STANDALONE"
print(
    "FQ_PREPASS_PPU_ISOLATE_ROOT "
    f"verdict={verdict} device_rc={device_rc} one_tu=1 dlopen=0 pytorch=0 copied_kernel_body=0")
PY

printf '[gguf-prepass-isolate] DIAGNOSTIC_COMPLETE sha=%s artifacts=%s\n' "$sha" "$out"
