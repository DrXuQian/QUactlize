#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:---local}"
OUT="${OUT:-/workspace/quactlize-l205-ppu-chunked-gdn}"
mkdir -p "$OUT"

python3 "$ROOT/dev/fold_derivation/check_l205_chunked_gdn_harness.py"
L203_BUILD_ROOT="$OUT/l203" bash "$ROOT/dev/fold_derivation/run_l203_chunked_gdn_oracle.sh"

# The host compile proves that the standalone harness sees only the public ABI
# and the HGGC runtime surface.  When a local NVIDIA device exists, the CUDA
# arm below additionally executes the unchanged scalar collective body; only
# the PPU AIU/shared-memory path remains a box postcondition.
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$ROOT/quactlize/include" \
  -isystem "$ROOT/dev/fold_derivation/stub_inc" \
  -c "$ROOT/tests/test_ppu_chunked_gdn_abi.cpp" \
  -o "$OUT/test_ppu_chunked_gdn_abi.host-contract.o"
echo "[L205 local] PASS: public-header/runtime compile contract; object=$OUT/test_ppu_chunked_gdn_abi.host-contract.o"

# RTX cannot donate the PPU kernel's 139776-byte shared ledger (sm_120 reports
# 101376 bytes/block).  The adapter executes the unchanged scalar collective
# body with that ledger in global scratch.  This is a correctness launch, not
# a performance proxy for the PPU shared-memory path.
if [[ "${QZ_GDN_SKIP_CUDA:-0}" == 1 ]]; then
  echo "[L205 CUDA] SKIP: QZ_GDN_SKIP_CUDA=1"
elif ! command -v nvcc >/dev/null 2>&1; then
  echo "[L205 CUDA] SKIP: nvcc is unavailable"
elif ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[L205 CUDA] SKIP: no NVIDIA device is visible"
else
  CUDA_ARCH="${QZ_GDN_CUDA_ARCH:-sm_120}"
  nvcc -std=c++17 -O3 -arch="$CUDA_ARCH" --expt-relaxed-constexpr \
    -DQZ_GDN_CUDA_RUNTIME=1 -DCUTLASS_USE_PACKED_TUPLE=1 \
    -I"$ROOT/quactlize/include" \
    -I"$ROOT/third_party/actlize/include" \
    -I"$ROOT/third_party/actlize/tools/util/include" \
    -I"$ROOT/dev/fold_derivation/stub_inc" \
    "$ROOT/tests/test_ppu_chunked_gdn_abi.cpp" \
    "$ROOT/dev/fold_derivation/l205_chunked_gdn_cuda_adapter.cu" \
    -o "$OUT/test_ppu_chunked_gdn_cuda_reference"
  "$OUT/test_ppu_chunked_gdn_cuda_reference" | tee "$OUT/cuda-correctness.log"
  echo "[L205 CUDA] PASS: exact scalar collective body launched with global test scratch"
fi

if [[ "$MODE" == "--local" ]]; then
  echo "[L205 device] SKIP: --local selected; PPU execution requires --box"
  exit 0
fi
if [[ "$MODE" != "--box" ]]; then
  echo "usage: $0 [--local|--box]" >&2
  exit 2
fi

PPU_SDK_ROOT="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-/usr/local/PPU_SDK}}}"
if [[ ! -x "$PPU_SDK_ROOT/bin/hgcc" ]]; then
  echo "[L205 device] FAIL: hgcc unavailable at $PPU_SDK_ROOT/bin/hgcc" >&2
  exit 1
fi

PPU_BUILD_DIR="$OUT/build" TARGET=quactlize_ppu JOBS="${JOBS:-16}" \
  PPU_SDK="$PPU_SDK_ROOT" bash "$ROOT/build.sh" | tee "$OUT/library-build.log"
LIB="$(find "$OUT/build" -type f -name libquactlize_ppu.so -print -quit)"
if [[ -z "$LIB" ]]; then
  echo "[L205 device] FAIL: shipping libquactlize_ppu.so was not produced" >&2
  exit 1
fi
LIBDIR="$(dirname "$LIB")"

g++ -std=c++17 -O3 -DSWITCH_TO_HGGCRT \
  -I"$ROOT/quactlize/include" \
  -I"$PPU_SDK_ROOT/include" \
  -I"$PPU_SDK_ROOT/targets/x86_64-linux/include" \
  "$ROOT/tests/test_ppu_chunked_gdn_abi.cpp" \
  -L"$LIBDIR" -lquactlize_ppu \
  -L"$PPU_SDK_ROOT/lib" -lhg_wrapper -lhggc_wrapper -lhggcrt1 -lhggc \
  -Wl,-rpath,"$LIBDIR" -Wl,-rpath,"$PPU_SDK_ROOT/lib" \
  -o "$OUT/test_ppu_chunked_gdn_abi"

sha256sum "$LIB" "$OUT/test_ppu_chunked_gdn_abi" | tee "$OUT/binary-sha256.txt"
"$OUT/test_ppu_chunked_gdn_abi" | tee "$OUT/device-correctness.log"
echo "[L205 device] PASS: artifacts=$OUT"
