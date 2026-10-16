#!/usr/bin/env bash
# Uses an existing llama .aoneci build; no alternate llama build entrypoint.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
LLAMA_CI_DIR=${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp}
LLAMA_BUILD_DIR=${LLAMA_BUILD_DIR:-${LLAMA_CI_DIR}/build-ci}
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
for path in "$LLAMA_CI_DIR/ggml/include/ggml.h" "$LLAMA_BUILD_DIR/CMakeCache.txt" \
            "$LLAMA_BUILD_DIR/bin/libggml-cuda.so" "$PPU_SDK/lib/libhggc_wrapper.so"; do
    if [ ! -f "$path" ]; then
        echo "LLAMA_AB_PRECHECK missing=$path" >&2
        echo 'Set LLAMA_CI_DIR and LLAMA_BUILD_DIR to the matching, already-built full K-pack caller.' >&2
        exit 1
    fi
done
args=(--llama-source "$LLAMA_CI_DIR" --llama-build "$LLAMA_BUILD_DIR" --sdk "$PPU_SDK"
      --devices "${DEVICES:-1,2,3,4,5,6,7}" --l2-bytes "${L2_BYTES:-0}"
      --phase "${PHASE:-all}" --hours "${HOURS:-24}" --idle-confirmed)
if [ -n "${QUACTLIZE_KPACK_EXECUTION:-}" ]; then args+=(--bundle "$QUACTLIZE_KPACK_EXECUTION"); fi
if [ -n "${QUACTLIZE_PPU_BUNDLE:-}" ]; then args+=(--legacy-bundle "$QUACTLIZE_PPU_BUNDLE"); fi
if [ -n "${QUACTLIZE_PPU_PACK_LIBRARY:-}" ]; then args+=(--pack-library "$QUACTLIZE_PPU_PACK_LIBRARY"); fi
if [ -n "${AB_RUN:-}" ]; then args+=(--output "$AB_RUN"); fi
if [ "${AB_RESUME:-0}" = 1 ]; then
    : "${AB_RUN:?AB_RESUME=1 requires AB_RUN}"
    args+=(--resume)
fi
cd -- "$ROOT"
exec python3 -u tools/run_kpack_llama_ab.py "${args[@]}" "$@"
