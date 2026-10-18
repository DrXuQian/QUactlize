#!/usr/bin/env bash
# Reuse the completed supplement's TC images; run only alternating finalists.
set -euo pipefail
confirm_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
confirm_previous=$(realpath -e -- "${PREVIOUS_RUN:?Set PREVIOUS_RUN to the completed seven-card supplement}")
confirm_sdk=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
confirm_python=$(command -v "${PYTHON:-python3}")
test -d "$confirm_previous"
for entry in bin/hgcc include/hggc_runtime_api.h lib/libhggc_wrapper.so; do
    if [[ ! -f "$confirm_sdk/$entry" ]]; then
        printf 'MISSING %s/%s\n' "$confirm_sdk" "$entry" >&2
        exit 2
    fi
done
if [[ "${IDLE_CONFIRMED:-0}" != 1 && "${PLAN_ONLY:-0}" != 1 ]]; then
    printf '%s\n' 'Confirm cards are idle, then set IDLE_CONFIRMED=1.' >&2
    exit 2
fi
export LD_LIBRARY_PATH="$confirm_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
args=(--previous "$confirm_previous" --sdk "$confirm_sdk" --devices "${DEVICES:-1,2,3,4,5,6,7}"
      --run-root "${RUN_ROOT:-/workspace}")
if [[ -n "${REVIEW_ARCHIVE:-}" ]]; then args+=(--review "$REVIEW_ARCHIVE"); fi
if [[ -n "${FINALIST_PLAN:-}" ]]; then args+=(--plan "$FINALIST_PLAN"); fi
if [[ -n "${CONFIRM_RUN:-}" ]]; then args+=(--output "$CONFIRM_RUN"); fi
if [[ "${RESUME:-0}" == 1 ]]; then
    : "${CONFIRM_RUN:?RESUME=1 needs CONFIRM_RUN}"
    args+=(--resume)
fi
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then args+=(--plan-only); else args+=(--idle-confirmed); fi
cd -- "$confirm_source"
"$confirm_python" -u tools/confirm_kpack_prefill.py "${args[@]}"
