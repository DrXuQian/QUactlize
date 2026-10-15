#!/usr/bin/env bash
# One dense BF16 provider, no CUDA Graph, no TC rebuild and no full campaign.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run with bash, not source.' >&2
    return 2
fi
set -euo pipefail
cublas_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cublas_python=$(command -v "${PYTHON:-python3}")
cublas_root=$(realpath -e -- "${RUN_ROOT:-/workspace}")
cublas_sdk=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
test -d "$cublas_root"
: "${CUDA_VISIBLE_DEVICES:?Choose exactly one idle physical device}"
[[ "${IDLE_CONFIRMED:-0}" == 1 ]] || { printf '%s\n' 'Set IDLE_CONFIRMED=1 on an idle card.' >&2; exit 2; }
for path in bin/hgcc asight/bin/asys asight/lib/libhgToolsExt.so lib/libhggc_wrapper.so; do
    [[ -f "$cublas_sdk/$path" ]] || { printf 'MISSING %s\n' "$cublas_sdk/$path" >&2; exit 1; }
done
cublas_run=$(mktemp -d "${cublas_root:?}/kpack-cublas-ordinary.XXXXXXXX")
trap 'cublas_rc=$?; printf "runner_rc=%s run=%s\nCurrent Docker shell is preserved.\n" "$cublas_rc" "$cublas_run"' EXIT
cublas_args=(--sdk "$cublas_sdk" --output "$cublas_run/probe" --idle-confirmed
             --l2-bytes "${L2_BYTES:-0}" --timeout-seconds "${TIMEOUT_SECONDS:-300}")
if [[ -n "${REUSE_BUILDS:-}" ]]; then
    cublas_builds=$(realpath -e -- "$REUSE_BUILDS")
    test -f "$cublas_builds/helpers/manifest.json"
    test -f "$cublas_builds/dequant/manifest.json"
    cublas_args+=(--reuse-builds "$cublas_builds")
fi
export LD_LIBRARY_PATH="$cublas_sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd -- "$cublas_source"
printf 'STRATEGY_CUBLAS_RUN run=%s cuda_graph=0 full_campaign=0\n' "$cublas_run"
"$cublas_python" -u tools/probe_kpack_cublas_ordinary.py "${cublas_args[@]}" 2>&1 | tee "$cublas_run/console.log"
