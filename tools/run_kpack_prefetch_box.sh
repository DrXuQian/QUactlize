#!/usr/bin/env bash
# Invoke with bash, never source: failures must not close the caller's shell.
set -Eeuo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${PPU_SDK:?Set PPU_SDK to the SDK root containing bin/hgcc and lib/}"
PYTHON=$(command -v "${PYTHON:-python3}")
test -x "$PYTHON"
test -f "$PPU_SDK/envsetup.sh"
test -d "${RESULT_ROOT:-/workspace}"
RUN=$(mktemp -d "${RESULT_ROOT:-/workspace}/kpack-prefetch.XXXXXX")
test -n "$RUN" && test -d "$RUN"
mkdir "$RUN/results"
finish() {
    rc=$?
    trap - EXIT
    printf 'runner_rc=%s\n' "$rc" | tee "$RUN/results/runner.txt"
    if tar -czf "$RUN.results.tgz" -C "$RUN" results; then
        printf 'results=%s.results.tgz\n' "$RUN"
    else
        printf 'archive failed; raw results preserved at %s/results\n' "$RUN" >&2
    fi
    exit "$rc"
}
trap finish EXIT
cases=(q12 q13)
options=()
while (($#)); do
    case "$1" in
        --case)
            if (($# < 2)); then
                printf 'missing --case value\n' >&2
                exit 2
            fi
            case "$2" in
                q12|q13|q4-to-q5) cases=("$2") ;;
                *) printf 'unknown prefetch case: %s\n' "$2" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        *) options+=("$1"); shift ;;
    esac
done
# Keep the chosen Python interpreter when SDK setup changes PATH.
set +u
source "$PPU_SDK/envsetup.sh"
set -Eeuo pipefail
unset DISABLE_PPU_INIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:$PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"
"$PYTHON" -c 'import sys, numpy, gguf; assert sys.version_info >= (3, 11), "Python >=3.11 required"'
git rev-parse HEAD > "$RUN/results/source.txt"
printf 'KPACK_PREFETCH_START no_compilation=1 profiler=none results=%s/results\n' "$RUN"
failed=0
for q in "${cases[@]}"; do
    if "$PYTHON" -u tools/run_kpack_prefetch.py --sdk "$PPU_SDK" \
        --case "$q" --output "$RUN/results" "${options[@]}" 2>&1 | tee "$RUN/results/$q.log"; then
        printf 'KPACK_PREFETCH_CASE case=%s status=PASS\n' "$q"
    else
        failed=1
        printf 'KPACK_PREFETCH_CASE case=%s status=FAIL remaining_cases_continue=1\n' "$q"
    fi
done
printf 'KPACK_PREFETCH_DONE failed=%s results=%s/results\n' "$failed" "$RUN"
exit "$failed"
