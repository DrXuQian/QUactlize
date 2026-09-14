#!/usr/bin/env bash
# Fetch only the pinned small composition package. Preserve the Docker shell.
(
    set -Eeuo pipefail
    RUN=""
    stage=precheck
    finish() {
        rc=$?
        trap - EXIT
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            if tar -czf "$RUN.results.tgz" -C "$RUN" results; then
                printf '\nresults=%s.results.tgz\n' "$RUN"
            else
                printf 'Results remain at %s/results\n' "$RUN" >&2
                rc=1
            fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "KPACK_PREFILL_COMPOSITION FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_prefill_runtime.py"
    cd "$ROOT"
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/lib/libhggc_wrapper.so"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    "$PYTHON" -c 'import numpy, gguf, torch; from deep_gemm.jit_kernels.m_grouped_gemm import m_grouped_gemm_bf16_bf16_bf16_nt_nopad'
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    if [[ -z ${BUNDLE:-} ]]; then
        stage=fetch
        RECEIPT="$ROOT/tools/kpack_prefill_runtime_artifact.json"
        mapfile -t INFO < <("$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["branch"]); print(m["commit"]); print(m["path"]); print(m["manifest_sha256"])' "$RECEIPT")
        [[ ${#INFO[@]} == 4 && ${INFO[0]} == artifacts/kpack-prefill-runtime-v1 && ${INFO[1]} =~ ^[0-9a-f]{40}$ && ${INFO[2]} == prebuilt/ppu0010/kpack-prefill-runtime-v1 && ${INFO[3]} =~ ^[0-9a-f]{64}$ ]]
        ART="$RESULT_DIR/quactlize-prefill-artifact-${INFO[1]:0:10}"
        git fetch origin "${INFO[0]}"
        git cat-file -e "${INFO[1]}^{commit}"
        if [[ -e "$ART" ]]; then
            test -d "$ART" && test "$(git -C "$ART" rev-parse HEAD)" == "${INFO[1]}"
        else
            GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$ART" "${INFO[1]}"
        fi
        git -C "$ART" lfs pull origin --include="${INFO[2]}/**" --exclude=""
        BUNDLE="$ART/${INFO[2]}"
        "$PYTHON" -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2],"artifact manifest differs"' "$BUNDLE/manifest.json" "${INFO[3]}"
    fi
    BUNDLE=$(realpath -e -- "$BUNDLE")
    test -n "$BUNDLE" && test -f "$BUNDLE/libquactlize_ppu_prefill.so"
    stage=verify
    "$PYTHON" tools/verify_kpack_dispatch.py "$BUNDLE" --sdk "$SDK"
    RUN=$(mktemp -d "$RESULT_DIR/kpack-prefill-composition.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    mkdir "$RUN/results"
    printf 'KPACK_PREFILL_COMPOSITION run=%s\nQuactlize compile=NONE DeepGEMM=SELECTED_COMPILE_ONLY first-use=EXCLUDED sweep=NONE\n' "$RUN"
    stage=composition
    "$PYTHON" -u tools/run_kpack_prefill_runtime.py --sdk "$SDK" --bundle "$BUNDLE" --output "$RUN/results" \
        2>&1 | tee "$RUN/results/console.log"
    stage=complete
)
