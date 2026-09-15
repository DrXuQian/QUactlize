#!/usr/bin/env bash
# Prebuilt local-work closure: independent experiments, no inference defaults changed.
(
    set -Eeuo pipefail
    RUN= stage=precheck
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
            if tar -czf "$RUN.results.tgz" -C "$RUN" results; then
                printf 'results=%s.results.tgz\n' "$RUN"
            else rc=1; fi
        fi
        printf 'runner_rc=%s\nCurrent Docker shell is preserved.\n' "$rc"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "LOCAL_CLOSURE FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/kpack_q4_model_artifact.json"
    cd "$ROOT"
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -r "$SDK/envsetup.sh"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK" LC_ALL=C CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$PYTHON" -c 'import numpy,gguf,torch; import platform; n,v=platform.libc_ver(); assert n=="glibc" and tuple(map(int,v.split(".")[:2]))>=(2,38)'
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    RUN=$(mktemp -d "$RESULT_DIR/kpack-local-closure.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    mkdir "$RUN/results"
    # 64 MiB is the operator-verified PPU-ZW810 capacity; a positive runtime
    # query must agree. This is not a fallback for an arbitrary device.
    L2_BYTES=${L2_BYTES:-67108864}
    [[ "$L2_BYTES" =~ ^[1-9][0-9]*$ ]]
    if [[ -z ${BUNDLE:-} ]]; then
        stage=fetch
        mapfile -t INFO < <("$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["branch"]); print(m["commit"]); print(m["path"]); print(m["manifest_sha256"])' tools/kpack_q4_model_artifact.json)
        [[ ${#INFO[@]} == 4 && ${INFO[0]} == artifacts/kpack-model-runtime-v1 && ${INFO[1]} =~ ^[0-9a-f]{40}$ && ${INFO[2]} == prebuilt/ppu0010/kpack-model-runtime-v1 && ${INFO[3]} =~ ^[0-9a-f]{64}$ ]]
        ART="$RESULT_DIR/quactlize-model-artifact-${INFO[1]:0:10}"
        git fetch origin "${INFO[0]}"
        git cat-file -e "${INFO[1]}^{commit}"
        if [[ -e "$ART" ]]; then
            test -d "$ART" && test "$(git -C "$ART" rev-parse HEAD)" == "${INFO[1]}"
        else
            GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$ART" "${INFO[1]}"
        fi
        git -C "$ART" lfs pull origin --include="${INFO[2]}/**" --exclude=""
        BUNDLE="$ART/${INFO[2]}"
        "$PYTHON" -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2]' "$BUNDLE/manifest.json" "${INFO[3]}"
    fi
    BUNDLE=$(realpath -e -- "$BUNDLE")
    stage=verify
    "$PYTHON" tools/verify_kpack_dispatch.py "$BUNDLE" --sdk "$SDK" | tee "$RUN/results/verify.log"
    "$PYTHON" -c 'import json,sys; from pathlib import Path; from tools.attach_kpack_local_gates import payload_paths; b=Path(sys.argv[1]); m=json.load(open(b/"manifest.json")); payload_paths(b,m["local_optimization_gate"],sdk=Path(sys.argv[2]),check_source=True)' "$BUNDLE" "$SDK"
    cp "$BUNDLE/manifest.json" "$RUN/results/bundle-manifest.json"
    printf 'LOCAL_CLOSURE START no_compile=1 no_model=1 production_defaults=UNCHANGED results=%s/results\n' "$RUN"
    failed=0
    stage=q8-numeric
    if "$PYTHON" -u dev/gemv_simt/q8_vector_run.py --bundle "$BUNDLE/local-gates/q8" --sdk "$SDK" \
        --phase numeric --output "$RUN/results/q8-numeric.json" 2>&1 | tee "$RUN/results/q8-numeric.log"; then
        stage=q8-performance
        if "$PYTHON" -u dev/gemv_simt/q8_vector_campaign.py --bundle "$BUNDLE/local-gates/q8" --sdk "$SDK" \
            --l2-bytes "$L2_BYTES" --output "$RUN/results/q8-perf" 2>&1 | tee "$RUN/results/q8-perf.log"; then :; else failed=$((failed+1)); fi
    else failed=$((failed+1)); fi
    stage=moe-prepare
    if "$PYTHON" -u dev/moe_prepare/run.py --bundle "$BUNDLE/local-gates/moe" \
        --output "$RUN/results/moe-prepare" 2>&1 | tee "$RUN/results/moe-prepare.log"; then :; else failed=$((failed+1)); fi
    stage=bf16-capability
    if "$PYTHON" -u dev/bf16_compute/run.py --sdk "$SDK" --package "$BUNDLE/bf16" \
        --output "$RUN/results/bf16" --repeats 2 --samples 0 2>&1 | tee "$RUN/results/bf16.log"; then :; else failed=$((failed+1)); fi
    stage=bf16-selected-q4
    if "$PYTHON" -u dev/bf16_fastpath/gate.py --sdk "$SDK" --bundle "$BUNDLE/q4-bf16-gate" \
        --output "$RUN/results/bf16-selected-q4" 2>&1 | tee "$RUN/results/bf16-selected-q4.log"; then :; else failed=$((failed+1)); fi
    stage=complete
    printf 'LOCAL_CLOSURE DONE failures=%s production_admission=PENDING_REVIEW\n' "$failed"
    [[ "$failed" == 0 ]]
)
