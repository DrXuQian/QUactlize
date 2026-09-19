#!/usr/bin/env bash
# TP2 caller build, two-device arithmetic gate, then the matched 122B model run.
(
    set -Eeuo pipefail
    RUN= stage=precheck
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n $RUN && -d $RUN/results ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
            tar --exclude='*.asysrep' --exclude='*.sqlite*' --exclude='*.tgz' \
                -czf "$RUN.results.tgz" -C "$RUN" results || rc=1
            printf 'results=%s.results.tgz\n' "$RUN"
            if [[ -d $RUN/results/trace ]]; then printf 'Asys: %s/results/trace/*/{reference,native}/proof.asysrep\n' "$RUN"; fi
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "KPACK_TP2 FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/kpack_batched_tp2_122b.json"
    cd "$ROOT"
    TP2_MODE=${TP2_MODE:-model}
    [[ $TP2_MODE == model || $TP2_MODE == communication || $TP2_MODE == q4-local || $TP2_MODE == chain ]]
    PYTHON=$(command -v "${PYTHON:-python3}")
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -r "$SDK/envsetup.sh" && test -x "$SDK/bin/hgobjdump"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK" CUDA_HOME="$SDK/CUDA_SDK" PPU_SDK_HOME="$SDK/CUDA_SDK"
    export DG_JIT_HGCC_COMPILER="$SDK/bin/hgcc"
    if [[ ! -x "$DG_JIT_HGCC_COMPILER" ]]; then
        printf 'KPACK_TP2 SDK missing executable: %s\n' "$DG_JIT_HGCC_COMPILER" >&2
        false
    fi
    # The native PPU API and CUDA compatibility API have separate include roots.
    for header in "$SDK/include/hggc_runtime_api.h" "$CUDA_HOME/include/cuda_runtime_api.h"; do
        if [[ ! -r "$header" ]]; then
            printf 'KPACK_TP2 SDK missing readable header: %s\n' "$header" >&2
            false
        fi
    done
    printf 'KPACK_TP2_SDK PASS compiler=%s native_include=%s cuda_include=%s\n' \
        "$DG_JIT_HGCC_COMPILER" "$SDK/include" "$CUDA_HOME/include"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ ]]
    [[ ${CUDA_VISIBLE_DEVICES%,*} != ${CUDA_VISIBLE_DEVICES#*,} ]]
    JOBS=${JOBS:-192}
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    LLAMA_DIR=$(realpath -e -- "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp}")
    NCP_SOURCE=$(realpath -e -- "${NCP_LIB_DIR:-/sim/eec/shared/junfu.qx/ncp_flash_lib}")
    test -f "$LLAMA_DIR/.aoneci/scripts/build.sh" && test -f "$NCP_SOURCE/CMakeLists.txt"
    grep -q 'ggml_backend_meta_buffer_type_count' "$LLAMA_DIR/ggml/include/ggml-backend.h"
    grep -q 'KPACK_TP2_DEVICE PASS' "$LLAMA_DIR/tests/test-quactlize-scheduler.cpp"
    grep -q 'KPACK_TP2_CACHE PASS' "$LLAMA_DIR/tests/test-quactlize-scheduler.cpp"
    grep -q 'KPACK_TP2_HOST PASS' "$LLAMA_DIR/tests/test-quactlize-scheduler.cpp"
    if [[ $TP2_MODE == communication ]]; then grep -q 'KPACK_TP2_COMM_WRAPPER' "$LLAMA_DIR/tests/test-quactlize-scheduler.cpp"; fi
    if [[ $TP2_MODE == q4-local ]]; then grep -q 'KPACK_TP2_SINGLE PASS' "$LLAMA_DIR/tests/test-quactlize-scheduler.cpp"; fi
    if [[ $TP2_MODE == chain ]]; then
        grep -q -- '--tp2-chain-dump' "$LLAMA_DIR/tests/test-quactlize-scheduler.cpp"
        "$PYTHON" -c 'import numpy'
    fi
    if [[ -z ${QUACTLIZE_PPU_BUNDLE:-} ]]; then
        printf 'KPACK_TP2 missing QUACTLIZE_PPU_BUNDLE: use the six-library bundle path recorded in the previous results/communication/environment.json\n' >&2
        false
    fi
    if [[ ! -s $QUACTLIZE_PPU_BUNDLE/manifest.json ]]; then
        printf 'KPACK_TP2 missing six-library manifest: %s/manifest.json\n' "$QUACTLIZE_PPU_BUNDLE" >&2
        false
    fi
    ASYS=${ASYS:-$SDK/asight/bin/asys}
    CORPUS=${GSM8K_FILE:-/sim/eec/shared/AI_workspace/llm-models/datasets/gsm8k/main/test-00000-of-000001.parquet}
    if [[ $TP2_MODE == model ]]; then test -x "$ASYS" && test -s "$CORPUS"; fi
    RUN=$(mktemp -d "$RESULT_DIR/kpack-tp2.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    mkdir "$RUN/results"
    printf 'KPACK_TP2 run=%s devices=%s jobs=%s\n' "$RUN" "$CUDA_VISIBLE_DEVICES" "$JOBS"
    git rev-parse HEAD > "$RUN/results/quactlize-source.txt"
    cp "$QUACTLIZE_PPU_BUNDLE/manifest.json" "$RUN/results/compatibility-bundle-manifest.json"
    if "$PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("schema") != "quactlize.ppu-compatibility-overlay.v1")' "$QUACTLIZE_PPU_BUNDLE/manifest.json"; then
        "$PYTHON" tools/build_kpack_tp2_q4.py verify "$QUACTLIZE_PPU_BUNDLE"
    fi

    if [[ $TP2_MODE == model ]]; then
        stage=model-paths
        "$PYTHON" tools/resolve_kpack_batched_models.py --plan "${MODEL_PLAN:-$ROOT/tools/kpack_batched_tp2_122b.json}" \
            --model-root "${MODEL_ROOT:-/sim/eec/shared/AI_workspace/llm-models}" --output "$RUN/results/model-plan.json"
        "$PYTHON" -c 'import json,os,sys; p=sys.argv[1]; m=json.load(open(p)); assert all(x["split"]=="tensor" for x in m["models"]); [x.update(devices=os.environ["CUDA_VISIBLE_DEVICES"]) for x in m["models"]]; json.dump(m,open(p,"w"),indent=2)' "$RUN/results/model-plan.json"
    fi

    stage=runtime
    mapfile -t PIN < <("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["branch"]); print(p["commit"]); print(p["path"]); print(p["manifest_sha256"])' tools/kpack_q4_model_artifact.json)
    [[ ${#PIN[@]} == 4 && ${PIN[1]} =~ ^[0-9a-f]{40}$ && ${PIN[2]} == prebuilt/ppu0010/kpack-model-runtime-v1 && ${PIN[3]} =~ ^[0-9a-f]{64}$ ]]
    if [[ -n ${KPACK_BUNDLE:-} ]]; then
        BUNDLE=$(realpath -e -- "$KPACK_BUNDLE")
    else
        ART="$RESULT_DIR/quactlize-model-artifact-${PIN[1]:0:10}"
        git fetch origin "${PIN[0]}"
        if [[ ! -e $ART ]]; then GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$ART" "${PIN[1]}"; fi
        test "$(git -C "$ART" rev-parse HEAD)" == "${PIN[1]}"
        git -C "$ART" lfs pull origin --include="${PIN[2]}/**" --exclude=''
        BUNDLE="$ART/${PIN[2]}"
    fi
    "$PYTHON" -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2],"runtime manifest differs"' "$BUNDLE/manifest.json" "${PIN[3]}"
    "$PYTHON" tools/verify_kpack_dispatch.py "$BUNDLE" --sdk "$SDK" | tee "$RUN/results/verify.log"
    cp "$BUNDLE/manifest.json" "$RUN/results/bundle-manifest.json"

    stage=caller-build
    BUILD_ARGS=(--local-llama)
    if [[ -n ${LLAMA_CI_BUILD_DIR:-} ]]; then BUILD_ARGS+=(--reuse-llama-build "$LLAMA_CI_BUILD_DIR"); fi
    if [[ -n ${NCP_CI_DIR:-} ]]; then BUILD_ARGS+=(--reuse-ncp-build "$NCP_CI_DIR"); fi
    "$PYTHON" -u tools/build_kpack_model_ci.py --llama "$LLAMA_DIR" --ncp "$NCP_SOURCE" --sdk "$SDK" \
        --output "$RUN/ci" --receipt "$RUN/results/caller-ci-build.json" --jobs "$JOBS" "${BUILD_ARGS[@]}" \
        2>&1 | tee "$RUN/results/caller-build.log"
    BUILD_DIR=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["build"])' "$RUN/results/caller-ci-build.json")
    test -n "$BUILD_DIR" && test -x "$BUILD_DIR/bin/test-quactlize-scheduler"
    test -x "$BUILD_DIR/bin/test-quactlize-tp-graph"
    export LD_LIBRARY_PATH="$BUILD_DIR/bin:$LD_LIBRARY_PATH"
    export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-$RESULT_DIR/kpack-tp2-deepgemm-jit}
    unset DG_LIBRARY_ROOT GGML_NCP_FA_LIB GGML_NCP_MOE_LIB
    export QUACTLIZE_PPU_PACK_LIBRARY="$BUNDLE/pack/libquactlize_ppu_pack.so"
    export QUACTLIZE_KPACK_EXECUTION="$BUNDLE" QUACTLIZE_KPACK_ROUTE=auto QUACTLIZE_KPACK_PAIR_WEIGHTS=1
    export QUACTLIZE_KPACK_COMPUTE=bf16 QUACTLIZE_KPACK_GATE_UP=1
    export QUACTLIZE_KPACK_JIT_HELPER="$ROOT/tools/kpack_jit.py" QUACTLIZE_KPACK_JIT_PYTHON="$PYTHON"
    export QUACTLIZE_KPACK_JIT_CACHE=${JIT_CACHE:-$RESULT_DIR/kpack-model-jit-cache}
    export QUACTLIZE_KPACK_DEEPGEMM_HELPER="$BUNDLE/kpack_deepgemm_prewarm.py"
    unset QUACTLIZE_KPACK_PREFILL_POLICY QUACTLIZE_KPACK_GEMV_POLICY GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_DISABLE_FUSION
    mkdir -p -- "$QUACTLIZE_KPACK_JIT_CACHE" "$DG_JIT_CACHE_DIR"
    stage=host-regression
    ctest --test-dir "$BUILD_DIR" -R '^(test-quactlize-loader|test-quactlize-loader-env|test-kpack-sidecar|test-quactlize-buffer|test-quactlize-tp-graph)$' \
        --output-on-failure 2>&1 | tee "$RUN/results/host-tests.log"
    if [[ $TP2_MODE == communication ]]; then
        stage=communication
        "$PYTHON" -u tools/run_kpack_tp2_comm.py --binary "$BUILD_DIR/bin/test-quactlize-scheduler" \
            --output "$RUN/results/communication" --wrapper-ab 2>&1 | tee "$RUN/results/communication.log"
        stage=diagnostic-complete
        exit 0
    fi
    if [[ $TP2_MODE == q4-local ]]; then
        stage=q4-local
        "$PYTHON" -u tools/run_kpack_tp2_comm.py --binary "$BUILD_DIR/bin/test-quactlize-scheduler" \
            --output "$RUN/results/q4-local" --q4-local 2>&1 | tee "$RUN/results/q4-local.log"
        stage=diagnostic-complete
        exit 0
    fi
    if [[ $TP2_MODE == chain ]]; then
        stage=chain-precision
        "$PYTHON" -u tools/run_kpack_tp2_chain.py --binary "$BUILD_DIR/bin/test-quactlize-scheduler" \
            --output "$RUN/results/chain-precision" 2>&1 | tee "$RUN/results/chain-precision.log"
        stage=diagnostic-complete
        exit 0
    fi
    mkdir "$RUN/device-cache"
    stage=two-device-cache-cold
    "$BUILD_DIR/bin/test-quactlize-scheduler" --tp2-cache-write "$RUN/device-cache" 2>&1 | tee "$RUN/results/tp2-device-cold.log"
    grep -qx 'KPACK_TP2_CACHE PASS mode=cold cases=72 chains=6' "$RUN/results/tp2-device-cold.log"
    stage=two-device-cache-hot
    "$BUILD_DIR/bin/test-quactlize-scheduler" --tp2-cache-read "$RUN/device-cache" 2>&1 | tee "$RUN/results/tp2-device-hot.log"
    grep -qx 'KPACK_TP2_CACHE PASS mode=hot cases=72 chains=6' "$RUN/results/tp2-device-hot.log"

    CACHE_ROOT=${CACHE_ROOT:-$RESULT_DIR/kpack-tp2-model-cache}
    mkdir -p -- "$CACHE_ROOT"
    printf 'KPACK_TP2 cache=%s\n' "$CACHE_ROOT"
    COMMON=(--llama "$LLAMA_DIR" --build "$BUILD_DIR" --bundle "$BUNDLE" --plan "$RUN/results/model-plan.json"
        --cache "$CACHE_ROOT" --jit-cache "$QUACTLIZE_KPACK_JIT_CACHE" --logits "$RUN/logits" --corpus "$CORPUS"
        --asys "$ASYS" --inspector "$SDK/bin/hgobjdump")
    stage=model-numerical
    "$PYTHON" -u tools/run_kpack_model_validation.py "${COMMON[@]}" --phase numerical \
        --output "$RUN/results/numerical" 2>&1 | tee "$RUN/results/numerical.log"
    stage=model-perf
    "$PYTHON" -u tools/run_kpack_batched_bench.py --binary "$BUILD_DIR/bin/llama-batched-bench" \
        --llama-dir "$LLAMA_DIR" --bundle "$BUNDLE" --jit-cache "$QUACTLIZE_KPACK_JIT_CACHE" \
        --cache-root "$CACHE_ROOT" --output-root "$RESULT_DIR" --output "$RUN/results/benchmark" \
        --plan "$RUN/results/model-plan.json" --order abba --require-selected --repeats "${MODEL_REPEATS:-2}" \
        2>&1 | tee "$RUN/results/benchmark.log"
    stage=model-trace
    "$PYTHON" -u tools/run_kpack_model_validation.py "${COMMON[@]}" --phase trace \
        --output "$RUN/results/trace" 2>&1 | tee "$RUN/results/trace.log"
    stage=complete
)
