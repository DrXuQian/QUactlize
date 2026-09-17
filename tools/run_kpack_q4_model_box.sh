#!/usr/bin/env bash
# Joint CI caller build, prebuilt runtime, mixed gate, model timings and Asys.
(
    set -Eeuo pipefail
    RUN= stage=precheck
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n $RUN && -d $RUN/results ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
            if tar --exclude='*.asysrep' --exclude='*.sqlite*' --exclude='*.tgz' \
                -czf "$RUN.results.tgz" -C "$RUN" results; then
                printf '\nresults=%s.results.tgz\n' "$RUN"
            else
                printf 'Archive failed; logs remain at %s/results\n' "$RUN" >&2
                rc=1
            fi
            printf 'Full Asys files: %s/results/trace/*/{reference,native}/proof.asysrep\n' "$RUN"
            printf 'ACU reports and raw counters: %s/results/acu/\n' "$RUN"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "KPACK_Q4_MODEL FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_model_validation.py"
    cd "$ROOT"
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -r "$SDK/envsetup.sh" && test -x "$SDK/bin/hgobjdump"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK" LC_ALL=C CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    ASYS=${ASYS:-$SDK/asight/bin/asys}
    test -x "$ASYS"
    MODEL_ACU=${MODEL_ACU:-0}
    [[ "$MODEL_ACU" == 0 || "$MODEL_ACU" == 1 ]]
    Q8_HOIST_AB=${Q8_HOIST_AB:-0}
    [[ "$Q8_HOIST_AB" == 0 || "$Q8_HOIST_AB" == 1 ]]
    ACU=${ACU:-$SDK/asight/bin/acu}
    if [[ "$MODEL_ACU" == 1 ]]; then test -x "$ACU"; fi
    "$PYTHON" -c 'import numpy, gguf, torch, pyarrow; from deep_gemm.jit_kernels.m_grouped_gemm import m_grouped_gemm_bf16_bf16_bf16_nt_nopad'
    "$PYTHON" -c 'import platform; name,version=platform.libc_ver(); assert name=="glibc" and tuple(map(int,version.split(".")[:2])) >= (2,38), "PPU SDK runtime requires glibc >= 2.38 (Ubuntu 24.04)"'
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    CORPUS=${GSM8K_FILE:-/sim/eec/shared/AI_workspace/llm-models/datasets/gsm8k/main/test-00000-of-000001.parquet}
    test -s "$CORPUS"
    NCP_SOURCE=$(realpath -e -- "${NCP_LIB_DIR:-/sim/eec/shared/junfu.qx/ncp_flash_lib}")
    test -n "$NCP_SOURCE" && test -f "$NCP_SOURCE/CMakeLists.txt"
    JOBS=${JOBS:-192}
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]
    MODEL_PHASES=${MODEL_PHASES:-all}
    [[ "$MODEL_PHASES" == all || "$MODEL_PHASES" == perf ]]
    MODEL_COMPUTE=${MODEL_COMPUTE:-fp16}
    [[ "$MODEL_COMPUTE" == fp16 || "$MODEL_COMPUTE" == bf16 ]]
    export QUACTLIZE_KPACK_COMPUTE="$MODEL_COMPUTE"
    MODEL_GATE_UP=${MODEL_GATE_UP:-0}
    [[ "$MODEL_GATE_UP" == 0 || "$MODEL_GATE_UP" == 1 ]]
    if [[ "$MODEL_GATE_UP" == 1 ]]; then
        [[ "$MODEL_COMPUTE" == bf16 ]]
    fi
    export QUACTLIZE_KPACK_GATE_UP="$MODEL_GATE_UP"
    # Check loader dependencies before spending time on the joint build.
    export QUACTLIZE_PPU_BUNDLE=${QUACTLIZE_PPU_BUNDLE:-/workspace/quactlize-runtime-artifact-2826cf1-46fc3096e1a1/prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1/bundle}
    test -s "$QUACTLIZE_PPU_BUNDLE/manifest.json"
    RUN=$(mktemp -d "$RESULT_DIR/kpack-q4-model.XXXXXX")
    test -n "$RUN" && test -d "$RUN"
    mkdir "$RUN/results"
    printf 'KPACK_Q4_MODEL run=%s\n' "$RUN"
    git rev-parse HEAD > "$RUN/results/quactlize-source.txt"

    stage=model-paths
    MODEL_ARGS=()
    if [[ -n ${MODEL_NAMES:-} ]]; then
        read -ra names <<< "$MODEL_NAMES"
        for name in "${names[@]}"; do MODEL_ARGS+=(--model "$name"); done
    fi
    "$PYTHON" tools/resolve_kpack_batched_models.py --plan "$ROOT/tools/kpack_batched_int4_2048.json" \
        --model-root "${MODEL_ROOT:-/sim/eec/shared/AI_workspace/llm-models}" "${MODEL_ARGS[@]}" \
        --output "$RUN/results/model-plan.json"

    stage=fetch
    mapfile -t INFO < <("$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["branch"]); print(m["commit"]); print(m["path"]); print(m["manifest_sha256"]); print(m["llama_ci_commit"])' "$ROOT/tools/kpack_q4_model_artifact.json")
    [[ ${#INFO[@]} == 5 && ( ${INFO[0]} == artifacts/kpack-model-runtime-v1 || ${INFO[0]} == artifacts/kpack-model-paired-n4-v1 || ${INFO[0]} == artifacts/kpack-model-readers-v1 ) && ${INFO[1]} =~ ^[0-9a-f]{40}$ && ${INFO[2]} == prebuilt/ppu0010/kpack-model-runtime-v1 && ${INFO[3]} =~ ^[0-9a-f]{64}$ && ${INFO[4]} =~ ^[0-9a-f]{40}$ ]]
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
    "$PYTHON" -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2],"model package manifest differs"' "$BUNDLE/manifest.json" "${INFO[3]}"
    "$PYTHON" tools/verify_kpack_dispatch.py "$BUNDLE" --sdk "$SDK" | tee "$RUN/results/verify.log"
    test ! -e "$BUNDLE/llama"
    if [[ "$MODEL_GATE_UP" == 1 ]]; then
        stage=paired-integration-gate
        "$PYTHON" -u tools/run_kpack_gate_up_integration.py --sdk "$SDK" --bundle "$BUNDLE" \
            --output "$RUN/results/paired-integration" 2>&1 | tee "$RUN/results/paired-integration.log"
    fi
    if "$PYTHON" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1])).get("model_reader_gate",False))' "$ROOT/tools/kpack_q4_model_artifact.json"; then
        stage=model-reader-integration
        mapfile -t READER_PIN < <("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["branch"]); print(p["commit"]); print(p["path"])' "$ROOT/tools/kpack_model_gemv_artifact.json")
        [[ ${#READER_PIN[@]} == 3 && ${READER_PIN[0]} == artifacts/model-gemv-reader-v1 && ${READER_PIN[1]} =~ ^[0-9a-f]{40}$ && ${READER_PIN[2]} == prebuilt/ppu0010/model-gemv-reader-v1 ]]
        READER_ART="$RESULT_DIR/quactlize-model-gemv-artifact-${READER_PIN[1]:0:12}"
        if [[ ! -e "$READER_ART" ]]; then
            GIT_LFS_SKIP_SMUDGE=1 git fetch origin "${READER_PIN[0]}"
            git cat-file -e "${READER_PIN[1]}^{commit}"
            GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$READER_ART" "${READER_PIN[1]}"
        fi
        [[ $(git -C "$READER_ART" rev-parse HEAD) == "${READER_PIN[1]}" ]]
        git -C "$READER_ART" lfs pull origin --include="${READER_PIN[2]}/*.so,${READER_PIN[2]}/*.isa.txt" --exclude=''
        "$PYTHON" -u tools/run_model_gemv_integration.py --sdk "$SDK" --bundle "$BUNDLE" \
            --reference "$READER_ART/${READER_PIN[2]}" --l2-bytes "${L2_BYTES:-67108864}" \
            --output "$RUN/results/model-readers" 2>&1 | tee "$RUN/results/model-readers.log"
    fi
    if [[ ${Q8_HOIST_AB:-0} == 1 ]]; then
        stage=q8-hoist-ab
        mapfile -t Q8_BASE < <("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["commit"]); print(p["path"]); print(p["manifest_sha256"])' "$ROOT/tools/kpack_q8_hoist_baseline.json")
        [[ ${#Q8_BASE[@]} == 3 && ${Q8_BASE[0]} =~ ^[0-9a-f]{40}$ && ${Q8_BASE[1]} == prebuilt/ppu0010/kpack-model-runtime-v1 && ${Q8_BASE[2]} =~ ^[0-9a-f]{64}$ ]]
        Q8_ART="$RESULT_DIR/quactlize-model-artifact-${Q8_BASE[0]:0:10}"
        if [[ -e "$Q8_ART" ]]; then
            test -d "$Q8_ART" && test "$(git -C "$Q8_ART" rev-parse HEAD)" == "${Q8_BASE[0]}"
        else
            GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$Q8_ART" "${Q8_BASE[0]}"
        fi
        git -C "$Q8_ART" lfs pull origin --include="${Q8_BASE[1]}/libquactlize_ppu_execution.so,${Q8_BASE[1]}/manifest.json" --exclude=""
        printf 'Q8_HOIST_AB start cases=3 configs=UNCHANGED cache=ROTATING roof_gbps=%s; keep this GPU idle\n' "${PEAK_BANDWIDTH_GBPS:-2700}"
        "$PYTHON" -u tools/run_kpack_q8_hoist.py --sdk "$SDK" --candidate "$BUNDLE" \
            --baseline "$Q8_ART/${Q8_BASE[1]}" --output "$RUN/results/q8-hoist" \
            --l2-bytes "${L2_BYTES:-0}" --peak-gbps "${PEAK_BANDWIDTH_GBPS:-2700}" \
            2>&1 | tee "$RUN/results/q8-hoist.log"
    fi
    stage=router-alias-gate
    if "$PYTHON" -c 'import json,sys; sys.exit("router_alias_gate" not in json.load(open(sys.argv[1])))' "$BUNDLE/manifest.json"; then
        "$BUNDLE/router-alias/bench" --router-alias-check 2>&1 | tee "$RUN/results/router-alias.log"
        grep -qx 'MOE_ROUTER_ALIAS PASS cases=360 replays=4 arms=2 scope=M1_TOP8_LOGITS_REUSE' "$RUN/results/router-alias.log"
        stage=model-decode-gate
        "$PYTHON" -u tools/check_kpack_model_decode_updates.py --sdk "$SDK" \
            --library "$BUNDLE/libquactlize_ppu_execution.so" --output "$RUN/results/model-decode-gate.json" \
            2>&1 | tee "$RUN/results/model-decode-gate.log"
    fi
    stage=production-q8-gate
    "$PYTHON" -u tools/run_kpack_decode_updates.py --sdk "$SDK" --bundle "$BUNDLE" \
        --output "$RUN/results/production-q8.json" 2>&1 | tee "$RUN/results/production-q8.log"
    if [[ "$MODEL_COMPUTE" == bf16 ]]; then
        stage=bf16-metadata
        "$PYTHON" -u dev/bf16_compute/check_metadata.py --sdk "$SDK" --bundle "$BUNDLE" \
            --output "$RUN/results/bf16-metadata" 2>&1 | tee "$RUN/results/bf16-metadata.log"
        stage=bf16-capability
        test -s "$BUNDLE/bf16/manifest.json"
        "$PYTHON" -u dev/bf16_compute/run.py --sdk "$SDK" --package "$BUNDLE/bf16" \
            --output "$RUN/results/bf16" --repeats 2 --samples 0 2>&1 | tee "$RUN/results/bf16.log"
        stage=bf16-selected-q4
        "$PYTHON" -u dev/bf16_fastpath/gate.py --sdk "$SDK" --bundle "$BUNDLE/q4-bf16-gate" \
            --output "$RUN/results/bf16-selected-q4" 2>&1 | tee "$RUN/results/bf16-selected-q4.log"
        stage=bf16-matched-prefill
        "$PYTHON" -u dev/bf16_compute/matched.py --sdk "$SDK" --package "$BUNDLE/bf16" \
            --output "$RUN/results/bf16-matched-prefill" 2>&1 | tee "$RUN/results/bf16-matched-prefill.log"
    fi
    stage=caller-source
    CI_SOURCE_ARGS=()
    if [[ -n ${LLAMA_CI_DIR:-} ]]; then
        LLAMA_DIR=$(realpath -e -- "$LLAMA_CI_DIR")
        test -f "$LLAMA_DIR/.aoneci/scripts/build.sh"
        CI_SOURCE_ARGS+=(--local-llama)
        if [[ -n ${LLAMA_CI_BUILD_DIR:-} ]]; then
            CI_SOURCE_ARGS+=(--reuse-llama-build "$LLAMA_CI_BUILD_DIR")
        fi
        printf 'KPACK_Q4_MODEL caller_source=LOCAL_WORKTREE path=%s local_edits=INCLUDED\n' "$LLAMA_DIR"
    else
        if [[ -n ${LLAMA_CI_BUILD_DIR:-} ]]; then
            printf 'LLAMA_CI_BUILD_DIR requires LLAMA_CI_DIR\n' >&2
            false
        fi
        LLAMA_DIR="$RESULT_DIR/llama-model-source-${INFO[4]:0:10}"
        if [[ ! -e "$LLAMA_DIR" ]]; then
            LLAMA_BRANCH=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("llama_branch","dev/quactlize-v0.3.0"))' "$ROOT/tools/kpack_q4_model_artifact.json")
            [[ "$LLAMA_BRANCH" == dev/quactlize-v0.3.0 || "$LLAMA_BRANCH" == dev/quactlize-gate-up-v0.3.0 ]]
            git clone --no-checkout --depth 1 --single-branch --branch "$LLAMA_BRANCH" \
                https://github.com/DrXuQian/llama.cpp.git "$LLAMA_DIR"
        fi
        test "$(git -C "$LLAMA_DIR" rev-parse --show-toplevel)" == "$LLAMA_DIR"
        if ! git -C "$LLAMA_DIR" cat-file -e "${INFO[4]}^{commit}" 2>/dev/null; then
            git -C "$LLAMA_DIR" fetch --depth 1 origin "${INFO[4]}"
        fi
        git -C "$LLAMA_DIR" cat-file -e "${INFO[4]}^{commit}"
        CI_SOURCE_ARGS+=(--llama-revision "${INFO[4]}")
        printf 'KPACK_Q4_MODEL caller_source=PINNED_COMMIT revision=%s cache=%s\n' "${INFO[4]}" "$LLAMA_DIR"
    fi

    stage=ci-build
    NCP_BUILD_ARGS=()
    if [[ -n ${NCP_CI_DIR:-} ]]; then
        NCP_BUILD_ARGS+=(--reuse-ncp-build "$NCP_CI_DIR")
    fi
    "$PYTHON" -u tools/build_kpack_model_ci.py --llama "$LLAMA_DIR" --ncp "$NCP_SOURCE" \
        --sdk "$SDK" --output "$RUN/ci" --jobs "$JOBS" \
        "${CI_SOURCE_ARGS[@]}" "${NCP_BUILD_ARGS[@]}" --receipt "$RUN/results/caller-ci-build.json" 2>&1 | tee "$RUN/results/caller-ci-build.log"
    if [[ -z ${LLAMA_CI_DIR:-} ]]; then
        LLAMA_DIR="$RUN/ci/llama"
        BUILD_DIR="$LLAMA_DIR/build-ci"
    else
        BUILD_DIR=$(realpath -e -- "${LLAMA_CI_BUILD_DIR:-$RUN/ci/llama-build}")
    fi
    export CUDA_HOME="$SDK/CUDA_SDK" DG_JIT_CACHE_DIR="$RUN/ci/ncp-jit-cache"
    unset DG_LIBRARY_ROOT GGML_NCP_FA_LIB GGML_NCP_MOE_LIB
    export LD_LIBRARY_PATH="$BUILD_DIR/bin:$LD_LIBRARY_PATH"
    if [[ "$Q8_HOIST_AB" == 1 ]]; then
        "$PYTHON" -c 'import sys; sys.path.insert(0,sys.argv[1]+"/tests"); from quactlize_native import simt_symbol_recipe; assert simt_symbol_recipe("quactlize::execution::simt::q8_vector::kernel<1,0,1,4,8,4,true>")== (8,1,5,4,8,4,0), "update the llama trace parser for Q8 hoist"' "$LLAMA_DIR"
    fi
    if "$PYTHON" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1])).get("model_reader_gate",False))' "$ROOT/tools/kpack_q4_model_artifact.json"; then
        "$PYTHON" -c 'import sys; sys.path.insert(0,sys.argv[1]+"/tests"); from quactlize_native import simt_symbol_recipe,paired_symbol_recipe; assert simt_symbol_recipe("quactlize::execution::simt::q8_vector::kernel_model<1,0,1,8,4,4,true,8192,2048,1>")== (8,1,5,8,4,4,0), "update the llama reader trace parser"; assert paired_symbol_recipe("quactlize::fusion::simt_gate_up_model<12,1,1,8>"), "update the llama paired trace parser"' "$LLAMA_DIR"
    fi
    "$BUILD_DIR/bin/llama-batched-bench" --help > "$RUN/results/binary-help.log" 2>&1

    export QUACTLIZE_PPU_PACK_LIBRARY="$BUNDLE/pack/libquactlize_ppu_pack.so"
    test -s "$QUACTLIZE_PPU_PACK_LIBRARY"
    export QUACTLIZE_KPACK_EXECUTION="$BUNDLE" QUACTLIZE_KPACK_ROUTE=auto QUACTLIZE_KPACK_PAIR_WEIGHTS=1
    export QUACTLIZE_KPACK_JIT_HELPER="$ROOT/tools/kpack_jit.py" QUACTLIZE_KPACK_JIT_PYTHON="$PYTHON"
    export QUACTLIZE_KPACK_JIT_CACHE=${JIT_CACHE:-$RESULT_DIR/kpack-model-jit-cache}
    export QUACTLIZE_KPACK_DEEPGEMM_HELPER="$BUNDLE/kpack_deepgemm_prewarm.py"
    unset QUACTLIZE_KPACK_PREFILL_POLICY QUACTLIZE_KPACK_GEMV_POLICY GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_DISABLE_FUSION
    mkdir -p -- "$QUACTLIZE_KPACK_JIT_CACHE"
    CACHE_DIR=${CACHE_DIR:-$RESULT_DIR/kpack-model-cache}
    mkdir -p -- "$CACHE_DIR"
    cp "$BUNDLE/manifest.json" "$RUN/results/bundle-manifest.json"
    git -C "$LLAMA_DIR" rev-parse HEAD > "$RUN/results/llama-source.txt"

    stage=mixed-decode-gate
    printf 'KPACK_Q4_MODEL caller=AONECI runtime=PREBUILT grouped_compute=%s dense_policy=FP16 full_sweep=NONE model_prewarm=SELECTED_JIT_ONLY\n' "$MODEL_COMPUTE"
    failed=0
    if [[ "$MODEL_PHASES" == all ]] && "$BUNDLE/mixed-stages" --mixed 2>&1 | tee "$RUN/results/mixed-stages.log"; then
        grep -qx 'KPACK_MOE_MIXED_STAGES PASS cells=80 PPU_GEMM_ADMISSION=NOT_TESTED' "$RUN/results/mixed-stages.log"
    elif [[ "$MODEL_PHASES" == all ]]; then failed=$((failed+1)); fi
    GATE_ARGS=(--smallm-table)
    if [[ "$MODEL_GATE_UP" == 1 ]]; then GATE_ARGS+=(--paired); fi
    if [[ "$MODEL_PHASES" == perf ]]; then
        printf 'KPACK_Q4_MODEL accuracy=NOT_RETESTED scope=SMALLM_COMPOSITION_PLUS_PERFORMANCE\n'
    fi
    if "$PYTHON" -u tools/run_kpack_moe_gate.py --mixed --sdk "$SDK" --bundle "$BUNDLE" \
        --pack-library "$QUACTLIZE_PPU_PACK_LIBRARY" --jit-cache "$QUACTLIZE_KPACK_JIT_CACHE" \
        --output "$RUN/results/mixed-chain" --samples 3 --compute "$MODEL_COMPUTE" "${GATE_ARGS[@]}" 2>&1 | tee "$RUN/results/mixed-chain.log"; then :; else failed=$((failed+1)); fi
    [[ $failed == 0 ]]

    COMMON=(--llama "$LLAMA_DIR" --build "$BUILD_DIR" --bundle "$BUNDLE" --plan "$RUN/results/model-plan.json"
        --cache "$CACHE_DIR" --jit-cache "$QUACTLIZE_KPACK_JIT_CACHE" --logits "$RUN/logits" --corpus "$CORPUS"
        --asys "$ASYS" --inspector "$SDK/bin/hgobjdump")
    if [[ "$MODEL_PHASES" == all ]]; then
        stage=model-numerical
        "$PYTHON" -u tools/run_kpack_model_validation.py "${COMMON[@]}" --phase numerical \
            --output "$RUN/results/numerical" 2>&1 | tee "$RUN/results/numerical.log"
    fi
    stage=model-benchmark
    if "$PYTHON" -u tools/run_kpack_batched_bench.py --binary "$BUILD_DIR/bin/llama-batched-bench" \
        --llama-dir "$LLAMA_DIR" --bundle "$BUNDLE" --jit-cache "$QUACTLIZE_KPACK_JIT_CACHE" \
        --cache-root "$CACHE_DIR" --output-root "$RESULT_DIR" --output "$RUN/results/benchmark" \
        --plan "$RUN/results/model-plan.json" --device "$CUDA_VISIBLE_DEVICES" --order abba --require-selected \
        --repeats "${MODEL_REPEATS:-2}" 2>&1 | tee "$RUN/results/benchmark.log"; then :; else failed=$((failed+1)); fi
    stage=model-trace
    if "$PYTHON" -u tools/run_kpack_model_validation.py "${COMMON[@]}" --phase trace \
        --output "$RUN/results/trace" 2>&1 | tee "$RUN/results/trace.log"; then :; else failed=$((failed+1)); fi
    if [[ "$MODEL_GATE_UP" == 1 ]]; then
        stage=paired-model-proof
        if "$PYTHON" tools/check_kpack_paired_model.py --results "$RUN/results"; then :; else failed=$((failed+1)); fi
    fi
    if [[ "$MODEL_ACU" == 1 ]]; then
        stage=model-acu
        if "$PYTHON" -u tools/profile_kpack_model_decode.py --sdk "$SDK" --bundle "$BUNDLE" \
            --llama "$LLAMA_DIR" --trace "$RUN/results/trace" --acu "$ACU" \
            --output "$RUN/results/acu" 2>&1 | tee "$RUN/results/acu.log"; then :; else failed=$((failed+1)); fi
    fi
    [[ $failed == 0 ]]
    stage=complete
    printf 'KPACK_Q4_MODEL COMPLETE phases=%s accuracy_admission=PENDING performance_vs_native=SEE_SUMMARY\n' "$MODEL_PHASES"
)
