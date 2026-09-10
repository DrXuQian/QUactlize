#!/usr/bin/env bash
# Bounded device gates, incremental llama adapter build, warmed model benchmark.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run with bash, not source.\n' >&2
    return 1
fi
if [[ ${1:-} == --help ]]; then
    printf '%s\n' \
        'Defaults: LLAMA_DIR=<sibling llama.cpp>, BUILD_DIR=$LLAMA_DIR/build-kpack-9f86a1340' \
        'PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK CUDA_VISIBLE_DEVICES=0 JOBS=192' \
        'RUN_MODEL_BENCH=1 MODEL_NAMES=qwen35-35b-q4km (all for the uploaded model list)' \
        'MODEL_ROOT=/sim/eec/shared/AI_workspace/llm-models (one root, no fallback directory)' \
        'MODEL_PLAN may override tools/kpack_batched_models.json; file paths resolve before device gates.' \
        'All timings omit the first whole pass; benchmark and trace use the same resolved model plan.' \
        'The old six-library bundle remains intake/fallback only; QUACTLIZE_PPU_BUNDLE can override its path.' \
        'No Cartesian sweep or large Quactlize bundle rebuild. JIT compiles selected missing parents only.' \
        'Gate failures are collected independently; failed numerics prevent model timing admission.' \
        'Optional RUN_MODEL_TRACE=1 captures Asys only (no second ABBA benchmark).' \
        'RUN_Q8_SIMT=1 compares eight SIMT choices on twelve small dense contexts and exports only measured winners.' \
        'TRACE_PROMPT=128 TRACE_GENERATE=8; set TRACE_PROMPT=2048 for the focused int4 plan.'
    exit 0
fi
set -Ee -o pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
LLAMA_DIR=${LLAMA_DIR:-$(dirname -- "$REPO")/llama.cpp}
BUILD_DIR=${BUILD_DIR:-$LLAMA_DIR/build-kpack-9f86a1340}
RESULT_ROOT=${RESULT_ROOT:-/workspace}
PPU_SDK=${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}
PYTHON=${PYTHON:-python3}
JOBS=${JOBS:-192}
RUN= stage=precheck
finish() {
    local rc=$?
    trap - EXIT ERR
    if [[ -n $RUN && -d $RUN/results ]]; then
        printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
        tar --exclude='*.asysrep' --exclude='*.sqlite*' --exclude='*.tgz' \
            -czf "$RUN.results.tgz" -C "$RUN" results && printf '\nresults=%s.results.tgz\n' "$RUN"
    fi
    printf 'runner_rc=%s stage=%s (calling shell preserved)\n' "$rc" "$stage"
}
trap finish EXIT
trap 'printf "FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
[[ $# == 0 && $JOBS =~ ^[1-9][0-9]*$ && -d $RESULT_ROOT ]]
[[ -s $BUILD_DIR/CMakeCache.txt && -s $LLAMA_DIR/ggml/src/ggml-cuda/quactlize/kpack_indexed.h ]]
if [[ ! -s $LLAMA_DIR/tests/test-kpack-model-loader.cpp ]]; then
    printf 'Update llama.cpp feat/kpack-gpu-cache: paired-weight loader regression is required (bd4e8bf93).\n' >&2
    false
fi
[[ -s $PPU_SDK/envsetup.sh ]]
source "$PPU_SDK/envsetup.sh"
set -Ee -o pipefail
export PPU_SDK LC_ALL=C CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[[ $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:$PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON=$(command -v -- "$PYTHON")
"$PYTHON" -c 'import sys,numpy,torch,gguf; assert sys.version_info >= (3,11)'
export QUACTLIZE_PPU_PACK_LIBRARY=${QUACTLIZE_PPU_PACK_LIBRARY:-$REPO/prebuilt/ppu0010/kpack-fusion-v1/libquactlize_ppu_pack.so}
export QUACTLIZE_KPACK_EXECUTION=${QUACTLIZE_KPACK_EXECUTION:-$REPO/prebuilt/ppu0010/kpack-fusion-v2/dispatch}
export QUACTLIZE_PPU_BUNDLE=${QUACTLIZE_PPU_BUNDLE:-/workspace/quactlize-runtime-artifact-2826cf1-46fc3096e1a1/prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1/bundle}
[[ -s $QUACTLIZE_PPU_BUNDLE/manifest.json && -s $QUACTLIZE_PPU_PACK_LIBRARY ]]
grep -qx 'GGML_USE_PPU:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"
grep -qx 'GGML_NCP_QUACTLIZE:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"
RUN=$(mktemp -d "$RESULT_ROOT/kpack-fusion.XXXXXX")
[[ -d $RUN ]]
mkdir "$RUN/results"
printf 'KPACK_FUSION_RUN run=%s\n' "$RUN"
git -C "$REPO" rev-parse HEAD > "$RUN/results/quactlize-source.txt"
git -C "$LLAMA_DIR" rev-parse HEAD > "$RUN/results/llama-source.txt"
export QUACTLIZE_KPACK_JIT_CACHE=${JIT_CACHE:-$RESULT_ROOT/kpack-fusion-jit-cache}
mkdir -p -- "$QUACTLIZE_KPACK_JIT_CACHE"
export QUACTLIZE_KPACK_JIT_HELPER="$REPO/tools/kpack_jit.py"
export QUACTLIZE_KPACK_JIT_PYTHON="$PYTHON"
export QUACTLIZE_KPACK_ROUTE=auto QUACTLIZE_KPACK_PAIR_WEIGHTS=1
unset QUACTLIZE_KPACK_PREFILL_POLICY QUACTLIZE_KPACK_GEMV_POLICY GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_DISABLE_FUSION
"$PYTHON" "$REPO/tools/verify_kpack_dispatch.py" "$QUACTLIZE_KPACK_EXECUTION" | tee "$RUN/results/verify.log"
(cd "$LLAMA_DIR/ggml/src/ggml-cuda/quactlize" && sha256sum -c ABI_SHA256) | tee "$RUN/results/abi.log"
MODEL_ARGS=()
if [[ ${MODEL_NAMES:-qwen35-35b-q4km} != all ]]; then
    read -r -a names <<< "${MODEL_NAMES:-qwen35-35b-q4km}"
    [[ ${#names[@]} -gt 0 ]]
    for name in "${names[@]}"; do MODEL_ARGS+=(--model "$name"); done
fi
if [[ ${RUN_MODEL_BENCH:-1} == 1 || ${RUN_MODEL_TRACE:-0} == 1 ]]; then
    stage=model-paths
    if [[ ${RUN_MODEL_TRACE:-0} == 1 ]]; then
        "$PYTHON" "$LLAMA_DIR/tests/quactlize_native.py" --help > "$RUN/results/model-trace-help.txt"
        if ! grep -q -- '--proof-only' "$RUN/results/model-trace-help.txt" || \
           ! grep -q -- '--tensor-inventory' "$RUN/results/model-trace-help.txt"; then
            printf 'Update llama.cpp feat/kpack-gpu-cache: the Asys-only runner is required.\n' >&2
            false
        fi
    fi
    RESOLVE_ARGS=()
    if [[ ${RUN_MODEL_BENCH:-1} == 1 ]]; then RESOLVE_ARGS+=("${MODEL_ARGS[@]}"); fi
    if [[ ${RUN_MODEL_TRACE:-0} == 1 ]]; then
        MODEL_NAME=${TRACE_MODEL_NAME:-qwen35-35b-q4km}
        if [[ ${RUN_MODEL_BENCH:-1} != 1 || ${MODEL_NAMES:-qwen35-35b-q4km} != all ]]; then
            RESOLVE_ARGS+=(--model "$MODEL_NAME")
        fi
    fi
    if [[ -n ${MODEL_ROOT:-} ]]; then RESOLVE_ARGS+=(--model-root "$MODEL_ROOT"); fi
    "$PYTHON" "$REPO/tools/resolve_kpack_batched_models.py" \
        --plan "${MODEL_PLAN:-$REPO/tools/kpack_batched_models.json}" "${RESOLVE_ARGS[@]}" \
        --output "$RUN/results/model-plan.json" | tee "$RUN/results/model-paths.log"
    if [[ ${RUN_MODEL_TRACE:-0} == 1 ]]; then
        MODEL=$("$PYTHON" -c 'import json,sys; m=next(x for x in json.load(open(sys.argv[1]))["models"] if x["name"]==sys.argv[2]); assert m["split"]=="none", "K-pack tensor-parallel trace is not admitted"; print(m["path"])' \
            "$RUN/results/model-plan.json" "$MODEL_NAME")
        (cd "$REPO" && "$PYTHON" -c 'import json,sys; from pathlib import Path; from tools.run_kpack_batched_bench import inventory; Path(sys.argv[2]).write_text(json.dumps(inventory(Path(sys.argv[1])), indent=2)+"\n")' \
            "$MODEL" "$RUN/results/model-trace-inventory.json")
    fi
fi
stage=device-gates
failed=0
GATES=(q8_kpack2 kpack_moe)
if [[ ${RUN_Q8_SIMT:-1} == 1 ]]; then GATES+=(q8_simt); fi
for item in "${GATES[@]}"; do
    printf 'KPACK_FUSION_PHASE gate=%s (selected-parent JIT, then numerical/replay/timing)\n' "$item"
    GATE_ARGS=()
    if [[ $item != q8_simt ]]; then GATE_ARGS+=(--pack-library "$QUACTLIZE_PPU_PACK_LIBRARY"); fi
    if OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PYTHON" -u "$REPO/tools/run_${item}_gate.py" \
        --sdk "$PPU_SDK" --bundle "$QUACTLIZE_KPACK_EXECUTION" \
        "${GATE_ARGS[@]}" --jit-cache "$QUACTLIZE_KPACK_JIT_CACHE" \
        --output "$RUN/results/$item" --samples 11 2>&1 | tee "$RUN/results/$item.log"; then
        printf 'KPACK_FUSION_GATE gate=%s status=PASS\n' "$item"
    else
        failed=$((failed+1)); printf 'KPACK_FUSION_GATE gate=%s status=FAIL remaining_continue=1\n' "$item"
    fi
done
[[ $failed == 0 ]]
if [[ ${RUN_Q8_SIMT:-1} == 1 ]]; then
    export QUACTLIZE_KPACK_GEMV_POLICY="$RUN/results/q8_simt/gemv-policy.tsv"
    [[ -s $QUACTLIZE_KPACK_GEMV_POLICY ]]
    printf 'KPACK_FUSION_POLICY Q8 measured SIMT winners loaded; misses retain selected W8A16 TC\n'
fi
stage=adapter-build
printf 'KPACK_FUSION_PHASE adapter-build jobs=%s no_quactlize_sweep=1\n' "$JOBS"
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_NCP_FA=OFF -DGGML_NCP_MOE=OFF -DGGML_NCP_GDN=OFF \
    2>&1 | tee "$RUN/results/configure.log"
cmake --build "$BUILD_DIR" --target llama-server llama-batched-bench \
    test-quactlize-execution test-quactlize-buffer test-quactlize-loader test-kpack-sidecar test-kpack-model-loader -j "$JOBS" \
    2>&1 | tee "$RUN/results/build.log"
stage=adapter-contract
env -u QUACTLIZE_KPACK_EXECUTION -u QUACTLIZE_KPACK_JIT_HELPER -u QUACTLIZE_KPACK_JIT_PYTHON \
    -u QUACTLIZE_KPACK_JIT_CACHE -u QUACTLIZE_KPACK_PAIR_WEIGHTS \
    ctest --test-dir "$BUILD_DIR" --output-on-failure \
    -R '^(test-quactlize-(execution-(auto|fq|sf|gemv)|buffer|loader(-env)?)|test-kpack-(sidecar|model-loader))$' \
    2>&1 | tee "$RUN/results/adapter-tests.log"
if [[ ${RUN_MODEL_BENCH:-1} == 1 ]]; then
    stage=batched-model
    printf 'KPACK_FUSION_PHASE benchmark first_whole_pass_excluded=1 reference+kpack-cold+kpack-hot\n'
    "$PYTHON" -u "$REPO/tools/run_kpack_batched_bench.py" \
        --binary "$BUILD_DIR/bin/llama-batched-bench" --llama-dir "$LLAMA_DIR" \
        --bundle "$QUACTLIZE_KPACK_EXECUTION" --jit-cache "$QUACTLIZE_KPACK_JIT_CACHE" \
        --plan "$RUN/results/model-plan.json" "${MODEL_ARGS[@]}" \
        --device "$CUDA_VISIBLE_DEVICES" \
        --cache-root "$RUN/cache" --output-root "$RESULT_ROOT" --output "$RUN/results/benchmark" \
        --repeats "${MODEL_REPEATS:-1}" 2>&1 | tee "$RUN/results/benchmark.log"
fi
if [[ ${RUN_MODEL_TRACE:-0} == 1 ]]; then
    stage=warm-model-trace
    "$PYTHON" -u "$LLAMA_DIR/tests/quactlize_native.py" --binary "$BUILD_DIR/bin/llama-server" \
        --model "$MODEL" --cache "$RUN/cache/$MODEL_NAME" --bundle "$QUACTLIZE_KPACK_EXECUTION" \
        --asys "$PPU_SDK/asight/bin/asys" --inspector "$PPU_SDK/bin/hgobjdump" \
        --jit-cache "$QUACTLIZE_KPACK_JIT_CACHE" --jit-helper "$QUACTLIZE_KPACK_JIT_HELPER" \
        --jit-python "$PYTHON" --output "$RUN/results/model-proof" --proof-only \
        --tensor-inventory "$RUN/results/model-trace-inventory.json" \
        --proof-prompt "${TRACE_PROMPT:-128}" --proof-generate "${TRACE_GENERATE:-8}" \
        2>&1 | tee "$RUN/results/model-proof.log"
fi
stage=done
printf 'KPACK_FUSION_COMPLETE results=%s/results\n' "$RUN"
