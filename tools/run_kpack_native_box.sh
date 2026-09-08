#!/usr/bin/env bash
# Execute locally compiled Quactlize modules, then build/test their llama adapter.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run with bash, not source.\n' >&2
    return 1
fi
if [[ ${1:-} == --help ]]; then
    printf '%s\n' \
      'Required: LLAMA_DIR MODEL PPU_SDK QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY CACHE_DIR BUILD_DIR' \
      'Optional: JOBS=192 CUDA_VISIBLE_DEVICES=0 RESULT_ROOT=/workspace PYTHON=python3' \
      'Prebuilt native gate -> GEMV comparison/export -> adapter build/tests -> real-model ABBA -> short trace.' \
      'No Quactlize device compilation. Existing llama PPU build is rebuilt for the changed context ABI.' \
      'Only Q2_K..Q6_K weights are supported; this model test includes experts and Q6 output.weight.'
    exit 0
fi
set -Ee -o pipefail
stage=precheck
RUN=
finish() {
    local rc=$?
    trap - ERR EXIT
    if [[ -n $RUN && -d $RUN/results ]]; then
        printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
        # Large raw profiler containers stay on the box; the JSON execution
        # receipt includes matched symbols, durations and selected build IDs.
        if tar --exclude='*.asysrep' --exclude='*.sqlite' --exclude='*.sqlite-*' \
             -czf "$RUN.results.tgz" -C "$RUN" results; then
            printf '\nresults=%s.results.tgz\n' "$RUN"
        fi
    fi
    printf 'runner_rc=%s stage=%s\n' "$rc" "$stage"
}
trap 'printf "FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
trap finish EXIT
[[ $# == 0 ]]
for key in LLAMA_DIR MODEL PPU_SDK QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY CACHE_DIR BUILD_DIR; do
    [[ -n ${!key:-} ]] || { printf 'MISSING %s\n' "$key" >&2; false; }
done
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ -n $REPO && -f $REPO/tools/verify_kpack_dispatch.py ]]
LLAMA_DIR=$(realpath -e -- "$LLAMA_DIR")
BUILD_DIR=$(realpath -e -- "$BUILD_DIR")
[[ -n $LLAMA_DIR && -n $BUILD_DIR && -s $BUILD_DIR/CMakeCache.txt ]]
[[ -r $MODEL && -s $QUACTLIZE_PPU_BUNDLE/manifest.json && -s $QUACTLIZE_PPU_PACK_LIBRARY ]]
[[ -d $CACHE_DIR && -s $CACHE_DIR/manifest.json && -f $LLAMA_DIR/tests/quactlize_native.py ]]
JOBS=${JOBS:-192}
PYTHON=${PYTHON:-python3}
RESULT_ROOT=${RESULT_ROOT:-/workspace}
[[ $JOBS =~ ^[1-9][0-9]*$ && -d $RESULT_ROOT ]]
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[[ $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]
source "$PPU_SDK/envsetup.sh"
set -Ee -o pipefail
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:$PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QUACTLIZE_PPU_BUNDLE QUACTLIZE_PPU_PACK_LIBRARY LC_ALL=C
export QUACTLIZE_KPACK_EXECUTION="$REPO/prebuilt/ppu0010/kpack-native-v1"
export QUACTLIZE_KPACK_ROUTE=auto
ASYS=${ASYS:-$PPU_SDK/asight/bin/asys}
[[ -x $ASYS && -x $PPU_SDK/bin/hgobjdump ]]
for tool in "$PYTHON" cmake ctest c++filt; do command -v "$tool" >/dev/null; done
"$PYTHON" -c 'import numpy, gguf'
grep -qx 'GGML_USE_PPU:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"
grep -qx 'GGML_NCP_QUACTLIZE:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"
RUN=$(mktemp -d "$RESULT_ROOT/kpack-native-model.XXXXXX")
[[ -n $RUN && -d $RUN ]]
mkdir "$RUN/results"
printf 'RUN=%s\n' "$RUN"
git -C "$REPO" rev-parse HEAD > "$RUN/results/quactlize-source.txt"
git -C "$LLAMA_DIR" rev-parse HEAD > "$RUN/results/llama-source.txt"
git -C "$REPO" diff --binary -- quactlize policies tools reference tests > "$RUN/results/quactlize-dirty.patch"
git -C "$LLAMA_DIR" diff --binary -- ggml tests common tools > "$RUN/results/llama-dirty.patch"
cp "$QUACTLIZE_KPACK_EXECUTION/manifest.json" "$RUN/results/native-manifest.json"
cp "$QUACTLIZE_PPU_BUNDLE/manifest.json" "$RUN/results/legacy-manifest.json"
cp "$CACHE_DIR/manifest.json" "$RUN/results/cache-before.json"
"$PYTHON" "$REPO/tools/verify_kpack_dispatch.py" "$QUACTLIZE_KPACK_EXECUTION" \
    | tee "$RUN/results/verify.log"
stage=native-selected-device-gate
"$PYTHON" -u "$REPO/tools/run_kpack_native_gate.py" --sdk "$PPU_SDK" \
    --bundle "$QUACTLIZE_KPACK_EXECUTION" --output "$RUN/results/native-gate" \
    2>&1 | tee "$RUN/results/native-gate.log"
export QUACTLIZE_KPACK_PREFILL_POLICY="$RUN/results/prefill-policy.tsv"
"$PYTHON" "$REPO/tools/export_kpack_prefill_policy.py" --results "$RUN/results/native-gate/summary.json" \
    --native-bundle "$QUACTLIZE_KPACK_EXECUTION" --output "$QUACTLIZE_KPACK_PREFILL_POLICY" \
    | tee "$RUN/results/prefill-policy.log"
stage=gemv-selection
"$PYTHON" -u "$REPO/tools/run_kpack_gemv_gate.py" --sdk "$PPU_SDK" \
    --bundle "$REPO/prebuilt/ppu0010/kpack-execution-v1" --gemm-bundle "$QUACTLIZE_PPU_BUNDLE" \
    --native-bundle "$QUACTLIZE_KPACK_EXECUTION" --model-decode-only --samples 11 \
    --output "$RUN/results/gemv-gate" 2>&1 | tee "$RUN/results/gemv-gate.log"
export QUACTLIZE_KPACK_GEMV_POLICY="$RUN/results/gemv-policy.tsv"
"$PYTHON" "$REPO/tools/export_kpack_gemv_policy.py" --results "$RUN/results/gemv-gate/summary.json" \
    --native-bundle "$QUACTLIZE_KPACK_EXECUTION" \
    --execution-library "$QUACTLIZE_KPACK_EXECUTION/libquactlize_ppu_execution.so" \
    --output "$QUACTLIZE_KPACK_GEMV_POLICY" | tee "$RUN/results/gemv-policy.log"
stage=build-llama-adapter
printf 'KPACK_NATIVE_BUILD jobs=%s target=llama-adapter quactlize_dso_rebuild=0\n' "$JOBS"
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_NCP_FA=OFF -DGGML_NCP_MOE=OFF -DGGML_NCP_GDN=OFF \
    2>&1 | tee "$RUN/results/configure.log"
cmake --build "$BUILD_DIR" --target llama-server llama-completion test-quactlize-execution \
    test-quactlize-buffer test-quactlize-loader test-kpack-sidecar -j "$JOBS" \
    2>&1 | tee "$RUN/results/build.log"
sha256sum "$BUILD_DIR/bin/llama-server" "$BUILD_DIR/bin/llama-completion" "$BUILD_DIR/bin/libggml-cuda.so" \
    "$BUILD_DIR/bin/libllama.so" "$QUACTLIZE_PPU_PACK_LIBRARY" > "$RUN/results/binaries.sha256"
for fmt in 0 1 2 3 4; do
    sha256sum "$QUACTLIZE_PPU_BUNDLE/libquactlize_ppu_fmt$fmt.so" >> "$RUN/results/binaries.sha256"
done
grep -E '^GGML_(USE_PPU|NCP_|CUDA_GRAPH)' "$BUILD_DIR/CMakeCache.txt" > "$RUN/results/build-options.txt"
stage=adapter-contract
"$PYTHON" "$LLAMA_DIR/tests/test-quactlize-native.py" 2>&1 | tee "$RUN/results/parser-tests.log"
ctest --test-dir "$BUILD_DIR" --output-on-failure \
    -R '^(test-quactlize-(execution-(fq|sf|gemv)|buffer|loader)|test-kpack-sidecar)$' \
    2>&1 | tee "$RUN/results/adapter-tests.log"
stage=real-model
"$PYTHON" -u "$LLAMA_DIR/tests/quactlize_native.py" --binary "$BUILD_DIR/bin/llama-server" \
    --proof-binary "$BUILD_DIR/bin/llama-completion" --model "$MODEL" --cache "$CACHE_DIR" \
    --bundle "$QUACTLIZE_KPACK_EXECUTION" --asys "$ASYS" --inspector "$PPU_SDK/bin/hgobjdump" \
    --output "$RUN/results/model" 2>&1 | tee "$RUN/results/model.log"
stage=done
printf 'KPACK_NATIVE_MODEL COMPLETE results=%s/results\n' "$RUN"
