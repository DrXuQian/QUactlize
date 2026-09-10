#!/usr/bin/env bash
# Compare warmed native/K-pack traces using one previous run's exact request.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run with bash, not source.\n' >&2
    return 1
fi
if [[ ${1:-} == --help ]]; then
    printf '%s\n' \
        'Usage: bash tools/run_kpack_reference_trace.sh /workspace/kpack-fusion.XXXXXX' \
        'Reuses the prior model, binary, token IDs, cache and prompt/generation lengths.' \
        'Captures reference then K-pack in separate processes; first request is warmup.' \
        'Set TRACE_ARM=native or reference to rerun only one arm into a fresh directory.' \
        'No build or config sweep. Original native fusions remain enabled.' \
        'Outputs reference/proof.asysrep, native/proof.asysrep and kernel-times.json.' \
        'Profiler timings diagnose overhead; they do not admit end-to-end performance.'
    exit 0
fi
set -Ee -o pipefail
RUN= stage=precheck
finish() {
    local rc=$?
    trap - EXIT ERR
    if [[ -n $RUN && -d $RUN/results ]]; then
        printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
        tar --exclude='*.asysrep' --exclude='*.sqlite*' -czf "$RUN.results.tgz" -C "$RUN" results
        printf '\nresults=%s.results.tgz\n' "$RUN"
    fi
    printf 'runner_rc=%s stage=%s (calling shell preserved)\n' "$rc" "$stage"
}
trap finish EXIT
trap 'printf "FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
[[ $# == 1 ]]
case ${TRACE_ARM:-both} in
    both) ARMS=(reference native) ;;
    reference|native) ARMS=("$TRACE_ARM") ;;
    *) printf 'TRACE_ARM must be both, reference or native.\n' >&2; false ;;
esac
PRIOR=$(realpath -e -- "$1")
[[ -d $PRIOR ]]
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
LLAMA_DIR=${LLAMA_DIR:-$(dirname -- "$REPO")/llama.cpp}
PPU_SDK=${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}
PYTHON=$(command -v -- "${PYTHON:-python3}")
INPUT="$PRIOR/results/model-proof/proof-request/input-tokens.json"
INVENTORY="$PRIOR/results/model-trace-inventory.json"
PROTOCOL="$PRIOR/results/model-proof/protocol.json"
COMMAND="$PRIOR/results/model-proof/proof-request/0-native.command.json"
[[ -s $INPUT && -s $INVENTORY && -s $PROTOCOL && -s $COMMAND ]]
SETTINGS=$("$PYTHON" - "$COMMAND" "$PROTOCOL" <<'PY'
import json, sys
command = json.load(open(sys.argv[1]))
protocol = json.load(open(sys.argv[2]))
position = command.index('-m')
values = [command[position-1], command[position+1], command[command.index('--kpack-cache')+1],
          str(protocol['proof_parameters']['prompts'][0]),
          str(protocol['proof_parameters']['generate']), protocol['jit_cache']]
assert all(isinstance(v, str) and v and '\n' not in v for v in values)
print('\n'.join(values))
PY
)
mapfile -t fields <<< "$SETTINGS"
[[ ${#fields[@]} == 6 ]]
BINARY=${fields[0]} MODEL=${fields[1]} CACHE=${fields[2]} PROMPT=${fields[3]} GENERATE=${fields[4]}
export QUACTLIZE_KPACK_JIT_CACHE=${fields[5]}
[[ -x $BINARY && -s $MODEL && -d $CACHE && -d $QUACTLIZE_KPACK_JIT_CACHE ]]
[[ $PROMPT =~ ^[1-9][0-9]*$ && $GENERATE =~ ^[1-9][0-9]*$ && -s $PPU_SDK/envsetup.sh ]]
HELP=$("$PYTHON" "$LLAMA_DIR/tests/quactlize_native.py" --help)
if [[ $HELP != *--proof-arm* || $HELP != *--proof-tokens* ]]; then
    printf 'Update llama.cpp feat/kpack-gpu-cache: matched reference trace options are required.\n' >&2
    false
fi
source "$PPU_SDK/envsetup.sh"
set -Ee -o pipefail
export PPU_SDK LC_ALL=C CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[[ $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]
[[ -x $PPU_SDK/asight/bin/asys && -x $PPU_SDK/bin/hgobjdump ]]
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:$PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QUACTLIZE_KPACK_EXECUTION=${QUACTLIZE_KPACK_EXECUTION:-$REPO/prebuilt/ppu0010/kpack-fusion-v3/dispatch}
export QUACTLIZE_PPU_PACK_LIBRARY=${QUACTLIZE_PPU_PACK_LIBRARY:-$REPO/prebuilt/ppu0010/kpack-fusion-v1/libquactlize_ppu_pack.so}
export QUACTLIZE_PPU_BUNDLE=${QUACTLIZE_PPU_BUNDLE:-/workspace/quactlize-runtime-artifact-2826cf1-46fc3096e1a1/prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1/bundle}
export QUACTLIZE_KPACK_JIT_HELPER="$REPO/tools/kpack_jit.py" QUACTLIZE_KPACK_JIT_PYTHON="$PYTHON"
export QUACTLIZE_KPACK_ROUTE=auto QUACTLIZE_KPACK_PAIR_WEIGHTS=1
unset QUACTLIZE_KPACK_PREFILL_POLICY QUACTLIZE_KPACK_GEMV_POLICY GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_DISABLE_FUSION
if [[ -s $PRIOR/results/q8_simt/gemv-policy.tsv ]]; then
    export QUACTLIZE_KPACK_GEMV_POLICY="$PRIOR/results/q8_simt/gemv-policy.tsv"
fi
"$PYTHON" "$REPO/tools/verify_kpack_dispatch.py" "$QUACTLIZE_KPACK_EXECUTION"
[[ -s $QUACTLIZE_PPU_BUNDLE/manifest.json && -s $QUACTLIZE_PPU_PACK_LIBRARY ]]
RESULT_ROOT=${RESULT_ROOT:-/workspace}
[[ -d $RESULT_ROOT ]]
RUN=$(mktemp -d "$RESULT_ROOT/kpack-reference-ab.XXXXXX")
[[ -d $RUN ]]
mkdir "$RUN/results"
printf 'KPACK_REFERENCE_TRACE run=%s prior=%s prompt=%s generate=%s\n' "$RUN" "$PRIOR" "$PROMPT" "$GENERATE"
git -C "$REPO" rev-parse HEAD > "$RUN/results/quactlize-source.txt"
git -C "$LLAMA_DIR" rev-parse HEAD > "$RUN/results/llama-source.txt"
failed=0
for arm in "${ARMS[@]}"; do
    stage="trace-$arm"
    if "$PYTHON" -u "$LLAMA_DIR/tests/quactlize_native.py" --binary "$BINARY" \
        --model "$MODEL" --cache "$CACHE" --bundle "$QUACTLIZE_KPACK_EXECUTION" \
        --asys "$PPU_SDK/asight/bin/asys" --inspector "$PPU_SDK/bin/hgobjdump" \
        --jit-cache "$QUACTLIZE_KPACK_JIT_CACHE" --jit-helper "$QUACTLIZE_KPACK_JIT_HELPER" \
        --jit-python "$PYTHON" --output "$RUN/results/$arm" --proof-only --proof-arm "$arm" \
        --proof-tokens "$INPUT" --tensor-inventory "$INVENTORY" \
        --proof-prompt "$PROMPT" --proof-generate "$GENERATE" 2>&1 | tee "$RUN/results/$arm.log"; then
        printf 'KPACK_REFERENCE_TRACE arm=%s PASS report=%s/results/%s/proof.asysrep\n' "$arm" "$RUN" "$arm"
    else
        failed=$((failed+1))
        printf 'KPACK_REFERENCE_TRACE arm=%s FAIL remaining_continue=1\n' "$arm"
    fi
done
[[ $failed == 0 ]]
stage=compare
"$PYTHON" - "$RUN/results" "${ARMS[@]}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
results = {arm: json.loads((root/arm/'proof.json').read_text()) for arm in sys.argv[2:]}
summary = dict(status='TRACE_ARM_COMPLETE', pair_comparison='NOT_RUN',
               performance_admission='NOT_MEASURED_BY_PROFILER',
               request_sha256=next(iter(results.values()))['request_sha256'],
               routes={arm: data['kernel_execution'] for arm, data in results.items()})
if set(results) == {'reference', 'native'}:
    for key in ('input_tokens_sha256', 'request_sha256', 'prompt_tokens', 'generated_tokens', 'prefill_token_batch'):
        assert results['reference'][key] == results['native'][key], f'paired trace differs: {key}'
    props = {arm: json.loads((root/arm/'proof-request'/f'0-{arm}.props.json').read_text()) for arm in results}
    for key in ('model_path', 'build_info'):
        assert key in props['reference'] and props['reference'][key] == props['native'][key], f'server identity differs: {key}'
    summary.update(status='TRACE_PAIR_COMPLETE', pair_comparison='PASS',
                   same_generated_text=results['reference']['response_sha256'] == results['native']['response_sha256'])
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print('KPACK_REFERENCE_TRACE '+json.dumps(summary))
PY
stage=done
