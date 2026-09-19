#!/usr/bin/env bash
# Prebuilt components only. Do not install these test DSOs as a model runtime.
(
    set -Eeuo pipefail
    RUN= stage=precheck
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n $RUN && -d $RUN/results ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
            tar --exclude='*.acurep' -czf "$RUN.results.tgz" -C "$RUN" results || rc=1
            printf '\nresults=%s.results.tgz\nACU reports: %s/results/sweep/*.acurep\n' "$RUN" "$RUN"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "DECODE_FALLBACK FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/dev/tp2_decode/fallback_prebuilt.json"
    cd "$ROOT"
    PYTHON=$(command -v "${PYTHON:-python3}")
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -r "$SDK/envsetup.sh"
    DEVICE=${CUDA_VISIBLE_DEVICES:-0}
    [[ "$DEVICE" =~ ^[0-9]+$ ]]
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK" CUDA_VISIBLE_DEVICES="$DEVICE"
    export PATH="$SDK/asight/bin:$SDK/bin:$PATH" ASIGHT_HOME="$SDK/asight"
    export LD_LIBRARY_PATH="$SDK/lib:$SDK/targets/x86_64-linux/lib:$SDK/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$PYTHON" -c 'import numpy, torch, gguf'
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    RESUME=()
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -f "$RUN/results/sweep/identity.json"
        RESUME=(--resume)
    else
        RUN=$(mktemp -d "$RESULT_DIR/decode-fallback.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    printf 'DECODE_FALLBACK run=%s device=%s prebuilt_only=1 points=18 production_replacement=0\n' "$RUN" "$DEVICE"
    git rev-parse HEAD > "$RUN/results/source.txt"
    cp dev/tp2_decode/fallback_prebuilt.json "$RUN/results/prebuilt.json"
    mapfile -t PIN < <("$PYTHON" -c 'import json; p=json.load(open("dev/tp2_decode/fallback_prebuilt.json")); print(p["artifact_branch"]); print(p["artifact_commit"]); print(p["gemv_path"]); print(p["prepare_path"])')
    [[ ${#PIN[@]} == 4 && ${PIN[0]} == artifacts/decode-fallback-v1 && ${PIN[1]} =~ ^[0-9a-f]{40}$ ]]
    [[ ${PIN[2]} == prebuilt/ppu0010/decode-fallback-v1 && ${PIN[3]} == prebuilt/ppu0010/decode-fallback-prepare-v1 ]]
    stage=fetch
    ART=${FALLBACK_ARTIFACT_DIR:-$RUN/artifact}
    if [[ -z ${FALLBACK_ARTIFACT_DIR:-} ]]; then
        if [[ ! -e $ART ]]; then
            GIT_LFS_SKIP_SMUDGE=1 git fetch origin "${PIN[0]}"
            git cat-file -e "${PIN[1]}^{commit}"
            GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$ART" "${PIN[1]}"
        fi
        [[ $(git -C "$ART" rev-parse HEAD) == "${PIN[1]}" ]]
        git -C "$ART" lfs pull --include="${PIN[2]}/*,${PIN[3]}/*" --exclude=''
    fi
    ART=$(realpath -e -- "$ART")
    test -n "$ART" && [[ $(git -C "$ART" rev-parse HEAD) == "${PIN[1]}" ]]
    GEMV="$ART/${PIN[2]}" PREPARE="$ART/${PIN[3]}"
    stage=verify
    "$PYTHON" - "$ART" "$SDK" "$RUN/results" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path.cwd();art,sdk,results=map(pathlib.Path,sys.argv[1:])
pin=json.loads((root/'dev/tp2_decode/fallback_prebuilt.json').read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for key in ('gemv','prepare'):
    if sha(art/pin[key+'_path']/'manifest.json')!=pin[key+'_manifest_sha256']:
        raise ValueError(key+' manifest differs from pinned handoff')
prepare=art/pin['prepare_path'];m=json.loads((prepare/'manifest.json').read_text())
if not m.get('production_candidate') or sha(prepare/'bench')!=m['binary_sha256']:
    raise ValueError('prepare is not the exact production-dispatch gate')
for name,h in m['source_hashes'].items():
    if sha(root/name)!=h:raise ValueError('prepare source differs: '+name)
gemv=json.loads((art/pin['gemv_path']/'manifest.json').read_text())
runtime={n:sha(sdk/'lib'/n) for n in gemv['runtime']}
(results/'environment.json').write_text(json.dumps(dict(sdk=str(sdk),runtime=runtime,
    runtime_matches_build=runtime==gemv['runtime'],production_replacement=False),indent=2)+'\n')
if runtime!=gemv['runtime']:raise ValueError('SDK runtime differs; see environment.json')
print('DECODE_FALLBACK_PREFLIGHT PASS source_and_binary=bound runtime=matched')
PY
    "$PYTHON" dev/tp2_decode/run_fallback.py --sdk "$SDK" --bundle "$GEMV" \
        --output "$RUN/results/sweep" --verify-only | tee "$RUN/results/verify.log"
    cp "$GEMV/manifest.json" "$RUN/results/gemv-manifest.json"
    cp "$GEMV/native-inspection.json" "$RUN/results/native-inspection.json"
    cp "$PREPARE/manifest.json" "$RUN/results/prepare-manifest.json"
    if [[ ${VERIFY_ONLY:-0} == 1 ]]; then stage=verified-only; exit 0; fi
    failures=0 prepare_failed=0
    stage=prepare-numerics
    for gate in shape-check router-edge-check router-alias-check; do
        printf 'DECODE_FALLBACK_PREPARE gate=%s\n' "$gate"
        if "$PREPARE/bench" "--$gate" 2>&1 | tee "$RUN/results/prepare-$gate.log"; then
            printf '%s\tPASS\n' "$gate" >> "$RUN/results/prepare-status.tsv"
        else
            failures=$((failures+1)); prepare_failed=1
            printf '%s\tFAIL\n' "$gate" >> "$RUN/results/prepare-status.tsv"
        fi
    done
    if [[ $prepare_failed == 0 ]]; then
        stage=prepare-timing
        for tokens in 1 8; do
            if "$PREPARE/bench" --case "$tokens" 3072 5 1 0 1 0 2>&1 | tee "$RUN/results/prepare-m$tokens.log"; then
                printf 'm%s\tPASS\n' "$tokens" >> "$RUN/results/prepare-status.tsv"
            else failures=$((failures+1)); fi
        done
    fi
    EXTRA=()
    if [[ -n ${GEMV_POINTS:-} ]]; then EXTRA+=(--points "$GEMV_POINTS"); fi
    if [[ ${PROFILE:-1} == 1 ]]; then
        ACU=$(realpath -e -- "${ACU:-$SDK/asight/bin/acu}")
        test -n "$ACU" && test -x "$ACU"
        EXTRA+=(--acu "$ACU")
    fi
    stage=fallback-compare-profile
    if "$PYTHON" -u dev/tp2_decode/run_fallback.py --sdk "$SDK" --bundle "$GEMV" \
        --output "$RUN/results/sweep" --l2-bytes "${L2_BYTES:-67108864}" \
        "${RESUME[@]}" "${EXTRA[@]}" 2>&1 | tee "$RUN/results/console.log"; then
        printf 'DECODE_FALLBACK_GATE status=PASS performance_winners_require_review=1\n'
    else failures=$((failures+1)); fi
    stage=complete
    printf 'DECODE_FALLBACK_DONE failed_phases=%s auto_promote=0\n' "$failures"
    [[ $failures == 0 ]]
)
