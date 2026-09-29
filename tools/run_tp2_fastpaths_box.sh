#!/usr/bin/env bash
# Isolated prebuilt component gate. No caller build or production installation.
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
    trap 'printf "TP2_FASTPATHS FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/dev/tp2_decode/prebuilt.json"
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
        RUN=$(mktemp -d "$RESULT_DIR/tp2-fastpaths.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
    fi
    printf 'TP2_FASTPATHS run=%s device=%s prebuilt_only=1 production_replacement=0\n' "$RUN" "$DEVICE"
    git rev-parse HEAD > "$RUN/results/source.txt"
    cp dev/tp2_decode/prebuilt.json "$RUN/results/prebuilt.json"
    mapfile -t PIN < <("$PYTHON" -c 'import json; p=json.load(open("dev/tp2_decode/prebuilt.json")); print(p["artifact_branch"]); print(p["artifact_commit"]); print(p["gemv_path"]); print(p["prepare_path"])')
    [[ ${#PIN[@]} == 4 && ${PIN[0]} == artifacts/tp2-fastpaths-v1 && ${PIN[1]} =~ ^[0-9a-f]{40}$ ]]
    [[ ${PIN[2]} == prebuilt/ppu0010/tp2-fastpaths-v1 && ${PIN[3]} == prebuilt/ppu0010/tp2-prepare-v1 ]]
    stage=fetch
    ART=${TP2_ARTIFACT_DIR:-$RUN/artifact}
    if [[ -z ${TP2_ARTIFACT_DIR:-} ]]; then
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
    GEMV="$ART/${PIN[2]}"
    PREPARE="$ART/${PIN[3]}"
    stage=verify
    "$PYTHON" - "$ART" "$SDK" "$RUN/results" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path.cwd()
artifact, sdk, results = map(pathlib.Path, sys.argv[1:])
pin = json.loads((root / 'dev/tp2_decode/prebuilt.json').read_text())
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
for key in ('gemv', 'prepare'):
    path = artifact / pin[key + '_path'] / 'manifest.json'
    if sha(path) != pin[key + '_manifest_sha256']:
        raise ValueError(key + ' manifest differs from pinned handoff')
prepare = artifact / pin['prepare_path']
manifest = json.loads((prepare / 'manifest.json').read_text())
if sha(prepare / 'bench') != pin['prepare_binary_sha256'] or manifest['binary_sha256'] != pin['prepare_binary_sha256']:
    raise ValueError('prepare binary differs from pinned handoff')
for name, digest in manifest['source_hashes'].items():
    if sha(root / name) != digest:
        raise ValueError('prepare compile source differs: ' + name)
gemv = json.loads((artifact / pin['gemv_path'] / 'manifest.json').read_text())
runtime = {name: sha(sdk / 'lib' / name) for name in gemv['runtime']}
receipt = dict(sdk=str(sdk), runtime=runtime, build_runtime=gemv['runtime'],
               runtime_matches_build=runtime == gemv['runtime'], production_replacement=False)
(results / 'environment.json').write_text(json.dumps(receipt, indent=2) + '\n')
if runtime != gemv['runtime']:
    raise ValueError('SDK runtime differs from prebuilt; see environment.json')
print('TP2_FASTPATHS_PREFLIGHT PASS manifests=2 prepare_source=verified runtime=matched')
PY
    "$PYTHON" dev/gemv_model/run.py --cohort tp2 --sdk "$SDK" --bundle "$GEMV" \
        --output "$RUN/results/sweep" --verify-only | tee "$RUN/results/verify.log"
    cp "$GEMV/manifest.json" "$RUN/results/gemv-manifest.json"
    cp "$GEMV/native-inspection.json" "$RUN/results/native-inspection.json"
    cp "$PREPARE/manifest.json" "$RUN/results/prepare-manifest.json"
    if [[ ${VERIFY_ONLY:-0} == 1 ]]; then stage=verified-only; exit 0; fi
    EXTRA=()
    if [[ -n ${GEMV_POINTS:-} ]]; then EXTRA+=(--points "$GEMV_POINTS"); fi
    if [[ ${PROFILE:-1} == 1 ]]; then
        ACU=$(realpath -e -- "${ACU:-$SDK/asight/bin/acu}")
        test -n "$ACU" && test -x "$ACU"
        EXTRA+=(--acu "$ACU")
    fi
    failures=0
    prepare_failed=0
    stage=prepare-numerics
    for gate in shape-check router-edge-check router-alias-check; do
        printf 'TP2_FASTPATHS_PREPARE gate=%s\n' "$gate"
        if "$PREPARE/bench" "--$gate" 2>&1 | tee "$RUN/results/prepare-$gate.log"; then
            printf '%s\tPASS\n' "$gate" >> "$RUN/results/prepare-status.tsv"
        else
            failures=$((failures+1)); prepare_failed=1
            printf '%s\tFAIL\n' "$gate" >> "$RUN/results/prepare-status.tsv"
        fi
    done
    if [[ $prepare_failed == 0 ]]; then
        stage=prepare-timing
        # Prepare-only K3072 experiment; not a whole TP2 chain timing.
        for tokens in 1 8; do
            if "$PREPARE/bench" --case "$tokens" 3072 5 1 0 1 0 2>&1 | tee "$RUN/results/prepare-m$tokens.log"; then
                printf 'm%s\tPASS\n' "$tokens" >> "$RUN/results/prepare-status.tsv"
            else
                failures=$((failures+1))
                printf 'm%s\tFAIL\n' "$tokens" >> "$RUN/results/prepare-status.tsv"
            fi
        done
    fi
    stage=gemv-compare-profile
    if "$PYTHON" -u dev/gemv_model/run.py --cohort tp2 --sdk "$SDK" --bundle "$GEMV" \
        --output "$RUN/results/sweep" --l2-bytes "${L2_BYTES:-67108864}" \
        "${RESUME[@]}" "${EXTRA[@]}" 2>&1 | tee "$RUN/results/console.log"; then
        printf 'TP2_FASTPATHS_GEMV status=PASS\n'
    else
        failures=$((failures+1))
    fi
    stage=complete
    printf 'TP2_FASTPATHS_DONE failed_phases=%s production_replacement=0\n' "$failures"
    [[ $failures == 0 ]]
)
