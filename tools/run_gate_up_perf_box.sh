#!/usr/bin/env bash
# Prebuilt-only component A/B. No caller/runtime package is replaced.
(
    set -Eeuo pipefail
    phase=precheck
    RUN=""
    finish() {
        rc=$?
        trap - EXIT
        if [[ -n "$RUN" && -d "$RUN/results" ]]; then
            tar -czf "$RUN.results.tgz" -C "$RUN" results || rc=1
            printf '\nresults=%s.results.tgz\n' "$RUN"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$phase"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "GATE_UP_PERF FAIL stage=%s line=%s rc=%s\n" "$phase" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/dev/gate_up_perf/run.py"
    SDK=$(realpath -e -- "${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}")
    test -n "$SDK" && test -f "$SDK/envsetup.sh"
    PYTHON=$(command -v "${PYTHON:-python3}")
    test -n "$PYTHON" && test -x "$PYTHON"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    cd "$ROOT"
    export PPU_SDK="$SDK"
    export LD_LIBRARY_PATH="$SDK/CUDA_SDK/targets/x86_64-linux/lib:$SDK/targets/x86_64-linux/lib:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
    RESULT_DIR=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    test -n "$RESULT_DIR" && test -d "$RESULT_DIR"
    if [[ -n ${RESUME_RUN:-} ]]; then
        RUN=$(realpath -e -- "$RESUME_RUN")
        test -n "$RUN" && test -f "$RUN/results/perf/identity.json"
        RESUME=(--resume)
    else
        RUN=$(mktemp -d "$RESULT_DIR/gate-up-perf.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
        RESUME=()
    fi
    if [[ -z ${PERF_BUNDLE:-} ]]; then
        phase=fetch
        mapfile -t PIN < <("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["branch"]); print(p["commit"]); print(p["path"]); print(p["manifest_sha256"])' "$ROOT/tools/kpack_gate_up_perf_artifact.json")
        [[ ${#PIN[@]} == 4 && ${PIN[0]} == artifacts/gate-up-paired-n4-v1 && ${PIN[1]} =~ ^[0-9a-f]{40}$ && ${PIN[2]} == prebuilt/ppu0010/gate-up-perf-v1 && ${PIN[3]} =~ ^[0-9a-f]{64}$ ]]
        ART="$RESULT_DIR/quactlize-gate-up-perf-artifact-${PIN[1]:0:12}"
        if [[ ! -e "$ART" ]]; then
            GIT_LFS_SKIP_SMUDGE=1 git fetch origin "${PIN[0]}"
            git cat-file -e "${PIN[1]}^{commit}"
            GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$ART" "${PIN[1]}"
        fi
        [[ $(git -C "$ART" rev-parse HEAD) == "${PIN[1]}" ]]
        git -C "$ART" lfs pull --include="${PIN[2]}/**/*.so,${PIN[2]}/*.so" --exclude=''
        PERF_BUNDLE="$ART/${PIN[2]}"
        "$PYTHON" -c 'import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1])/"manifest.json"; assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2], "manifest hash differs"' "$PERF_BUNDLE" "${PIN[3]}"
    fi
    PERF_BUNDLE=$(realpath -e -- "$PERF_BUNDLE")
    test -n "$PERF_BUNDLE" && test -f "$PERF_BUNDLE/manifest.json"
    phase=verify
    "$PYTHON" dev/gate_up_perf/run.py --sdk "$SDK" --bundle "$PERF_BUNDLE" \
        --output "$RUN/results/perf" --verify-only | tee "$RUN/results/verify.log"
    cp "$PERF_BUNDLE/manifest.json" "$RUN/results/manifest.json"
    git rev-parse HEAD | tee "$RUN/results/source.txt"
    phase=performance
    "$PYTHON" dev/gate_up_perf/run.py --sdk "$SDK" --bundle "$PERF_BUNDLE" \
        --output "$RUN/results/perf" "${RESUME[@]}" --l2-bytes "${L2_BYTES:-0}" \
        --rounds "${PERF_ROUNDS:-4}" --samples "${PERF_SAMPLES:-11}" 2>&1 | tee "$RUN/results/console.log"
    phase=complete
)
