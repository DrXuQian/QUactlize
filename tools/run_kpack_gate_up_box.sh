#!/usr/bin/env bash
# Prebuilt-only gate. Run in a subshell to preserve the caller's Docker shell.
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
    trap 'printf "GATE_UP FAIL stage=%s line=%s rc=%s\n" "$phase" "$LINENO" "$?" >&2' ERR
    ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)
    test -n "$ROOT" && test -f "$ROOT/tools/run_kpack_gate_up.py"
    cd "$ROOT"
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
        test -n "$RUN" && test -f "$RUN/results/gate/identity.json"
        RESUME=(--resume)
    else
        RUN=$(mktemp -d "$RESULT_DIR/gate-up-paired-n4.XXXXXX")
        test -n "$RUN" && test -d "$RUN"
        mkdir "$RUN/results"
        RESUME=()
    fi
    if [[ -z ${BUNDLE:-} ]]; then
        phase=fetch
        mapfile -t PIN < <("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["branch"]); print(p["commit"]); print(p["path"]); print(p["manifest_sha256"])' "$ROOT/tools/kpack_gate_up_artifact.json")
        [[ ${#PIN[@]} == 4 && ${PIN[0]} == artifacts/gate-up-paired-n4-v1 && ${PIN[1]} =~ ^[0-9a-f]{40}$ && ${PIN[2]} == prebuilt/ppu0010/gate-up-paired-n4-v1 && ${PIN[3]} =~ ^[0-9a-f]{64}$ ]]
        ART="$RESULT_DIR/quactlize-gate-up-artifact-${PIN[1]:0:12}"
        if [[ ! -e "$ART" ]]; then
            GIT_LFS_SKIP_SMUDGE=1 git fetch origin "${PIN[0]}"
            git cat-file -e "${PIN[1]}^{commit}"
            GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$ART" "${PIN[1]}"
        fi
        [[ $(git -C "$ART" rev-parse HEAD) == "${PIN[1]}" ]]
        git -C "$ART" lfs pull --include="${PIN[2]}/*.so" --exclude=''
        BUNDLE="$ART/${PIN[2]}"
        "$PYTHON" -c 'import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1])/"manifest.json"; assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2], "manifest hash differs"' "$BUNDLE" "${PIN[3]}"
    fi
    BUNDLE=$(realpath -e -- "$BUNDLE")
    test -n "$BUNDLE" && test -f "$BUNDLE/manifest.json"
    phase=verify
    "$PYTHON" tools/run_kpack_gate_up.py --sdk "$SDK" --bundle "$BUNDLE" \
        --output "$RUN/results/gate" --verify-only | tee "$RUN/results/verify.log"
    cp "$BUNDLE/manifest.json" "$RUN/results/manifest.json"
    git rev-parse HEAD | tee "$RUN/results/source.txt"
    phase=numerical
    if [[ ${GATE_UP_DIAGNOSTIC:-0} == 1 ]]; then
        test -z "${RESUME_RUN:-}"
        phase=rounding-diagnostic
        "$PYTHON" tools/diagnose_gate_up_rounding.py --sdk "$SDK" --bundle "$BUNDLE" \
            --output "$RUN/results/rounding" 2>&1 | tee "$RUN/results/console.log"
    else
        "$PYTHON" tools/run_kpack_gate_up.py --sdk "$SDK" --bundle "$BUNDLE" \
            --output "$RUN/results/gate" "${RESUME[@]}" \
            --formats "${GATE_UP_FORMATS:-8,10,11,12,13,14}" \
            --backends "${GATE_UP_BACKENDS:-simt,tc}" 2>&1 | tee "$RUN/results/console.log"
    fi
    phase=complete
    printf 'GATE_UP_DONE timing=NOT_MEASURED production_selection=UNCHANGED\n'
)
