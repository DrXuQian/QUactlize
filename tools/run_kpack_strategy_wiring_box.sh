#!/usr/bin/env bash
# Selected-call admission, not another configuration sweep. Caller build uses
# only llama's existing .aoneci/scripts/build.sh through its isolated wrapper.
(
    set -Eeuo pipefail
    ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
    stage=precheck
    RUN=
    finish() {
        local rc=$?
        trap - EXIT ERR
        if [[ -n $RUN && -d $RUN/results ]]; then
            printf 'runner_rc=%s stage=%s\n' "$rc" "$stage" > "$RUN/results/runner-status.txt"
            for report in "$RUN"/ab/review-*.tgz; do
                if [[ -f $report && ! -L $report ]]; then cp -- "$report" "$RUN/results/" || rc=1; fi
            done
            tar -czf "$RUN.results.tgz" -C "$RUN" results || rc=1
            printf 'results=%s.results.tgz\nWIRING_RUN=%s\n' "$RUN" "$RUN"
        fi
        printf 'runner_rc=%s stage=%s\nCurrent Docker shell is preserved.\n' "$rc" "$stage"
        exit "$rc"
    }
    trap finish EXIT
    trap 'printf "STRATEGY_WIRING FAIL stage=%s line=%s rc=%s\n" "$stage" "$LINENO" "$?" >&2' ERR
    [[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'Set IDLE_CONFIRMED=1 only after checking the selected GPUs are idle.' >&2; exit 2; }
    SDK=$(realpath -e -- "${PPU_SDK:-/usr/local/PPU_SDK}")
    LLAMA=$(realpath -e -- "${LLAMA_CI_DIR:-/sim/eec/shared/junfu.qx/llama.cpp}")
    NCP=$(realpath -e -- "${NCP_LIB_DIR:-/sim/eec/shared/junfu.qx/ncp_flash_lib}")
    RESULT_ROOT=$(realpath -e -- "${RESULT_ROOT:-/workspace}")
    JOBS=${JOBS:-192}
    DEVICES=${DEVICES:-1,2,3,4,5,6,7}
    AB_PHASE=${AB_PHASE:-all}
    [[ $AB_PHASE == all || $AB_PHASE == pilot ]] || { echo 'AB_PHASE must be all or pilot.' >&2; exit 2; }
    # This handoff is explicitly for the uploaded ZW810/64 MiB cohort. The
    # SDK can return zero for L2; preserve the independently verified value.
    L2_BYTES=${L2_BYTES:-67108864}
    [[ $JOBS =~ ^[1-9][0-9]*$ && $DEVICES =~ ^[0-9]+(,[0-9]+)*$ ]]
    [[ $L2_BYTES == 67108864 ]] || { echo 'This confirmation is ZW810/64 MiB only; use a separate experiment for another device.' >&2; exit 2; }
    [[ -f $SDK/envsetup.sh && -f $LLAMA/.aoneci/scripts/build.sh && -f $NCP/CMakeLists.txt ]]
    : "${QUACTLIZE_PPU_BUNDLE:?Set the existing six-library legacy intake bundle; this runner never fetches old artifacts}"
    LEGACY=$(realpath -e -- "$QUACTLIZE_PPU_BUNDLE")
    test -s "$LEGACY/manifest.json"
    set +u
    source "$SDK/envsetup.sh"
    set -Eeuo pipefail
    export PPU_SDK="$SDK" LC_ALL=C CUDA_VISIBLE_DEVICES="${DEVICES%%,*}"
    export LD_LIBRARY_PATH="$SDK/lib:$SDK/targets/x86_64-linux/lib:$SDK/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    PYTHON=$(command -v "${PYTHON:-python3}")
    "$PYTHON" -c 'import numpy, gguf, torch; from deep_gemm.jit_kernels.m_grouped_gemm import m_grouped_gemm_bf16_bf16_bf16_nt_nopad'
    LLAMA_REV=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["llama_commit"])' "$ROOT/tools/kpack_strategy_wiring.json")
    if ! git -C "$LLAMA" cat-file -e "$LLAMA_REV^{commit}"; then
        echo 'Fetch the quactlize branch from DrXuQian/llama.cpp into LLAMA_CI_DIR first; do not reset its worktree.' >&2
        exit 2
    fi
    cd "$ROOT"
    "$PYTHON" tools/fit_prefill_confirmation.py --check
    if [[ -n ${WIRING_RUN:-} ]]; then
        RUN=$(realpath -e -- "$WIRING_RUN")
        test -s "$RUN/results/inputs.json"
    else
        RUN=$(mktemp -d "$RESULT_ROOT/kpack-strategy-wiring.XXXXXXXX")
        mkdir "$RUN/results"
    fi
    export WIRING_RUN="$RUN"
    "$PYTHON" - "$RUN" "$ROOT" "$SDK" "$LLAMA_REV" "$DEVICES" "$L2_BYTES" <<'PY'
import json,os,sys,subprocess
from pathlib import Path
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import Compiler
from tools.fit_prefill_confirmation import EVIDENCE
run,root,sdk=map(Path,sys.argv[1:4])
if subprocess.check_output(['git','diff','--name-only','HEAD'],text=True).strip():
    raise ValueError('use a clean committed Quactlize tree for a resumable wiring run')
identity=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              caller=sys.argv[4],devices=sys.argv[5],sdk=Compiler(sdk,run/'unused').sdk_identity)
if {a['sdk'] for a in json.loads(EVIDENCE.read_text())['authorities']}!={identity['sdk']}:
    raise ValueError('SDK differs from the confirmed component cohort; do not relabel it as the same experiment')
selectors=sys.argv[5].split(',')
if len(set(selectors))!=len(selectors):raise ValueError('device ordinals must be distinct')
hardware=[]
for selector in selectors:
    command=[sys.executable,str(root/'tools/run_kpack_llama_ab.py'),'--probe-device',
             '--sdk',str(sdk),'--l2-bytes',sys.argv[6]]
    h=json.loads(subprocess.check_output(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=selector),text=True))
    if h['device']['compute_units']!=72 or h['l2']['bytes']!=67108864:
        raise ValueError('selected device is outside the confirmed ZW810/72-CU/64-MiB cohort')
    hardware.append(h)
if len({(h['device']['host'],h['device']['pci']) for h in hardware})!=len(selectors):
    raise ValueError('selectors refer to the same physical device')
identity['hardware']=hardware
path=run/'results/inputs.json'
if path.exists():
    if json.loads(path.read_text())!=identity:raise ValueError('resume source/caller/SDK/device differs; use a new run')
else:atomic_json(path,identity)
PY
    JIT_CACHE=$(realpath -m -- "${JIT_CACHE:-/workspace/kpack-prefill-7gpu.js14yGTG/tc-cache}")
    export QUACTLIZE_KPACK_JIT_CACHE="$JIT_CACHE"
    stage=runtime-build
    if [[ ! -f $RUN/public/manifest.json ]]; then
        [[ ! -e $RUN/public ]] || { echo 'Incomplete public build: inspect results/runtime-build.log; use a fresh run after fixing the compiler error.' >&2; exit 1; }
        "$PYTHON" -u tools/build_ppu_library.py --sdk "$SDK" --output "$RUN/public" --jobs "$JOBS" \
            2>&1 | tee "$RUN/results/runtime-build.log"
    fi
    "$PYTHON" tools/build_ppu_library.py --verify --output "$RUN/public"
    if [[ ! -f $RUN/bundle/manifest.json ]]; then
        "$PYTHON" -u tools/build_kpack_dispatch.py --sdk "$SDK" --output "$RUN/bundle" --jit-only \
            --execution-bundle "$RUN/public.build/execution" --prefill-runtime "$RUN/public.build/prefill" --jobs "$JOBS" \
            2>&1 | tee "$RUN/results/dispatcher-build.log"
    fi
    "$PYTHON" tools/verify_kpack_dispatch.py "$RUN/bundle" --sdk "$SDK"
    stage=selected-prewarm
    for plan in final-selection prefill-selection; do
        "$PYTHON" -u tools/kpack_jit.py prewarm --sdk "$SDK" --cache "$JIT_CACHE" --jobs "$JOBS" \
            --plan "$RUN/bundle/$plan.json" --receipt "$RUN/results/$plan-prewarm.json" \
            2>&1 | tee "$RUN/results/$plan-prewarm.log"
    done
    stage=selected-tickets
    if [[ ! -f $RUN/results/tickets/summary.json ]]; then
        CUDA_VISIBLE_DEVICES="${DEVICES%%,*}" "$PYTHON" -u tools/check_kpack_prefill_wiring.py \
            --sdk "$SDK" --bundle "$RUN/bundle" --jit-cache "$JIT_CACHE" --output "$RUN/results/tickets" \
            2>&1 | tee "$RUN/results/tickets.log"
    fi
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="PASS" and d["points"]==430' "$RUN/results/tickets/summary.json"
    stage=caller-build
    if [[ ! -f $RUN/results/caller-build.json ]]; then
        reuse=()
        if [[ -n ${NCP_CI_DIR:-} ]]; then reuse+=(--reuse-ncp-build "$NCP_CI_DIR"); fi
        "$PYTHON" -u tools/build_kpack_model_ci.py --llama "$LLAMA" --ncp "$NCP" --sdk "$SDK" \
            --llama-revision "$LLAMA_REV" --output "$RUN/ci" --receipt "$RUN/results/caller-build.json" \
            --jobs "$JOBS" "${reuse[@]}" 2>&1 | tee "$RUN/results/caller-build.log"
    fi
    CALLER_BUILD=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["build"])' "$RUN/results/caller-build.json")
    stage=native-ab-$AB_PHASE
    resume=()
    if [[ -f $RUN/ab/request.json ]]; then resume+=(--resume); fi
    "$PYTHON" -u tools/run_kpack_llama_ab.py --llama-source "$RUN/ci/llama" --llama-build "$CALLER_BUILD" \
        --sdk "$SDK" --bundle "$RUN/bundle" --pack-library "$RUN/public/libquactlize_ppu.so" \
        --legacy-bundle "$LEGACY" --jit-cache "$JIT_CACHE" --jit-helper "$ROOT/tools/kpack_jit.py" \
        --devices "$DEVICES" --l2-bytes "$L2_BYTES" --phase "$AB_PHASE" --hours 0 --idle-confirmed --output "$RUN/ab" \
        "${resume[@]}" 2>&1 | tee "$RUN/results/native-ab.log"
    stage=complete
    printf 'STRATEGY_WIRING_DONE run=%s scope=SELECTED_TICKETS_AND_NATIVE_OPERATOR_AB_NOT_MODEL_TPOT\n' "$RUN"
)
