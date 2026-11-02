#!/usr/bin/env bash
# Snapshot only the host recovery controller; execute the old frozen harness.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-python3}
[[ ${IDLE_CONFIRMED:-0} == 1 ]] || { echo 'ERROR: confirm original cards are idle.' >&2; exit 1; }
: "${PREVIOUS_RESULTS:?set the previous results directory}"
: "${MEASUREMENT_SOURCE:?set the original frozen source directory}"
PREVIOUS_RESULTS=$(realpath -e -- "$PREVIOUS_RESULTS")
MEASUREMENT_SOURCE=$(realpath -e -- "$MEASUREMENT_SOURCE")
test -f "$PREVIOUS_RESULTS/run.json"
test -f "$MEASUREMENT_SOURCE/.quactlize-source.json"
export PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
test -x "$PPU_SDK/asight/bin/asys"
export PATH="$PPU_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$PPU_SDK/lib:$PPU_SDK/targets/x86_64-linux/lib:$PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
args=()
if [[ -n ${RESUME_RETRY:-} ]]; then
    RETRY_RUN=$(realpath -e -- "$RESUME_RETRY")
    test -f "$RETRY_RUN/results/recovery.json"
    args+=(--resume)
else
    REV=$(git -C "$ROOT" rev-parse --verify HEAD)
    TREE=$(git -C "$ROOT" rev-parse "${REV}^{tree}")
    ACT=$(git -C "$ROOT" rev-parse "${REV}:third_party/actlize")
    [[ $REV =~ ^[0-9a-f]{40}$ && $TREE =~ ^[0-9a-f]{40}$ && $ACT =~ ^[0-9a-f]{40}$ ]] || exit 1
    RETRY_PARENT=${OUTPUT_ROOT:-/workspace}
    test -d "$RETRY_PARENT"
    RETRY_RUN=$(mktemp -d "$RETRY_PARENT/kpack-moe-retry.XXXXXXXX")
    test -d "${RETRY_RUN:?}"
    mkdir "$RETRY_RUN/controller"
    git -C "$ROOT" archive "$REV" tools/retry_moe_refine.py \
        dev/moe_confirm/staged_grid.py dev/strategy/moe_prefill_campaign.py | tar -x -C "$RETRY_RUN/controller"
    "$PYTHON" - "$RETRY_RUN/controller/.quactlize-source.json" "$REV" "$TREE" "$ACT" <<'PY'
import json,sys
with open(sys.argv[1],'x') as out:
    json.dump(dict(quactlize_commit=sys.argv[2],quactlize_tree=sys.argv[3],
                   actlize_commit=sys.argv[4],snapshot='COMMITTED_ARCHIVE'),out,sort_keys=True)
PY
fi
[[ -z ${GPUS:-} ]] || args+=(--devices "$GPUS")
echo "MOE_RETRY_RUN=$RETRY_RUN"
rc=0
"$PYTHON" -u "$RETRY_RUN/controller/tools/retry_moe_refine.py" \
    --previous-run "$PREVIOUS_RESULTS" --source "$MEASUREMENT_SOURCE" \
    --output "$RETRY_RUN/results" --sdk "$PPU_SDK" \
    --l2-bytes "${L2_BYTES:-67108864}" --budget-hours "${BUDGET_HOURS:-12}" \
    --confirm-idle "${args[@]}" 2>&1 | tee "$RETRY_RUN/console-$(date +%s%N).log" || rc=$?
echo "runner_rc=$rc run=$RETRY_RUN"
echo "Resume: RESUME_RETRY=$RETRY_RUN (with the same PREVIOUS_RESULTS and MEASUREMENT_SOURCE)"
exit "$rc"
