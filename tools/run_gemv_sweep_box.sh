#!/usr/bin/env bash
set -euo pipefail

# Build and run one finite GEMV tactic space.  The default is the only run that
# may publish a full-space winner: all ten format/layout groups in one binary,
# one manifest and one run identity.  GEMV_SWEEP_GROUPS=i4-native is a bounded
# partial-space smoke test and is labelled as such by the binary/analyser.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

GROUPS=${GEMV_SWEEP_GROUPS:-all}
OUT=${GEMV_SWEEP_DIR:-/tmp/quactlize-gemv-sweep}
BUILD_TIMEOUT=${GEMV_SWEEP_BUILD_TIMEOUT_SECONDS:-7200}
RUN_DEADLINE=${GEMV_SWEEP_DEADLINE_SECONDS:-7200}
SHAPE_TIMEOUT=${GEMV_SWEEP_SHAPE_TIMEOUT_SECONDS:-900}
CORES=${MOE_CORES:-72}
SHA=$(git rev-parse HEAD)

# The raw build identity below is only honest for an exact committed tree.
if [[ -n $(git status --porcelain=v1 --untracked-files=all) ]]; then
  echo "[gemv-box] refusing a dirty source tree; commit/stash every root and submodule change" >&2
  exit 2
fi
SUBMODULE_STATUS=$(git submodule status --recursive)
if grep -Eq '^[+\-U]' <<<"$SUBMODULE_STATUS"; then
  echo "[gemv-box] refusing a submodule checkout that differs from the recorded gitlink" >&2
  printf '%s\n' "$SUBMODULE_STATUS" >&2
  exit 2
fi

if [[ "$GROUPS" == all ]]; then
  TAG=full
  BUILD_GROUP_ARGS=(env -u GEMV_GROUPS)
else
  TAG=${GROUPS//[^A-Za-z0-9_.-]/_}
  BUILD_GROUP_ARGS=(env "GEMV_GROUPS=$GROUPS")
fi

ROOT_RUN="$OUT/$SHA/$TAG"
BUILD="$ROOT_RUN/build"
mkdir -p "$ROOT_RUN"
echo "[gemv-box] sha=$SHA groups=$GROUPS build=$BUILD"

EXISTING_BIN=$(find "$BUILD" -type f -name test_gemv_perf -perm -u+x -print -quit 2>/dev/null || true)
REUSE_MODE=${GEMV_SWEEP_REUSE_BUILD:-auto}
if [[ "$REUSE_MODE" == 1 && -z "$EXISTING_BIN" ]]; then
  echo "[gemv-box] GEMV_SWEEP_REUSE_BUILD=1 but no existing binary is present" >&2
  exit 2
fi
if [[ "$REUSE_MODE" == 0 || ( "$REUSE_MODE" == auto && -z "$EXISTING_BIN" ) ]]; then
  timeout "${BUILD_TIMEOUT}s" "${BUILD_GROUP_ARGS[@]}" \
    PPU_BUILD_DIR="$BUILD" MOE_CORES="$CORES" JOBS="$CORES" TARGET=test_gemv_perf \
    ./build.sh 2>&1 | tee "$ROOT_RUN/build.log"
elif [[ "$REUSE_MODE" == 1 || "$REUSE_MODE" == auto ]]; then
  echo "[gemv-box] reusing the existing binary (set GEMV_SWEEP_REUSE_BUILD=0 to rebuild)"
else
  echo "[gemv-box] GEMV_SWEEP_REUSE_BUILD must be auto, 0, or 1" >&2
  exit 2
fi

BIN=$(find "$BUILD" -type f -name test_gemv_perf -perm -u+x -print -quit)
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "[gemv-box] missing executable below $BUILD" >&2
  exit 2
fi

BIN_SHA=$(sha256sum "$BIN" | awk '{print $1}')
PROTOCOL=samples20
BUILD_ID="$SHA/bin-sha256:$BIN_SHA/protocol:$PROTOCOL"
RUN_ID="gemv-${SHA:0:12}-${BIN_SHA:0:16}-$PROTOCOL"
BASE="$ROOT_RUN/$BIN_SHA-$PROTOCOL"
mkdir -p "$BASE/logs"
exec 9>"$BASE/run.lock"
if ! flock -n 9; then
  echo "[gemv-box] another sweep owns $BASE" >&2
  exit 2
fi
echo "[gemv-box] binary_sha256=$BIN_SHA protocol=$PROTOCOL out=$BASE"

PLAN="$BASE/manifest.json"
RAW="$BASE/raw.jsonl"
PROGRESS="$BASE/progress.jsonl"
RESULT="$BASE/result.json"
"$BIN" --manifest-json "$PLAN"
python3 benchmarks/sweep_gemv_perf.py run "$PLAN" --bin "$BIN" \
  --raw "$RAW" --progress "$PROGRESS" --shape-timeout "$SHAPE_TIMEOUT" \
  --deadline-seconds "$RUN_DEADLINE" --build-id "$BUILD_ID" --run-id "$RUN_ID" --dry-run \
  --dry-run-manifest "$BASE/pending.audit.jsonl" > "$BASE/pending.summary.jsonl"

RUN_ARGS=()
if [[ -e "$RAW" || -e "$PROGRESS" ]]; then RUN_ARGS+=(--resume); fi
set +e
python3 benchmarks/sweep_gemv_perf.py run "$PLAN" --bin "$BIN" \
  --raw "$RAW" --progress "$PROGRESS" --shape-timeout "$SHAPE_TIMEOUT" \
  --deadline-seconds "$RUN_DEADLINE" --build-id "$BUILD_ID" --run-id "$RUN_ID" \
  --logs-dir "$BASE/logs" "${RUN_ARGS[@]}" 2>&1 | tee -a "$BASE/run.log"
RUN_RC=${PIPESTATUS[0]}
set -e

if [[ -s "$RAW" ]]; then
  set +e
  python3 benchmarks/sweep_gemv_perf.py analyse "$RAW" --manifest "$PLAN" \
    --output "$RESULT"
  ANALYSE_RC=$?
  set -e
else
  ANALYSE_RC=2
fi

python3 - "$PLAN" "$RUN_RC" "$ANALYSE_RC" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
print("[gemv-box] manifest:", json.dumps({
    "space_id": plan["space_id"], "partial_space": plan["partial_space"],
    "jobs": len(plan["jobs"]), "counts": plan["counts"],
    "run_rc": int(sys.argv[2]), "analyse_rc": int(sys.argv[3]),
}, sort_keys=True))
PY
echo "[gemv-box] binary=$BIN"
echo "[gemv-box] build_id=$BUILD_ID run_id=$RUN_ID"
echo "[gemv-box] manifest=$PLAN raw=$RAW progress=$PROGRESS result=$RESULT"
if (( RUN_RC != 0 )); then
  echo "[gemv-box] bounded run incomplete; rerun the same command to resume" >&2
  exit "$RUN_RC"
fi
exit "$ANALYSE_RC"
