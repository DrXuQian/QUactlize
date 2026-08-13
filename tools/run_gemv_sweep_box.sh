#!/usr/bin/env bash
set -euo pipefail

# Build and run one finite GEMV tactic space.  The default is the only run that
# may publish a full-space winner: all ten format/layout groups in one binary,
# one manifest and one run identity.  GEMV_SWEEP_GROUPS=i4-native is a bounded
# partial-space smoke test and is labelled as such by the binary/analyser.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
ORIGINAL_ARGV=("$0" "$@")
if [[ "$#" -ne 0 ]]; then
  echo "[gemv-box] this runner accepts no positional arguments" >&2
  exit 2
fi

GROUPS=${GEMV_SWEEP_GROUPS:-all}
OUT=${GEMV_SWEEP_DIR:-/tmp/quactlize-gemv-sweep}
BUILD_TIMEOUT=${GEMV_SWEEP_BUILD_TIMEOUT_SECONDS:-7200}
RUN_DEADLINE=${GEMV_SWEEP_DEADLINE_SECONDS:-7200}
SHAPE_TIMEOUT=${GEMV_SWEEP_SHAPE_TIMEOUT_SECONDS:-900}
CORES=${MOE_CORES:-72}
SHA=$(git rev-parse HEAD)
ACTLIZE_SHA=$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)
PROVENANCE_TOOL="$ROOT/tools/write_box_run_provenance.py"
POLICY="$ROOT/dev/fold_derivation/BOX_RUN_PREREGISTRATION.md"
SAMPLES=$(python3 "$PROVENANCE_TOOL" policy-sample-count --policy "$POLICY" --kind gemv)
if [[ "$SAMPLES" -ne 20 ]]; then
  echo "[gemv-box] harness supports exactly 20 samples, policy requests $SAMPLES" >&2
  exit 2
fi
export GEMV_SWEEP_SAMPLES="$SAMPLES"

require_identity() {
  local name="$1" value="${!1:-}" normalized
  if [[ -z "$value" ]]; then
    echo "[gemv-box] required explicit device identity $name is unset" >&2
    exit 2
  fi
  normalized="${value,,}"
  case "$normalized" in
    unknown|unset|n/a|na|none)
      echo "[gemv-box] required explicit device identity $name is not measured" >&2
      exit 2 ;;
  esac
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "[gemv-box] required explicit device identity $name must be one line" >&2
    exit 2
  fi
}
for identity in QUACTLIZE_BOX_DEVICE_MODEL QUACTLIZE_BOX_PCI_IDENTITY \
                QUACTLIZE_BOX_DRIVER_VERSION QUACTLIZE_BOX_SDK_COMPILER_IDENTITY; do
  require_identity "$identity"
done

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
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%S.%NZ).$$"
ATTEMPT_ROOT="$(mktemp -d "$ROOT_RUN/.attempt.${ATTEMPT_ID}.XXXXXX")"
RUNNER_ATTEMPT_LOG="$ATTEMPT_ROOT/runner.log"
COMMANDS_LOG="$ATTEMPT_ROOT/commands.jsonl"
SUBMODULE_STATUS_FILE="$ATTEMPT_ROOT/submodule-status.txt"
RUN_IDENTITY_CANDIDATE="$ATTEMPT_ROOT/run-identity.json"
printf '%s\n' "$SUBMODULE_STATUS" >"$SUBMODULE_STATUS_FILE"

# Preserve this exact attempt without requiring an operator-side tee.  Restore
# the original descriptors before copying so the bundle never races tee.
exec 3>&1 4>&2
exec > >(tee "$RUNNER_ATTEMPT_LOG") 2>&1
RUNNER_TEE_PID=$!
RUNNER_STREAM_CLOSED=0
BASE_LOCK_HELD=0
IDENTITY_ACCEPTED=0
ATTEMPT_ARCHIVED=0

finish_runner_log() {
  if [[ "$RUNNER_STREAM_CLOSED" -eq 0 ]]; then
    exec 1>&3 2>&4
    wait "$RUNNER_TEE_PID"
    RUNNER_STREAM_CLOSED=1
  fi
}

archive_attempt() {
  local rc="$1" destination
  [[ "$BASE_LOCK_HELD" -eq 1 && "$IDENTITY_ACCEPTED" -eq 1 \
      && "$ATTEMPT_ARCHIVED" -eq 0 ]] || return 0
  destination="$BASE/attempts/$ATTEMPT_ID"
  if [[ -e "$destination" ]]; then
    echo "[gemv-box] refusing duplicate attempt identity $ATTEMPT_ID" >&2
    return 1
  fi
  mv "$ATTEMPT_ROOT" "$destination"
  ATTEMPT_ROOT="$destination"
  RUNNER_ATTEMPT_LOG="$destination/runner.log"
  SUBMODULE_STATUS_FILE="$destination/submodule-status.txt"
  COMMANDS_LOG="$destination/commands.jsonl"
  RUN_IDENTITY_CANDIDATE="$destination/run-identity.json"
  if [[ "$rc" -ne 0 && -n "${BIN_SHA:-}" && -s "$COMMANDS_LOG" ]]; then
    python3 "$PROVENANCE_TOOL" write \
      --output "$destination/provenance.json" \
      --root-sha "$SHA" --root-status clean \
      --submodule-status-file "$SUBMODULE_STATUS_FILE" \
      --actlize-sha "$ACTLIZE_SHA" --binary-sha256 "$BIN_SHA" \
      --device-model "$QUACTLIZE_BOX_DEVICE_MODEL" \
      --pci-identity "$QUACTLIZE_BOX_PCI_IDENTITY" \
      --driver-version "$QUACTLIZE_BOX_DRIVER_VERSION" \
      --sdk-compiler-identity "$QUACTLIZE_BOX_SDK_COMPILER_IDENTITY" \
      --run-identity-file "$BASE/run-identity.json" \
      --commands-file "$COMMANDS_LOG" --runner-exit-status "$rc" \
      --protocol-sample-count "$SAMPLES" -- "${ORIGINAL_ARGV[@]}" || true
  fi
  ATTEMPT_ARCHIVED=1
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ "$rc" -ne 0 ]]; then
    printf '[gemv-box] runner_exit_status=%d\n' "$rc"
  fi
  finish_runner_log
  archive_attempt "$rc" || true
  exit "$rc"
}
trap on_exit EXIT

record_command() {
  local role="$1" rc="$2"; shift 2
  python3 "$PROVENANCE_TOOL" record --path "$COMMANDS_LOG" \
    --role "$role" --exit-status "$rc" -- "$@"
}

verify_source_identity() {
  local current_root current_actlize current_submodules current_binary
  if [[ -n $(git status --porcelain=v1 --untracked-files=all) ]]; then
    echo "[gemv-box] final source identity check found a dirty root or submodule tree" >&2
    return 1
  fi
  current_root=$(git rev-parse HEAD)
  current_actlize=$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)
  current_submodules=$(git submodule status --recursive)
  current_binary=$(sha256sum "$BIN" | awk '{print $1}')
  [[ "$current_root" == "$SHA" ]] || {
    echo "[gemv-box] root HEAD changed during the run: $SHA -> $current_root" >&2
    return 1
  }
  [[ "$current_actlize" == "$ACTLIZE_SHA" ]] || {
    echo "[gemv-box] actlize HEAD changed during the run: $ACTLIZE_SHA -> $current_actlize" >&2
    return 1
  }
  cmp -s <(printf '%s\n' "$current_submodules") "$SUBMODULE_STATUS_FILE" || {
    echo "[gemv-box] recursive submodule status changed during the run" >&2
    return 1
  }
  [[ "$current_binary" == "$BIN_SHA" ]] || {
    echo "[gemv-box] selected binary changed during the run: $BIN_SHA -> $current_binary" >&2
    return 1
  }
  echo "[gemv-box] final-source-identity=EXACT root=$current_root actlize=$current_actlize binary=$current_binary"
}

compose_canonical_journals() {
  local commands_tmp="$BASE/commands.jsonl.tmp.$$"
  local runner_tmp="$BASE/runner.log.tmp.$$"
  : >"$commands_tmp"
  : >"$runner_tmp"
  while IFS= read -r attempt; do
    [[ -s "$attempt/commands.jsonl" ]] && cat "$attempt/commands.jsonl" >>"$commands_tmp"
    [[ -s "$attempt/runner.log" ]] && cat "$attempt/runner.log" >>"$runner_tmp"
  done < <(find "$BASE/attempts" -mindepth 1 -maxdepth 1 -type d -print | LC_ALL=C sort)
  [[ -s "$commands_tmp" && -s "$runner_tmp" ]] || {
    echo "[gemv-box] cannot promote an empty canonical journal" >&2
    return 1
  }
  mv "$commands_tmp" "$BASE/commands.jsonl"
  mv "$runner_tmp" "$BASE/runner.log"
}

echo "[gemv-box] root-status=clean actlize-sha=$ACTLIZE_SHA samples=$SAMPLES"
echo "[gemv-box] device-model=$QUACTLIZE_BOX_DEVICE_MODEL pci=$QUACTLIZE_BOX_PCI_IDENTITY driver=$QUACTLIZE_BOX_DRIVER_VERSION sdk=$QUACTLIZE_BOX_SDK_COMPILER_IDENTITY"
echo "[gemv-box] sha=$SHA groups=$GROUPS build=$BUILD"

# Building and selecting a binary are one transaction.  Holding this lock
# through the run also prevents a forced rebuild from replacing the inode after
# its SHA-256 has become part of the immutable run identity.
exec 8>"$ROOT_RUN/build.lock"
if ! flock -n 8; then
  echo "[gemv-box] another invocation owns the build/select transaction at $ROOT_RUN" >&2
  exit 2
fi
echo "[gemv-box] build-lock=HELD path=$ROOT_RUN/build.lock"

EXISTING_BIN=$(find "$BUILD" -type f -name test_gemv_perf -perm -u+x -print -quit 2>/dev/null || true)
REUSE_MODE=${GEMV_SWEEP_REUSE_BUILD:-auto}
if [[ "$REUSE_MODE" == 1 && -z "$EXISTING_BIN" ]]; then
  echo "[gemv-box] GEMV_SWEEP_REUSE_BUILD=1 but no existing binary is present" >&2
  exit 2
fi
if [[ "$REUSE_MODE" == 0 || ( "$REUSE_MODE" == auto && -z "$EXISTING_BIN" ) ]]; then
  BUILD_CMD=(timeout "${BUILD_TIMEOUT}s" "${BUILD_GROUP_ARGS[@]}"
    PPU_BUILD_DIR="$BUILD" MOE_CORES="$CORES" JOBS="$CORES" TARGET=test_gemv_perf \
    ./build.sh)
  set +e
  "${BUILD_CMD[@]}" 2>&1 | tee "$ROOT_RUN/build.log"
  BUILD_RC=${PIPESTATUS[0]}
  set -e
  record_command device-build "$BUILD_RC" "${BUILD_CMD[@]}"
  if (( BUILD_RC != 0 )); then exit "$BUILD_RC"; fi
elif [[ "$REUSE_MODE" == 1 || "$REUSE_MODE" == auto ]]; then
  echo "[gemv-box] reusing the existing binary (set GEMV_SWEEP_REUSE_BUILD=0 to rebuild)"
  [[ -s "$ROOT_RUN/build.log" ]] || {
    echo "[gemv-box] refusing reused binary without its original build.log" >&2
    exit 2
  }
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
PROTOCOL="samples${SAMPLES}"
BUILD_ID="$SHA/bin-sha256:$BIN_SHA/protocol:$PROTOCOL"
RUN_ID="gemv-${SHA:0:12}-${BIN_SHA:0:16}-$PROTOCOL"
BASE="$ROOT_RUN/$BIN_SHA-$PROTOCOL"
PLAN="$BASE/manifest.json"
RAW="$BASE/raw.jsonl"
PROGRESS="$BASE/progress.jsonl"
RESULT="$BASE/result.json"
python3 "$PROVENANCE_TOOL" write-identity \
  --output "$RUN_IDENTITY_CANDIDATE" \
  --root-sha "$SHA" --submodule-status-file "$SUBMODULE_STATUS_FILE" \
  --actlize-sha "$ACTLIZE_SHA" --binary-sha256 "$BIN_SHA" \
  --device-model "$QUACTLIZE_BOX_DEVICE_MODEL" \
  --pci-identity "$QUACTLIZE_BOX_PCI_IDENTITY" \
  --driver-version "$QUACTLIZE_BOX_DRIVER_VERSION" \
  --sdk-compiler-identity "$QUACTLIZE_BOX_SDK_COMPILER_IDENTITY" \
  --protocol-sample-count "$SAMPLES" >/dev/null

# The BASE lock lives outside BASE: creating BASE/run.lock before flock would
# itself mutate the bundle owned by another process.  No BASE path is written
# until both this lock and the immutable identity comparison succeed.
exec 9>"$ROOT_RUN/base-${BIN_SHA}-${PROTOCOL}.lock"
if ! flock -n 9; then
  echo "[gemv-box] another sweep owns $BASE" >&2
  exit 2
fi
BASE_LOCK_HELD=1

if [[ -e "$BASE/run-identity.json" ]]; then
  python3 "$PROVENANCE_TOOL" verify-identity \
    --expected "$BASE/run-identity.json" \
    --candidate "$RUN_IDENTITY_CANDIDATE" >/dev/null || {
      echo "[gemv-box] resume identity mismatch; raw/progress were not touched" >&2
      exit 2
    }
elif [[ -d "$BASE" && -n $(find "$BASE" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  echo "[gemv-box] refusing an existing bundle without immutable run-identity.json" >&2
  exit 2
else
  mkdir -p "$BASE"
  cp "$RUN_IDENTITY_CANDIDATE" "$BASE/run-identity.json.tmp.$$"
  mv "$BASE/run-identity.json.tmp.$$" "$BASE/run-identity.json"
fi

# A completed canonical bundle is immutable.  In particular, a later failed
# attempt must not append commands/raw or replace its successful provenance.
if [[ -e "$BASE/provenance.json" ]]; then
  set +e
  python3 - "$BASE/provenance.json" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as exc:
    print(f"malformed canonical provenance: {exc}", file=sys.stderr)
    raise SystemExit(2)
raise SystemExit(0 if value.get("runner_exit_status") == 0 else 1)
PY
  canonical_rc=$?
  set -e
  if [[ "$canonical_rc" -eq 0 ]]; then
    echo "[gemv-box] completed canonical bundle is immutable: $BASE" >&2
    exit 2
  elif [[ "$canonical_rc" -eq 2 ]]; then
    echo "[gemv-box] refusing malformed canonical provenance at $BASE" >&2
    exit 2
  fi
fi

IDENTITY_ACCEPTED=1
mkdir -p "$BASE/logs" "$BASE/attempts"

# A reused executable is admissible only when an earlier attempt for this exact
# immutable identity recorded the real build command.  A build.log by itself
# cannot prove which command produced the selected bytes.
if ! grep -Fq '"role":"device-build"' "$COMMANDS_LOG" 2>/dev/null \
    && ! grep -FRq '"role":"device-build"' "$BASE/attempts" 2>/dev/null; then
  echo "[gemv-box] binary has no identity-matched device-build journal; rebuild with GEMV_SWEEP_REUSE_BUILD=0" >&2
  exit 2
fi
echo "[gemv-box] binary_sha256=$BIN_SHA protocol=$PROTOCOL out=$BASE"

CENSUS_CMD=(python3 "$ROOT/tools/export_gemv_base_census.py"
  --output "$BASE/base-census.json" --authority-log "$BASE/base-census-authority.log")
set +e
"${CENSUS_CMD[@]}"
CENSUS_RC=$?
set -e
record_command base-tactic-census "$CENSUS_RC" "${CENSUS_CMD[@]}"
if (( CENSUS_RC != 0 )); then
  echo "[gemv-box] base tactic census export failed" >&2
  exit "$CENSUS_RC"
fi

MANIFEST_CMD=("$BIN" --manifest-json "$PLAN")
set +e
"${MANIFEST_CMD[@]}"
MANIFEST_RC=$?
set -e
record_command manifest "$MANIFEST_RC" "${MANIFEST_CMD[@]}"
if (( MANIFEST_RC != 0 )); then exit "$MANIFEST_RC"; fi

DRY_CMD=(python3 benchmarks/sweep_gemv_perf.py run "$PLAN" --bin "$BIN"
  --raw "$RAW" --progress "$PROGRESS" --shape-timeout "$SHAPE_TIMEOUT" \
  --deadline-seconds "$RUN_DEADLINE" --build-id "$BUILD_ID" --run-id "$RUN_ID" --dry-run \
  --dry-run-manifest "$BASE/pending.audit.jsonl")
set +e
"${DRY_CMD[@]}" > "$BASE/pending.summary.jsonl"
DRY_RC=$?
set -e
record_command dry-run-audit "$DRY_RC" "${DRY_CMD[@]}"
if (( DRY_RC != 0 )); then exit "$DRY_RC"; fi

RUN_ARGS=()
if [[ -e "$RAW" || -e "$PROGRESS" ]]; then RUN_ARGS+=(--resume); fi
set +e
RUN_CMD=(python3 benchmarks/sweep_gemv_perf.py run "$PLAN" --bin "$BIN"
  --raw "$RAW" --progress "$PROGRESS" --shape-timeout "$SHAPE_TIMEOUT" \
  --deadline-seconds "$RUN_DEADLINE" --build-id "$BUILD_ID" --run-id "$RUN_ID" \
  --logs-dir "$BASE/logs" "${RUN_ARGS[@]}")
"${RUN_CMD[@]}" 2>&1 | tee -a "$BASE/run.log"
RUN_RC=${PIPESTATUS[0]}
set -e
record_command measured-sweep "$RUN_RC" "${RUN_CMD[@]}"

if [[ -s "$RAW" ]]; then
  set +e
  ANALYSE_CMD=(python3 benchmarks/sweep_gemv_perf.py analyse "$RAW" --manifest "$PLAN"
    --output "$RESULT")
  "${ANALYSE_CMD[@]}"
  ANALYSE_RC=$?
  set -e
  record_command analyse "$ANALYSE_RC" "${ANALYSE_CMD[@]}"
else
  ANALYSE_RC=2
  record_command analyse "$ANALYSE_RC" python3 benchmarks/sweep_gemv_perf.py analyse \
    "$RAW" --manifest "$PLAN" --output "$RESULT"
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
FINAL_RC=$ANALYSE_RC
if (( RUN_RC != 0 )); then
  echo "[gemv-box] bounded run incomplete; rerun the same command to resume" >&2
  FINAL_RC=$RUN_RC
fi
echo "[gemv-box] runner_exit_status=$FINAL_RC"

# Close and join tee before materialising runner.log in the hash-qualified
# bundle.  This guarantees the copied file is the exact complete stream.
finish_runner_log
cp "$ROOT_RUN/build.log" "$BASE/build.log"
cp "$SUBMODULE_STATUS_FILE" "$BASE/submodule-status.txt"

python3 "$PROVENANCE_TOOL" write \
  --output "$BASE/provenance.json" \
  --root-sha "$SHA" --root-status clean \
  --submodule-status-file "$SUBMODULE_STATUS_FILE" \
  --actlize-sha "$ACTLIZE_SHA" --binary-sha256 "$BIN_SHA" \
  --device-model "$QUACTLIZE_BOX_DEVICE_MODEL" \
  --pci-identity "$QUACTLIZE_BOX_PCI_IDENTITY" \
  --driver-version "$QUACTLIZE_BOX_DRIVER_VERSION" \
  --sdk-compiler-identity "$QUACTLIZE_BOX_SDK_COMPILER_IDENTITY" \
  --commands-file "$COMMANDS_LOG" --runner-exit-status "$FINAL_RC" \
  --protocol-sample-count "$SAMPLES" -- "${ORIGINAL_ARGV[@]}"
trap - EXIT
exit "$FINAL_RC"
