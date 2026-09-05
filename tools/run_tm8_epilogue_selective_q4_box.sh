#!/usr/bin/env bash
# Build and execute the exact Q4 subset invalidated by the TM8 epilogue fix.
# The command is resumable and runs long compilation directly on the PPU box.

set -uo pipefail

fail() {
  printf '[tm8-selective-q4] FAIL: %s\n' "$*" >&2
  return 2
}

artifact_args() {
  local file="$1" id path
  ARTIFACT_ARGS=()
  while IFS=$'\t' read -r id path; do
    [ -n "$id" ] && [ -n "$path" ] || {
      fail "malformed artifact-root row in $file"; return $?; }
    ARTIFACT_ARGS+=(--artifact-root "$id=$path")
  done <"$file"
  [ "${#ARTIFACT_ARGS[@]}" -gt 0 ] || {
    fail "empty artifact-root file $file"; return $?; }
}

main() {
  [ "$#" -eq 0 ] || { fail 'no positional arguments are accepted'; return $?; }
  local root sdk output_parent out local_parent requested_local_parent local_base out_tag source_sha short resume jobs
  local scope plan campaign publish logs results expected_manifests expected_items
  local worker pid alive failures manifests completed marker log
  local -a pids run_pids result_args resume_arg

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 2
  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" || {
    fail 'set PPU_SDK to the box SDK'; return $?; }
  [ -x "$sdk/bin/hgcc" ] && [ -x "$sdk/bin/hgobjdump" ] || {
    fail 'PPU_SDK lacks hgcc/hgobjdump'; return $?; }
  output_parent="$(realpath -e -- "${KPACK_SELECTIVE_OUTPUT_ROOT:-/workspace}")" || {
    fail 'KPACK_SELECTIVE_OUTPUT_ROOT is missing'; return $?; }
  requested_local_parent="${KPACK_LOCAL_SCRATCH_ROOT:-/root/autodl-tmp}"
  [ -d "$requested_local_parent" ] && [ ! -L "$requested_local_parent" ] || {
    fail 'KPACK_LOCAL_SCRATCH_ROOT must be a regular non-symlink directory'; return $?; }
  local_parent="$(realpath -e -- "$requested_local_parent")" || {
    fail 'KPACK_LOCAL_SCRATCH_ROOT is not a readable directory'; return $?; }
  [ -d "$output_parent" ] && [ ! -L "$output_parent" ] || {
    fail 'output root must be a regular directory'; return $?; }
  case "$local_parent" in /|/root|/workspace)
    fail 'KPACK_LOCAL_SCRATCH_ROOT is too broad'; return $?;; esac
  source_sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${source_sha:0:8}"
  out="$(realpath -m -- "${OUT:-$output_parent/quactlize-tm8-selective-q4-$short}")" || return 2
  case "$out" in "$output_parent"/*) ;; *)
    fail 'OUT must be a strict KPACK_SELECTIVE_OUTPUT_ROOT child'; return $?;; esac
  case "$out" in "$local_parent"|"$local_parent"/*)
    fail 'OUT may not be inside KPACK_LOCAL_SCRATCH_ROOT'; return $?;; esac
  case "$local_parent" in "$out"|"$out"/*)
    fail 'KPACK_LOCAL_SCRATCH_ROOT may not be inside OUT'; return $?;; esac
  out_tag="$(printf '%s' "$out" | sha256sum | cut -c1-12)" || return 2
  local_base="$local_parent/quactlize-tm8-selective-q4-$short-$out_tag"
  case "$local_base" in "$local_parent"/*) ;; *)
    fail 'local build path escaped KPACK_LOCAL_SCRATCH_ROOT'; return $?;; esac
  resume="${RESUME:-0}"; jobs="${JOBS:-24}"
  case "$resume" in 0|1) ;; *) fail 'RESUME must be 0 or 1'; return $?;; esac
  case "$jobs" in ''|*[!0-9]*|0) fail 'JOBS must be positive'; return $?;; esac
  if [ -e "$out" ] || [ -L "$out" ]; then
    [ "$resume" = 1 ] && [ -d "$out" ] && [ ! -L "$out" ] || {
      fail 'existing OUT requires RESUME=1 and a regular directory'; return $?; }
  else
    [ "$resume" = 0 ] || { fail 'RESUME=1 requires existing OUT'; return $?; }
    mkdir -p "$out" || return 2
  fi
  mkdir -p "$out/inputs" "$out/published" "$out/logs" "$out/results" \
    "$local_base" || return 2
  scope="$out/inputs/tm8-scope.json"
  plan="$out/inputs/q4-build-plan.json"
  campaign="$out/campaign"; publish="$out/published"
  logs="$out/logs"; results="$out/results"
  export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib64:$sdk/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  python3 -B "$root/tools/plan_tm8_epilogue_fix_scope.py" self-test || return 2
  python3 -B "$root/tools/tm8_epilogue_selective_campaign.py" self-test || return 2
  python3 -B "$root/tools/kpack_discovery_build_partitions.py" self-test || return 2
  python3 -B "$root/tools/kpack_discovery_worker_plan.py" self-test || return 2
  if [ -f "$scope" ] && [ ! -L "$scope" ]; then
    python3 -B "$root/tools/plan_tm8_epilogue_fix_scope.py" validate \
      --input "$scope" || return 2
  else
    [ ! -e "$scope" ] && [ ! -L "$scope" ] || {
      fail 'scope authority is not a regular file'; return $?; }
    python3 -B "$root/tools/plan_tm8_epilogue_fix_scope.py" emit \
      --out "$scope" || return 2
  fi
  if [ -f "$plan" ] && [ ! -L "$plan" ]; then
    python3 -B "$root/tools/tm8_epilogue_selective_campaign.py" validate \
      --scope "$scope" --plan "$plan" || return 2
  else
    [ ! -e "$plan" ] && [ ! -L "$plan" ] || {
      fail 'build plan is not a regular file'; return $?; }
    python3 -B "$root/tools/tm8_epilogue_selective_campaign.py" emit \
      --scope "$scope" --partitions 8 --qtype 12 --output "$plan" || return 2
  fi
  read -r expected_manifests expected_items < <(python3 -B - "$plan" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p["partition_count"]*2,p["denominator"]["runtime_candidate_work_items"])
PY
  ) || return 2
  [ "$expected_manifests" = 16 ] && [ "$expected_items" = 5296 ] || {
    fail "Q4 denominator differs manifests=$expected_manifests items=$expected_items"; return $?; }
  printf '[tm8-selective-q4] PLAN source=%s shards=83 parents=2656 atoms=%s\n' \
    "$source_sha" "$expected_items"

  # The progress sampler runs immediately after the workers are forked.  Own
  # the exact source directory up front so `find` cannot race the first
  # publisher under `set -o pipefail`.
  mkdir -p "$publish/$source_sha" || return 2

  pids=()
  for worker in 0 1 2 3 4 5 6 7; do
    log="$logs/build-worker-$worker.log"
    printf '\n[tm8-selective-q4] build-attempt worker=%s resume=%s utc=%s\n' \
      "$worker" "$resume" "$(date -u +%Y%m%dT%H%M%SZ)" >>"$log"
    WORKER_ID="$worker" WORKER_COUNT=8 JOBS="$jobs" PPU_SDK="$sdk" \
      KPACK_LOCAL_SCRATCH_ROOT="$local_parent" \
      KPACK_PAYLOAD_SOURCE_ROOT="$root" \
      KPACK_BUILD_PARTITION_PLAN="$plan" \
      KPACK_PARTITION_LOCAL_ROOT="$local_base/worker-$worker" \
      KPACK_PARTITION_PUBLISH_ROOT="$publish" \
      KPACK_REMOVE_LOCAL_AFTER_PUBLISH=1 \
      bash "$root/tools/build_kpack_discovery_partition_worker.sh" \
      >>"$log" 2>&1 &
    pids[$worker]="$!"
    printf '[tm8-selective-q4] BUILD_STARTED worker=%s pid=%s\n' \
      "$worker" "${pids[$worker]}"
  done
  while :; do
    alive=0
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
    done
    manifests="$(find "$publish/$source_sha" -type f \
      -name partition-bundle.json 2>/dev/null | wc -l)" || return 2
    printf '[tm8-selective-q4] BUILD_PROGRESS alive=%s/8 manifests=%s/%s\n' \
      "$alive" "$manifests" "$expected_manifests"
    [ "$alive" -gt 0 ] || break
    sleep 30
  done
  failures=0
  for worker in 0 1 2 3 4 5 6 7; do
    if wait "${pids[$worker]}"; then
      printf '[tm8-selective-q4] BUILD_PASS worker=%s\n' "$worker"
    else
      printf '[tm8-selective-q4] BUILD_FAIL worker=%s log=%s\n' \
        "$worker" "$logs/build-worker-$worker.log" >&2
      failures=$((failures + 1))
    fi
  done
  [ "$failures" -eq 0 ] || { fail "$failures build workers failed"; return $?; }
  manifests="$(find "$publish/$source_sha" -type f \
    -name partition-bundle.json | wc -l)" || return 2
  [ "$manifests" = "$expected_manifests" ] || {
    fail "partition manifest census $manifests/$expected_manifests"; return $?; }

  python3 -B "$root/tools/tm8_epilogue_selective_campaign.py" finalize \
    --plan "$plan" --publish-root "$publish" --workers 8 \
    --output "$campaign" || return 2

  pids=()
  for worker in 0 1 2 3 4 5 6 7; do
    (
      artifact_args "$campaign/artifact-roots/worker-$worker.tsv" || return 2
      CUDA_VISIBLE_DEVICES="$worker" PPU_SDK="$sdk" \
        python3 -B "$root/tools/run_kpack_discovery_worker.py" probe-device \
        --bundle "$campaign/catalog.json" \
        --plan "$campaign/workload-plan.json" \
        --master "$campaign/master.json" \
        --assignment "$campaign/assignment.json" \
        --selection "$campaign/selections/worker-$worker.json" \
        --worker-id "$worker" "${ARTIFACT_ARGS[@]}" \
        --output "$campaign/worker-$worker-device.json"
    ) >"$logs/probe-worker-$worker.log" 2>&1 &
    pids[$worker]="$!"
  done
  failures=0
  for worker in 0 1 2 3 4 5 6 7; do
    if wait "${pids[$worker]}"; then
      printf '[tm8-selective-q4] PROBE_PASS worker=%s\n' "$worker"
    else
      printf '[tm8-selective-q4] PROBE_FAIL worker=%s log=%s\n' \
        "$worker" "$logs/probe-worker-$worker.log" >&2
      failures=$((failures + 1))
    fi
  done
  [ "$failures" -eq 0 ] || { fail "$failures device probes failed"; return $?; }
  python3 -B "$root/tools/tm8_epilogue_selective_campaign.py" bind-devices \
    --campaign "$campaign" || return 2

  run_pids=()
  for worker in 0 1 2 3 4 5 6 7; do
    marker="$results/worker-$worker/worker-result.json"
    if [ -f "$marker" ] && [ ! -L "$marker" ]; then
      run_pids[$worker]=''
      printf '[tm8-selective-q4] RUN_REUSE worker=%s\n' "$worker"
      continue
    fi
    [ ! -L "$results/worker-$worker" ] || {
      fail "worker $worker result root is a symlink"; return $?; }
    resume_arg=()
    [ ! -d "$results/worker-$worker" ] || resume_arg=(--resume)
    (
      artifact_args "$campaign/artifact-roots/worker-$worker.tsv" || return 2
      CUDA_VISIBLE_DEVICES="$worker" PPU_SDK="$sdk" \
        python3 -B "$root/tools/run_kpack_discovery_worker.py" run \
        --bundle "$campaign/catalog.json" \
        --plan "$campaign/workload-plan.json" \
        --master "$campaign/master.json" \
        --assignment "$campaign/assignment.json" \
        --selection "$campaign/selections/worker-$worker.json" \
        --worker-id "$worker" "${ARTIFACT_ARGS[@]}" \
        --device-identity "$campaign/worker-$worker-device.json" \
        --device-homogeneity "$campaign/device-homogeneity.json" \
        --output "$results/worker-$worker" --phase all \
        --screen-iterations 5 --confirm-iterations 11 --confirm-rounds 3 \
        --correctness-repeats 256 --warmups 3 "${resume_arg[@]}"
    ) >>"$logs/run-worker-$worker.log" 2>&1 &
    run_pids[$worker]="$!"
    printf '[tm8-selective-q4] RUN_STARTED worker=%s pid=%s\n' \
      "$worker" "${run_pids[$worker]}"
  done
  while :; do
    alive=0
    for pid in "${run_pids[@]}"; do
      [ -z "$pid" ] || { kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); }
    done
    completed="$(find "$results" -type f -path '*/completion/*.json' \
      2>/dev/null | wc -l)" || return 2
    printf '[tm8-selective-q4] RUN_PROGRESS alive=%s/8 completed=%s/%s\n' \
      "$alive" "$completed" "$expected_items"
    [ "$alive" -gt 0 ] || break
    sleep 30
  done
  failures=0
  for worker in 0 1 2 3 4 5 6 7; do
    pid="${run_pids[$worker]}"
    if [ -z "$pid" ] || wait "$pid"; then
      printf '[tm8-selective-q4] RUN_PASS worker=%s\n' "$worker"
    else
      printf '[tm8-selective-q4] RUN_FAIL worker=%s log=%s\n' \
        "$worker" "$logs/run-worker-$worker.log" >&2
      failures=$((failures + 1))
    fi
  done
  [ "$failures" -eq 0 ] || { fail "$failures execution workers failed"; return $?; }
  completed="$(find "$results" -type f -path '*/completion/*.json' | wc -l)" || return 2
  [ "$completed" = "$expected_items" ] || {
    fail "completion census $completed/$expected_items"; return $?; }
  result_args=()
  for worker in 0 1 2 3 4 5 6 7; do
    marker="$results/worker-$worker/worker-result.json"
    [ -f "$marker" ] && [ ! -L "$marker" ] || {
      fail "worker $worker result authority is missing"; return $?; }
    result_args+=(--result "$marker")
  done
  python3 -B "$root/tools/kpack_discovery_worker_plan.py" validate-results \
    --bundle "$campaign/catalog.json" --plan "$campaign/workload-plan.json" \
    --master "$campaign/master.json" --assignment "$campaign/assignment.json" \
    --device-homogeneity "$campaign/device-homogeneity.json" \
    "${result_args[@]}" || return 2
  printf '[tm8-selective-q4] PASS source=%s shards=83 atoms=%s workers=8 output=%s\n' \
    "$source_sha" "$expected_items" "$out"
}

main "$@"
