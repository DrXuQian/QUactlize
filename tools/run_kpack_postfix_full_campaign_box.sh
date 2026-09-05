#!/usr/bin/env bash
# Rebuild and execute the complete five-format post-fix K-pack campaign.
#
# This is intentionally not an overlay/resume of historical result bundles.
# Resume is permitted only inside this OUT, whose frozen plan, catalog, source,
# submodule, SDK, workload, assignment, device, and result hashes are checked
# before reuse.

set -uo pipefail

fail() {
  printf '[kpack-postfix-full] FAIL: %s\n' "$*" >&2
  return 2
}

terminate_children() {
  local signal_name="$1" status="$2" pid
  trap - INT TERM
  interrupted_status="$status"
  for pid in "${child_pids[@]}"; do
    [ -n "$pid" ] || continue
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  printf '[kpack-postfix-full] INTERRUPTED signal=%s direct_workers=%s output_preserved=1\n' \
    "$signal_name" "${#child_pids[@]}" >&2
}

artifact_args() {
  local file="$1" artifact_id artifact_root
  ARTIFACT_ARGS=()
  while IFS=$'\t' read -r artifact_id artifact_root; do
    [ -n "$artifact_id" ] && [ -n "$artifact_root" ] || {
      fail "malformed artifact-root row in $file"; return $?; }
    ARTIFACT_ARGS+=(--artifact-root "$artifact_id=$artifact_root")
  done <"$file"
  [ "${#ARTIFACT_ARGS[@]}" -gt 0 ] || {
    fail "artifact-root file is empty: $file"; return $?; }
}

cpu_snapshot() {
  local label user nice system idle iowait irq softirq steal guest guest_nice
  read -r label user nice system idle iowait irq softirq steal guest guest_nice \
    </proc/stat || return 2
  [ "$label" = cpu ] || return 2
  cpu_current_idle=$((idle + ${iowait:-0}))
  cpu_current_total=$((user + nice + system + idle + ${iowait:-0} + \
    ${irq:-0} + ${softirq:-0} + ${steal:-0}))
}

cpu_report() {
  local delta_total delta_idle delta_busy report
  cpu_snapshot || {
    printf '[kpack-postfix-full] BUILD_CPU_WARNING reason=PROC_STAT_UNREADABLE\n' >&2
    return 0
  }
  delta_total=$((cpu_current_total - cpu_previous_total))
  delta_idle=$((cpu_current_idle - cpu_previous_idle))
  delta_busy=$((delta_total - delta_idle))
  cpu_previous_total="$cpu_current_total"
  cpu_previous_idle="$cpu_current_idle"
  [ "$delta_total" -gt 0 ] || {
    printf '[kpack-postfix-full] BUILD_CPU_WARNING reason=ZERO_SAMPLE_INTERVAL\n' >&2
    return 0
  }
  report="$(awk -v busy="$delta_busy" -v total="$delta_total" \
    -v logical="$cpu_logical" 'BEGIN {
      pct=100.0*busy/total; cores=logical*busy/total;
      verdict=(pct >= 80.0 ? "PASS" : "LOW_UTILIZATION_WARNING");
      printf "busy_pct=%.1f core_equivalents=%.1f threshold_pct=80.0 verdict=%s", pct, cores, verdict
    }')" || return 2
  printf '[kpack-postfix-full] BUILD_CPU logical_cpus=%s %s\n' \
    "$cpu_logical" "$report"
  case "$report" in *LOW_UTILIZATION_WARNING*)
    printf '[kpack-postfix-full] WARNING: build CPU utilization is below the 80%% steady-state target; workers continue and completed artifacts remain resumable\n' >&2
  esac
}

main() {
  [ "$#" -eq 0 ] || { fail 'no positional arguments are accepted'; return $?; }
  local root sdk output_parent out source_sha short resume jobs
  local requested_scratch scratch out_tag local_build publish inputs campaign logs results
  local build_partitions build_workers runtime_workers nominal_slots minimum_slots remove_local
  local expected_manifests expected_items plan manifests alive failures completed
  local worker pid marker log aggregate interrupted_status value
  local cpu_logical cpu_previous_total cpu_previous_idle cpu_current_total cpu_current_idle
  local -a build_pids run_pids child_pids result_args evidence_args resume_arg

  build_partitions="${KPACK_BUILD_PARTITIONS:-32}"
  build_workers="${KPACK_BUILD_WORKERS:-32}"
  runtime_workers="${KPACK_RUNTIME_WORKERS:-8}"
  jobs="${JOBS:-6}"
  remove_local="${KPACK_REMOVE_LOCAL_AFTER_PUBLISH:-1}"
  resume="${RESUME:-0}"
  for value in "$build_partitions" "$build_workers" "$runtime_workers" "$jobs"; do
    case "$value" in ''|*[!0-9]*|0)
      fail 'partition/worker/JOBS values must be positive integers'; return $?;;
    esac
  done
  [ "$build_partitions" -le 32 ] || {
    fail 'KPACK_BUILD_PARTITIONS exceeds the canonical maximum 32'; return $?; }
  [ "$build_workers" -le "$build_partitions" ] || {
    fail 'KPACK_BUILD_WORKERS exceeds KPACK_BUILD_PARTITIONS'; return $?; }
  [ "$runtime_workers" -le "$build_partitions" ] || {
    fail 'KPACK_RUNTIME_WORKERS exceeds KPACK_BUILD_PARTITIONS'; return $?; }
  case "$resume" in 0|1) ;; *) fail 'RESUME must be 0 or 1'; return $?;; esac
  case "$remove_local" in 0|1) ;; *)
    fail 'KPACK_REMOVE_LOCAL_AFTER_PUBLISH must be 0 or 1'; return $?;; esac

  interrupted_status=0
  child_pids=()
  trap 'terminate_children INT 130' INT
  trap 'terminate_children TERM 143' TERM
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 2
  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" || {
    fail 'set PPU_SDK to the exact box SDK'; return $?; }
  [ -x "$sdk/bin/hgcc" ] && [ -x "$sdk/bin/hgobjdump" ] || {
    fail 'PPU_SDK lacks hgcc/hgobjdump'; return $?; }
  output_parent="$(realpath -e -- "${KPACK_CAMPAIGN_OUTPUT_ROOT:-/workspace}")" || {
    fail 'KPACK_CAMPAIGN_OUTPUT_ROOT is missing'; return $?; }
  requested_scratch="${KPACK_LOCAL_SCRATCH_ROOT:-/root/autodl-tmp}"
  [ -d "$requested_scratch" ] && [ ! -L "$requested_scratch" ] || {
    fail 'KPACK_LOCAL_SCRATCH_ROOT must be a regular non-symlink directory'; return $?; }
  scratch="$(realpath -e -- "$requested_scratch")" || return 2
  case "$scratch" in /|/root|/workspace)
    fail 'KPACK_LOCAL_SCRATCH_ROOT is too broad'; return $?;; esac

  source_sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${source_sha:0:8}"
  out="$(realpath -m -- "${OUT:-$output_parent/quactlize-kpack-postfix-full-$short}")" || return 2
  case "$out" in "$output_parent"/*) ;; *)
    fail 'OUT must be a strict KPACK_CAMPAIGN_OUTPUT_ROOT child'; return $?;; esac
  case "$out" in "$scratch"|"$scratch"/*)
    fail 'OUT may not be inside KPACK_LOCAL_SCRATCH_ROOT'; return $?;; esac
  case "$scratch" in "$out"|"$out"/*)
    fail 'KPACK_LOCAL_SCRATCH_ROOT may not be inside OUT'; return $?;; esac
  if [ -e "$out" ] || [ -L "$out" ]; then
    [ "$resume" = 1 ] && [ -d "$out" ] && [ ! -L "$out" ] || {
      fail 'existing OUT requires RESUME=1 and a regular directory'; return $?; }
  else
    [ "$resume" = 0 ] || {
      fail 'RESUME=1 requires an existing OUT'; return $?; }
    mkdir -p "$out" || return 2
  fi

  out_tag="$(printf '%s' "$out" | sha256sum | cut -c1-12)" || return 2
  local_build="$scratch/quactlize-kpack-postfix-full-$short-$out_tag"
  case "$local_build" in "$scratch"/*) ;; *)
    fail 'local build root escaped KPACK_LOCAL_SCRATCH_ROOT'; return $?;; esac
  publish="$out/published"
  inputs="$out/inputs"
  campaign="$out/campaign"
  logs="$out/logs"
  results="$out/results"
  aggregate="$results/aggregate"
  mkdir -p "$publish" "$inputs" "$logs" "$results" "$local_build" || return 2
  export PPU_SDK="$sdk"
  export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib64:$sdk/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  python3 -B "$root/tools/kpack_postfix_full_campaign.py" self-test || return 2
  python3 -B "$root/tools/kpack_postfix_full_campaign.py" check-source || return 2
  python3 -B "$root/tools/kpack_discovery_build_partitions.py" self-test || return 2
  python3 -B "$root/tools/kpack_discovery_worker_plan.py" self-test || return 2
  python3 -B "$root/tools/aggregate_kpack_discovery_results.py" self-test || return 2
  plan="$inputs/build-plan.json"
  if [ -e "$plan" ] || [ -L "$plan" ]; then
    [ -f "$plan" ] && [ ! -L "$plan" ] || {
      fail 'build plan is not a regular file'; return $?; }
  fi
  python3 -B "$root/tools/kpack_postfix_full_campaign.py" emit-plan \
    --partitions "$build_partitions" --output "$plan" || return 2
  read -r expected_manifests expected_items < <(python3 -B - "$plan" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p["partition_count"]*2,339196)
PY
  ) || return 2
  [ "$expected_manifests" -eq $((build_partitions * 2)) ] && \
    [ "$expected_items" -eq 339196 ] || {
    fail 'full campaign denominator differs'; return $?; }
  nominal_slots=$((build_workers * jobs))
  cpu_logical="$(getconf _NPROCESSORS_ONLN)" || return 2
  case "$cpu_logical" in ''|*[!0-9]*|0)
    fail 'cannot determine logical CPU count'; return $?;; esac
  minimum_slots=$(((cpu_logical * 80 + 99) / 100))
  [ "$nominal_slots" -ge "$minimum_slots" ] || {
    fail "nominal build slots $nominal_slots are below 80% of $cpu_logical logical CPUs"; return $?; }
  cpu_previous_total=0
  cpu_previous_idle=0
  cpu_current_total=0
  cpu_current_idle=0
  cpu_snapshot || { fail 'cannot sample /proc/stat'; return $?; }
  cpu_previous_total="$cpu_current_total"
  cpu_previous_idle="$cpu_current_idle"
  printf '[kpack-postfix-full] PLAN source=%s formats=5 routes=2 operators=2 shards=2211 parents=70483 workloads=1381 atoms=%s build_partitions=%s build_workers=%s jobs_per_worker=%s nominal_build_slots=%s runtime_workers=%s overlay=FORBIDDEN\n' \
    "$source_sha" "$expected_items" "$build_partitions" "$build_workers" \
    "$jobs" "$nominal_slots" "$runtime_workers"

  mkdir -p "$publish/$source_sha" || return 2
  build_pids=()
  child_pids=()
  for ((worker = 0; worker < build_workers; ++worker)); do
    log="$logs/build-worker-$worker.log"
    printf '\n[kpack-postfix-full] build-attempt worker=%s resume=%s utc=%s\n' \
      "$worker" "$resume" "$(date -u +%Y%m%dT%H%M%SZ)" >>"$log"
    WORKER_ID="$worker" WORKER_COUNT="$build_workers" JOBS="$jobs" \
      PPU_SDK="$sdk" KPACK_LOCAL_SCRATCH_ROOT="$scratch" \
      KPACK_PAYLOAD_SOURCE_ROOT="$root" KPACK_BUILD_PARTITION_PLAN="$plan" \
      KPACK_PARTITION_LOCAL_ROOT="$local_build/worker-$worker" \
      KPACK_PARTITION_PUBLISH_ROOT="$publish" \
      KPACK_REMOVE_LOCAL_AFTER_PUBLISH="$remove_local" \
      KPACK_CONTINUE_ON_PARTITION_ERROR=1 \
      bash "$root/tools/build_kpack_discovery_partition_worker.sh" \
      >>"$log" 2>&1 &
    build_pids[$worker]="$!"
    child_pids+=("${build_pids[$worker]}")
    printf '[kpack-postfix-full] BUILD_STARTED worker=%s pid=%s\n' \
      "$worker" "${build_pids[$worker]}"
  done
  sleep 5
  while :; do
    alive=0
    for pid in "${build_pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
    done
    manifests="$(find "$publish/$source_sha" -type f \
      -name partition-bundle.json 2>/dev/null | wc -l)" || return 2
    printf '[kpack-postfix-full] BUILD_PROGRESS alive=%s/%s manifests=%s/%s slots=%s\n' \
      "$alive" "$build_workers" "$manifests" "$expected_manifests" "$nominal_slots"
    cpu_report || return 2
    [ "$alive" -gt 0 ] || break
    sleep 30
    [ "$interrupted_status" -eq 0 ] || return "$interrupted_status"
  done
  failures=0
  for ((worker = 0; worker < build_workers; ++worker)); do
    if wait "${build_pids[$worker]}"; then
      printf '[kpack-postfix-full] BUILD_PASS worker=%s\n' "$worker"
    else
      printf '[kpack-postfix-full] BUILD_FAIL worker=%s log=%s\n' \
        "$worker" "$logs/build-worker-$worker.log" >&2
      failures=$((failures + 1))
    fi
  done
  child_pids=()
  [ "$interrupted_status" -eq 0 ] || return "$interrupted_status"
  [ "$failures" -eq 0 ] || { fail "$failures build workers failed"; return $?; }
  manifests="$(find "$publish/$source_sha" -type f \
    -name partition-bundle.json | wc -l)" || return 2
  [ "$manifests" -eq "$expected_manifests" ] || {
    fail "partition manifest census $manifests/$expected_manifests"; return $?; }

  python3 -B "$root/tools/kpack_postfix_full_campaign.py" finalize \
    --plan "$plan" --publish-root "$publish" --workers "$runtime_workers" \
    --output "$campaign" || return 2

  child_pids=()
  build_pids=()
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    (
      artifact_args "$campaign/artifact-roots/worker-$worker.tsv" || return 2
      CUDA_VISIBLE_DEVICES="$worker" PPU_SDK="$sdk" \
        python3 -B "$root/tools/run_kpack_discovery_worker.py" probe-device \
        --bundle "$campaign/catalog.json" --plan "$campaign/workload-plan.json" \
        --master "$campaign/master.json" --assignment "$campaign/assignment.json" \
        --selection "$campaign/selections/worker-$worker.json" \
        --worker-id "$worker" "${ARTIFACT_ARGS[@]}" \
        --output "$campaign/worker-$worker-device.json"
    ) >"$logs/probe-worker-$worker.log" 2>&1 &
    build_pids[$worker]="$!"
    child_pids+=("${build_pids[$worker]}")
  done
  failures=0
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    if wait "${build_pids[$worker]}"; then
      printf '[kpack-postfix-full] PROBE_PASS worker=%s\n' "$worker"
    else
      printf '[kpack-postfix-full] PROBE_FAIL worker=%s log=%s\n' \
        "$worker" "$logs/probe-worker-$worker.log" >&2
      failures=$((failures + 1))
    fi
  done
  child_pids=()
  [ "$failures" -eq 0 ] || { fail "$failures device probes failed"; return $?; }
  python3 -B "$root/tools/kpack_postfix_full_campaign.py" bind-devices \
    --campaign "$campaign" || return 2

  run_pids=()
  child_pids=()
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    marker="$results/worker-$worker/worker-result.json"
    if [ -f "$marker" ] && [ ! -L "$marker" ]; then
      run_pids[$worker]=''
      printf '[kpack-postfix-full] RUN_REUSE worker=%s\n' "$worker"
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
        --bundle "$campaign/catalog.json" --plan "$campaign/workload-plan.json" \
        --master "$campaign/master.json" --assignment "$campaign/assignment.json" \
        --selection "$campaign/selections/worker-$worker.json" \
        --worker-id "$worker" "${ARTIFACT_ARGS[@]}" \
        --device-identity "$campaign/worker-$worker-device.json" \
        --device-homogeneity "$campaign/device-homogeneity.json" \
        --output "$results/worker-$worker" --phase all \
        --screen-iterations 5 --confirm-iterations 11 --confirm-rounds 3 \
        --correctness-repeats 256 --warmups 3 --continue-on-atom-error \
        "${resume_arg[@]}"
    ) >>"$logs/run-worker-$worker.log" 2>&1 &
    run_pids[$worker]="$!"
    child_pids+=("${run_pids[$worker]}")
    printf '[kpack-postfix-full] RUN_STARTED worker=%s pid=%s\n' \
      "$worker" "${run_pids[$worker]}"
  done
  while :; do
    alive=0
    for pid in "${run_pids[@]}"; do
      [ -z "$pid" ] || {
        kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); }
    done
    completed="$(find "$results" -type f -path '*/completion/*.json' \
      2>/dev/null | wc -l)" || return 2
    printf '[kpack-postfix-full] RUN_PROGRESS alive=%s/%s completed=%s/%s\n' \
      "$alive" "$runtime_workers" "$completed" "$expected_items"
    [ "$alive" -gt 0 ] || break
    sleep 30
    [ "$interrupted_status" -eq 0 ] || return "$interrupted_status"
  done
  failures=0
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    pid="${run_pids[$worker]}"
    if [ -z "$pid" ] || wait "$pid"; then
      printf '[kpack-postfix-full] RUN_PASS worker=%s\n' "$worker"
    else
      printf '[kpack-postfix-full] RUN_FAIL worker=%s log=%s\n' \
        "$worker" "$logs/run-worker-$worker.log" >&2
      failures=$((failures + 1))
    fi
  done
  child_pids=()
  [ "$failures" -eq 0 ] || { fail "$failures execution workers failed"; return $?; }
  completed="$(find "$results" -type f -path '*/completion/*.json' | wc -l)" || return 2
  [ "$completed" -eq "$expected_items" ] || {
    fail "completion census $completed/$expected_items"; return $?; }

  result_args=()
  evidence_args=()
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    marker="$results/worker-$worker/worker-result.json"
    [ -f "$marker" ] && [ ! -L "$marker" ] || {
      fail "worker $worker result authority is missing"; return $?; }
    [ -f "$results/worker-$worker/worker-evidence.json" ] && \
      [ ! -L "$results/worker-$worker/worker-evidence.json" ] || {
      fail "worker $worker evidence authority is missing"; return $?; }
    result_args+=(--result "$marker")
    evidence_args+=(--evidence "$results/worker-$worker/worker-evidence.json")
  done
  python3 -B "$root/tools/kpack_discovery_worker_plan.py" validate-results \
    --bundle "$campaign/catalog.json" --plan "$campaign/workload-plan.json" \
    --master "$campaign/master.json" --assignment "$campaign/assignment.json" \
    --device-homogeneity "$campaign/device-homogeneity.json" \
    "${result_args[@]}" || return 2

  # Aggregate even on resume.  The aggregator reconstructs a temporary output
  # from the current eight evidence documents and accepts an existing output
  # only when every staged byte is identical.  A stale sealed summary can
  # therefore never hide changed worker evidence.
  python3 -B "$root/tools/aggregate_kpack_discovery_results.py" aggregate \
    --bundle "$campaign/catalog.json" --plan "$campaign/workload-plan.json" \
    --workload-index "$campaign/workloads/index.json" \
    --master "$campaign/master.json" --assignment "$campaign/assignment.json" \
    --device-homogeneity "$campaign/device-homogeneity.json" \
    "${evidence_args[@]}" --output-dir "$aggregate" || return 2
  python3 -B "$root/tools/aggregate_kpack_discovery_results.py" \
    validate-output --output-dir "$aggregate" || return 2
  trap - INT TERM
  printf '[kpack-postfix-full] NEXT steady_census=COMPLETE reducer_cases=1035 confidence_followup=cold_compute+prepass+first_use heuristic=NOT_YET_EMITTED\n'
  printf '[kpack-postfix-full] PASS source=%s formats=5 shards=2211 parents=70483 workloads=1381 atoms=%s old_bundle_overlay=0 aggregate=%s output=%s\n' \
    "$source_sha" "$expected_items" "$aggregate" "$out"
}

main "$@"
