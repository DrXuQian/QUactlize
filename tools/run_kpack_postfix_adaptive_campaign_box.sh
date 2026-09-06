#!/usr/bin/env bash
# Run a fresh adaptive K-pack timing epoch from one finalized full build.
#
# Required:
#   REUSE_CAMPAIGN=/workspace/.../campaign
#   OUT=/workspace/quactlize-kpack-adaptive-...
#   PPU_SDK=/workspace/.../PPU_SDK
#
# RESUME=1 continues only the same OUT.  Reuse applies to hash-validated build
# payloads and frozen catalog metadata; historical timing output is never read
# or copied.  This first adapter stops at the global eight-device screen
# barrier.  The next stage is deliberately explicit rather than silently
# confirming every compiled candidate.

set -uo pipefail

fail() {
  printf '[kpack-postfix-adaptive] FAIL: %s\n' "$*" >&2
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
  printf '[kpack-postfix-adaptive] INTERRUPTED signal=%s workers=%s output_preserved=1\n' \
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

main() {
  [ "$#" -eq 0 ] || { fail 'no positional arguments are accepted'; return $?; }
  local root sdk output_parent source out inputs results logs resume
  local runtime_workers worker pid alive completed failures interrupted_status
  local marker log expected_items
  local -a probe_pids run_pids child_pids probe_args resume_arg

  runtime_workers="${KPACK_RUNTIME_WORKERS:-8}"
  resume="${RESUME:-0}"
  [ "$runtime_workers" = 8 ] || {
    fail 'the adaptive screen barrier requires exactly 8 workers'; return $?; }
  case "$resume" in 0|1) ;; *) fail 'RESUME must be 0 or 1'; return $?;; esac

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 2
  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" || {
    fail 'set PPU_SDK to the exact build SDK'; return $?; }
  [ -x "$sdk/bin/hgcc" ] && [ -x "$sdk/bin/hgobjdump" ] || {
    fail 'PPU_SDK lacks hgcc/hgobjdump'; return $?; }
  source="$(realpath -e -- "${REUSE_CAMPAIGN:-/nonexistent}")" || {
    fail 'set REUSE_CAMPAIGN to a finalized full campaign directory'; return $?; }
  [ -d "$source" ] && [ ! -L "${REUSE_CAMPAIGN:-/nonexistent}" ] || {
    fail 'REUSE_CAMPAIGN must be a regular non-symlink directory'; return $?; }
  output_parent="$(realpath -e -- "${KPACK_CAMPAIGN_OUTPUT_ROOT:-/workspace}")" || {
    fail 'KPACK_CAMPAIGN_OUTPUT_ROOT is missing'; return $?; }
  [ -n "${OUT:-}" ] || { fail 'set OUT to a fresh result epoch'; return $?; }
  out="$(realpath -m -- "$OUT")" || return 2
  case "$out" in "$output_parent"/*) ;; *)
    fail 'OUT must be a strict KPACK_CAMPAIGN_OUTPUT_ROOT child'; return $?;; esac
  if [ -e "$out" ] || [ -L "$out" ]; then
    [ "$resume" = 1 ] && [ -d "$out" ] && [ ! -L "$out" ] || {
      fail 'existing OUT requires RESUME=1 and a regular directory'; return $?; }
  else
    [ "$resume" = 0 ] || {
      fail 'RESUME=1 requires an existing OUT'; return $?; }
    mkdir -p "$out" || return 2
  fi
  case "$out" in "$source"|"$source"/*) fail 'OUT overlaps REUSE_CAMPAIGN'; return $?;; esac
  case "$source" in "$out"/*) fail 'REUSE_CAMPAIGN is inside OUT'; return $?;; esac

  inputs="$out/inputs"
  results="$out/results"
  logs="$out/logs"
  if [ "$resume" = 0 ]; then
    for marker in "$inputs" "$results" "$logs"; do
      [ ! -e "$marker" ] && [ ! -L "$marker" ] || {
        fail 'fresh OUT unexpectedly contains campaign state'; return $?; }
    done
  fi
  mkdir -p "$inputs" "$results" "$logs" || return 2
  export PPU_SDK="$sdk"
  export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib64:$sdk/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  child_pids=()
  interrupted_status=0
  trap 'terminate_children INT 130' INT
  trap 'terminate_children TERM 143' TERM

  python3 -B "$root/tools/kpack_postfix_adaptive_campaign.py" self-test || return 2
  python3 -B "$root/tools/plan_kpack_screen_retention.py" self-test || return 2
  python3 -B "$root/tools/kpack_postfix_adaptive_campaign.py" prepare \
    --reuse-campaign "$source" --output "$inputs" || return 2
  expected_items="$(python3 -B - "$inputs/epoch.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["denominator"]["work_items"])
PY
  )" || return 2
  case "$expected_items" in ''|*[!0-9]*|0)
    fail 'adaptive denominator is not positive'; return $?;; esac

  probe_pids=()
  child_pids=()
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    log="$logs/probe-worker-$worker.log"
    (
      artifact_args "$inputs/artifact-roots/worker-$worker.tsv" || return 2
      CUDA_VISIBLE_DEVICES="$worker" PPU_SDK="$sdk" \
        python3 -B "$root/tools/run_kpack_discovery_worker.py" probe-device \
        --bundle "$inputs/catalog.json" --plan "$inputs/workload-plan.json" \
        --master "$inputs/master.json" --assignment "$inputs/assignment.json" \
        --selection "$inputs/selections/worker-$worker.json" \
        --worker-id "$worker" "${ARTIFACT_ARGS[@]}" \
        --output "$inputs/worker-$worker-device.json"
    ) >"$log" 2>&1 &
    probe_pids[$worker]="$!"
    child_pids+=("${probe_pids[$worker]}")
  done
  failures=0
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    if wait "${probe_pids[$worker]}"; then
      printf '[kpack-postfix-adaptive] PROBE_PASS worker=%s\n' "$worker"
    else
      printf '[kpack-postfix-adaptive] PROBE_FAIL worker=%s log=%s\n' \
        "$worker" "$logs/probe-worker-$worker.log" >&2
      failures=$((failures + 1))
    fi
  done
  child_pids=()
  [ "$failures" -eq 0 ] || { fail "$failures device probes failed"; return $?; }
  probe_args=()
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    probe_args+=(--probe "$inputs/worker-$worker-device.json")
  done
  python3 -B "$root/tools/kpack_postfix_adaptive_campaign.py" bind-devices \
    --output "$inputs" "${probe_args[@]}" || return 2

  run_pids=()
  child_pids=()
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    [ ! -L "$results/worker-$worker" ] || {
      fail "worker $worker result root is a symlink"; return $?; }
    resume_arg=()
    if [ -d "$results/worker-$worker" ]; then
      [ "$resume" = 1 ] || {
        fail "fresh worker $worker output already exists"; return $?; }
      resume_arg=(--resume)
    fi
    (
      artifact_args "$inputs/artifact-roots/worker-$worker.tsv" || return 2
      CUDA_VISIBLE_DEVICES="$worker" PPU_SDK="$sdk" \
        python3 -B "$root/tools/run_kpack_discovery_worker.py" run \
        --bundle "$inputs/catalog.json" --plan "$inputs/workload-plan.json" \
        --master "$inputs/master.json" --assignment "$inputs/assignment.json" \
        --selection "$inputs/selections/worker-$worker.json" \
        --worker-id "$worker" "${ARTIFACT_ARGS[@]}" \
        --device-identity "$inputs/worker-$worker-device.json" \
        --device-homogeneity "$inputs/device-homogeneity.json" \
        --output "$results/worker-$worker" --phase screen \
        --screen-iterations 2 --confirm-iterations 11 --confirm-rounds 3 \
        --correctness-repeats 1 --screen-warmups 1 --confirm-warmups 3 \
        --continue-on-atom-error "${resume_arg[@]}"
    ) >>"$logs/run-worker-$worker.log" 2>&1 &
    run_pids[$worker]="$!"
    child_pids+=("${run_pids[$worker]}")
    printf '[kpack-postfix-adaptive] SCREEN_STARTED worker=%s pid=%s\n' \
      "$worker" "${run_pids[$worker]}"
  done
  while :; do
    alive=0
    completed=0
    for ((worker = 0; worker < runtime_workers; ++worker)); do
      pid="${run_pids[$worker]}"
      kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
      [ -f "$results/worker-$worker/screen-completed.ids" ] && \
        completed=$((completed + 1))
    done
    printf '[kpack-postfix-adaptive] SCREEN_PROGRESS alive=%s/%s barrier_workers=%s/%s work_items=%s\n' \
      "$alive" "$runtime_workers" "$completed" "$runtime_workers" "$expected_items"
    [ "$alive" -gt 0 ] || break
    sleep 30
    [ "$interrupted_status" -eq 0 ] || return "$interrupted_status"
  done
  failures=0
  for ((worker = 0; worker < runtime_workers; ++worker)); do
    if wait "${run_pids[$worker]}"; then
      printf '[kpack-postfix-adaptive] SCREEN_PASS worker=%s\n' "$worker"
    else
      printf '[kpack-postfix-adaptive] SCREEN_FAIL worker=%s log=%s\n' \
        "$worker" "$logs/run-worker-$worker.log" >&2
      failures=$((failures + 1))
    fi
  done
  child_pids=()
  [ "$failures" -eq 0 ] || {
    fail "$failures screen workers failed; rerun the same OUT with RESUME=1"; return $?; }

  python3 -B "$root/tools/kpack_postfix_adaptive_campaign.py" seal-screen \
    --inputs "$inputs" --results "$results" || return 2
  trap - INT TERM
  printf '[kpack-postfix-adaptive] SCREEN_COMPLETE_CONFIRM_PENDING screen=2 correctness=1 screen_warmups=1 confirm=3x11 confirm_warmups=3 full_confirm_fallback=0 output=%s\n' \
    "$out"
}

main "$@"
