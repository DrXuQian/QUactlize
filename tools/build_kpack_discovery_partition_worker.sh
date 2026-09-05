#!/usr/bin/env bash
# Build one disjoint share of the exhaustive K-pack discovery partitions.
#
# Every machine uses the same immutable partition plan.  A deterministic
# greedy-LPT assignment keeps whole partitions on one worker while balancing
# their compiled-parent counts.  Builds use the machine-local fast disk; only
# completed, revalidated partition bundles are atomically published.  Set
# KPACK_REMOVE_LOCAL_AFTER_PUBLISH=1 to remove each verified local partition
# after its published manifest is proved byte-identical; the default is 0.
# KPACK_PAYLOAD_SOURCE_ROOT may point at the immutable source checkout named by
# an existing build authority while this script and its validators come from a
# later, clean repair checkout.

set -uo pipefail

fail() { printf '[kpack-build-worker] FAIL: %s\n' "$*" >&2; return 2; }

verify_published_partition() {
  [ "$#" -eq 3 ] || {
    fail "internal published verification argument count differs"; return $?; }
  local repo_root="$1" sdk_root="$2" publish_dir="$3"
  [ -d "$publish_dir" ] && [ ! -L "$publish_dir" ] && \
    [ -f "$publish_dir/partition-bundle.json" ] && \
    [ ! -L "$publish_dir/partition-bundle.json" ] || {
      fail "published partition is not one regular verified tree: $publish_dir"; return $?; }
  PPU_SDK="$sdk_root" python3 -B \
    "$repo_root/tools/kpack_discovery_build_partitions.py" verify \
    --root "$publish_dir" --manifest "$publish_dir/partition-bundle.json"
}

verify_published_partition_and_maybe_remove_local() {
  [ "$#" -eq 8 ] || {
    fail "internal publish verification argument count differs"; return $?; }
  local repo_root="$1" sdk_root="$2" local_root="$3" out="$4"
  local publish_dir="$5" route="$6" partition="$7" remove_local="$8"
  local partition_number partition_tag expected_out resolved_local resolved_out link_path

  [ "$route" = scalefirst ] || [ "$route" = fully-quantized ] || {
    fail "refusing cleanup for an unknown route"; return $?; }
  case "$partition" in ''|*[!0-9]*)
    fail "refusing cleanup for a malformed partition"; return $?;; esac
  case "$remove_local" in 0|1) ;; *)
    fail "KPACK_REMOVE_LOCAL_AFTER_PUBLISH must be 0 or 1"; return $?;; esac

  verify_published_partition "$repo_root" "$sdk_root" "$publish_dir" || return 2

  [ -d "$local_root" ] && [ ! -L "$local_root" ] || {
    fail "local partition root is not a regular directory"; return $?; }
  resolved_local="$(realpath -e -- "$local_root")" || return 2
  [ "$resolved_local" = "$local_root" ] || {
    fail "local partition root is not canonical"; return $?; }
  partition_number=$((10#$partition))
  printf -v partition_tag '%02d' "$partition_number"
  expected_out="$resolved_local/$route-p$partition_tag"
  [ "$out" = "$expected_out" ] || {
    fail "local partition output is not its exact expected child: $out"; return $?; }
  [ -d "$out" ] && [ ! -L "$out" ] && \
    [ -f "$out/partition-bundle.json" ] && \
    [ ! -L "$out/partition-bundle.json" ] || {
      fail "local partition output is not one regular tree: $out"; return $?; }
  resolved_out="$(realpath -e -- "$out")" || return 2
  [ "$resolved_out" = "$expected_out" ] || {
    fail "local partition output escaped its exact expected child"; return $?; }
  case "$resolved_out" in "$resolved_local"/*) ;; *)
    fail "local partition output escaped its local root"; return $?;; esac

  PPU_SDK="$sdk_root" python3 -B \
    "$repo_root/tools/kpack_discovery_build_partitions.py" verify \
    --root "$resolved_out" --manifest "$resolved_out/partition-bundle.json" || return 2
  cmp -s "$resolved_out/partition-bundle.json" \
    "$publish_dir/partition-bundle.json" || {
      fail "published partition authority differs: $publish_dir"; return $?; }
  [ "$remove_local" = 1 ] || return 0

  link_path="$(find "$resolved_out" -xdev -type l -print -quit)" || {
    fail "cannot audit local partition tree before cleanup"; return $?; }
  [ -z "$link_path" ] || {
    fail "refusing local cleanup because the tree contains a symlink: $link_path"; return $?; }
  # Recheck the compared authority after the tree audit, immediately before
  # the destructive operation.  The target is canonical, non-symlink, and an
  # exact named child of the already-canonical local root.
  cmp -s "$resolved_out/partition-bundle.json" \
    "$publish_dir/partition-bundle.json" || {
      fail "published partition authority changed before cleanup"; return $?; }
  rm -r --one-file-system -- "${resolved_out:?}" || return 2
  [ ! -e "$resolved_out" ] && [ ! -L "$resolved_out" ] || {
    fail "local partition cleanup was incomplete: $resolved_out"; return $?; }
  printf '[kpack-build-worker] REMOVED_LOCAL %s\n' "$resolved_out"
}

main() {
  [ "$#" -eq 0 ] || { fail "no positional arguments are accepted"; return $?; }
  local tool_root root local_parent local_root publish_root plan sdk worker_id worker_count jobs
  local requested_local_parent
  local source_sha partition_count preflight route partition out resume remove_local
  local publish_base publish_dir stage manifest list_path list_current
  local assigned_tasks

  tool_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 2
  root="$(realpath -e -- "${KPACK_PAYLOAD_SOURCE_ROOT:-$tool_root}")" || {
    fail "KPACK_PAYLOAD_SOURCE_ROOT is not a readable source checkout"; return $?; }
  [ -d "$root/.git" ] || [ -f "$root/.git" ] || {
    fail "KPACK_PAYLOAD_SOURCE_ROOT is not a Git worktree"; return $?; }
  worker_id="${WORKER_ID:-}"
  worker_count="${WORKER_COUNT:-}"
  plan="${KPACK_BUILD_PARTITION_PLAN:-}"
  publish_root="${KPACK_PARTITION_PUBLISH_ROOT:-}"
  local_root="${KPACK_PARTITION_LOCAL_ROOT:-}"
  jobs="${JOBS:-32}"
  sdk="${PPU_SDK:-${PPU_HOME:-}}"
  remove_local="${KPACK_REMOVE_LOCAL_AFTER_PUBLISH:-0}"

  case "$worker_id:$worker_count:$jobs" in
    *[!0-9:]*) fail "WORKER_ID, WORKER_COUNT and JOBS must be integers"; return $?;;
    :*|*::*) fail "WORKER_ID and WORKER_COUNT are required"; return $?;;
  esac
  [ "$worker_count" -gt 0 ] && [ "$worker_id" -lt "$worker_count" ] && \
    [ "$jobs" -gt 0 ] || {
      fail "require 0 <= WORKER_ID < WORKER_COUNT and JOBS > 0"; return $?; }
  case "$remove_local" in 0|1) ;; *)
    fail "KPACK_REMOVE_LOCAL_AFTER_PUBLISH must be 0 or 1"; return $?;; esac
  [ -n "$plan" ] && [ -n "$publish_root" ] && [ -n "$local_root" ] && [ -n "$sdk" ] || {
    fail "set KPACK_BUILD_PARTITION_PLAN, KPACK_PARTITION_PUBLISH_ROOT, KPACK_PARTITION_LOCAL_ROOT and PPU_SDK"
    return $?
  }
  plan="$(realpath -e -- "$plan")" || return 2
  sdk="$(realpath -e -- "$sdk")" || return 2
  [ -x "$sdk/bin/hgcc" ] && [ -x "$sdk/bin/hgobjdump" ] || {
    fail "PPU_SDK lacks hgcc/hgobjdump"; return $?; }

  requested_local_parent="${KPACK_LOCAL_SCRATCH_ROOT:-/root/autodl-tmp}"
  [ -d "$requested_local_parent" ] && [ ! -L "$requested_local_parent" ] || {
    fail "KPACK_LOCAL_SCRATCH_ROOT must be a regular non-symlink directory"; return $?; }
  local_parent="$(realpath -e -- "$requested_local_parent")" || {
    fail "KPACK_LOCAL_SCRATCH_ROOT is not a readable directory"; return $?; }
  case "$local_parent" in /|/root|/workspace)
    fail "KPACK_LOCAL_SCRATCH_ROOT is too broad"; return $?;; esac
  if [ -e "$local_root" ]; then
    [ -d "$local_root" ] && [ ! -L "$local_root" ] || {
      fail "KPACK_PARTITION_LOCAL_ROOT is not a regular directory"; return $?; }
    local_root="$(realpath -e -- "$local_root")" || return 2
  else
    case "$local_root" in "$local_parent"/*) ;; *)
      fail "KPACK_PARTITION_LOCAL_ROOT must be a strict configured scratch child"; return $?;;
    esac
    mkdir -p "$local_root" || return 2
    local_root="$(realpath -e -- "$local_root")" || return 2
  fi
  case "$local_root" in "$local_parent"/*) ;; *)
    fail "KPACK_PARTITION_LOCAL_ROOT escaped the configured scratch root"; return $?;;
  esac

  if [ -e "$publish_root" ]; then
    [ -d "$publish_root" ] && [ ! -L "$publish_root" ] || {
      fail "KPACK_PARTITION_PUBLISH_ROOT is not a regular directory"; return $?; }
    publish_root="$(realpath -e -- "$publish_root")" || return 2
  else
    mkdir -p "$publish_root" || return 2
    publish_root="$(realpath -e -- "$publish_root")" || return 2
  fi
  [ "$publish_root" != / ] && [ "$publish_root" != /root ] || {
    fail "refusing broad publish root"; return $?; }

  source_sha="$(git -C "$root" rev-parse HEAD)" || return 2
  partition_count="$(python3 -B - "$root" "$plan" <<'PY'
import pathlib,sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "tools"))
import kpack_discovery_build_partitions as p
print(p.read_plan(pathlib.Path(sys.argv[2]))["partition_count"])
PY
  )" || return 2
  case "$partition_count" in ''|*[!0-9]*)
    fail "partition plan did not expose an integer count"; return $?;; esac
  [ "$worker_count" -le "$partition_count" ] || {
    fail "WORKER_COUNT exceeds partition count"; return $?; }
  assigned_tasks="$(python3 -B - "$root" "$plan" "$worker_id" "$worker_count" <<'PY'
import pathlib,sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "tools"))
import kpack_discovery_build_partitions as p
plan=p.read_plan(pathlib.Path(sys.argv[2])); wanted=int(sys.argv[3]); workers=int(sys.argv[4])
loads=[0]*workers; assignments=[[] for _ in range(workers)]
tasks=[(route,int(row["partition_id"]),int(row["parents_by_route"][route]))
       for row in plan["partitions"] for route in ("scalefirst","fully-quantized")]
for task in sorted(tasks, key=lambda item: (-item[2],item[0],item[1])):
    worker=min(range(workers), key=lambda item: (loads[item], item))
    assignments[worker].append(task); loads[worker]+=task[2]

# Greedy LPT is close; deterministic one-move/one-swap descent removes the
# avoidable tail caused by the 64 route-partition weights without splitting an
# artifact.  Only strict improvements are accepted, so this always terminates.
def objective(values): return (max(values)-min(values),max(values),tuple(values))
while True:
    current=objective(loads); best=(current,None)
    for left in range(workers):
        for right in range(workers):
            if left==right: continue
            for li,lhs in enumerate(assignments[left]):
                trial=loads.copy(); trial[left]-=lhs[2]; trial[right]+=lhs[2]
                candidate=(objective(trial),("move",left,right,li,trial))
                if candidate[0] < best[0]: best=candidate
    for left in range(workers):
        for right in range(left+1,workers):
            for li,lhs in enumerate(assignments[left]):
                for ri,rhs in enumerate(assignments[right]):
                    trial=loads.copy(); trial[left]+=rhs[2]-lhs[2]; trial[right]+=lhs[2]-rhs[2]
                    candidate=(objective(trial),("swap",left,right,li,ri,trial))
                    if candidate[0] < best[0]: best=candidate
    if best[1] is None: break
    action=best[1]
    if action[0]=="move":
        _,left,right,li,loads=action
        assignments[right].append(assignments[left].pop(li))
    else:
        _,left,right,li,ri,loads=action
        assignments[left][li],assignments[right][ri]=assignments[right][ri],assignments[left][li]
for route,partition,_weight in sorted(assignments[wanted]): print(f"{route}\t{partition}")
PY
  )" || return 2
  [ -n "$assigned_tasks" ] || {
    fail "worker received no build partitions"; return $?; }

  preflight="$local_root/kpack-global-preflight.json"
  if [ -e "$preflight" ]; then
    [ -f "$preflight" ] && [ ! -L "$preflight" ] || {
      fail "global preflight receipt is not a regular file"; return $?; }
    python3 -B "$root/tools/kpack_global_build_preflight.py" verify \
      --root "$root" --receipt "$preflight" || return 2
  else
    python3 -B "$root/tools/kpack_global_build_preflight.py" create \
      --root "$root" --output "$preflight" || return 2
  fi

  publish_base="$publish_root/$source_sha"
  mkdir -p "$publish_base/scalefirst" "$publish_base/fully-quantized" || return 2

  while IFS=$'\t' read -r route partition; do
      [ "$route" = scalefirst ] || [ "$route" = fully-quantized ] || {
        fail "assignment contains an unknown route"; return $?; }
      case "$partition" in ''|*[!0-9]*)
        fail "assignment contains a malformed partition"; return $?;; esac
      out="$local_root/$route-p$(printf '%02d' "$partition")"
      publish_dir="$publish_base/$route/p$(printf '%02d' "$partition")"
      if [ -e "$publish_dir" ] || [ -L "$publish_dir" ]; then
        verify_published_partition "$tool_root" "$sdk" "$publish_dir" || return 2
        if [ -e "$out" ] || [ -L "$out" ]; then
          verify_published_partition_and_maybe_remove_local \
            "$tool_root" "$sdk" "$local_root" "$out" "$publish_dir" \
            "$route" "$partition" "$remove_local" || return $?
        fi
        printf '[kpack-build-worker] REUSED_PUBLISHED %s\n' "$publish_dir"
        continue
      fi
      resume=0; [ ! -e "$out" ] || resume=1
      printf '[kpack-build-worker] worker=%s/%s route=%s partition=%s/%s resume=%s\n' \
        "$worker_id" "$worker_count" "$route" "$partition" "$partition_count" "$resume"
      if [ "$route" = scalefirst ]; then
        OUT="$out" RESUME="$resume" JOBS="$jobs" PPU_SDK="$sdk" \
          KPACK_LOCAL_SCRATCH_ROOT="$local_parent" \
          KPACK_BUILD_PARTITION_PLAN="$plan" KPACK_BUILD_PARTITION_ID="$partition" \
          KPACK_GLOBAL_PREFLIGHT_RECEIPT="$preflight" \
          bash "$root/tools/build_scalefirst_kpack_discovery_bundle.sh" || return $?
      else
        OUT="$out" RESUME="$resume" JOBS="$jobs" PPU_SDK="$sdk" \
          KPACK_LOCAL_SCRATCH_ROOT="$local_parent" \
          FQ_KPACK_PAYLOAD_SOURCE_ROOT="$root" \
          KPACK_BUILD_PARTITION_PLAN="$plan" KPACK_BUILD_PARTITION_ID="$partition" \
          KPACK_GLOBAL_PREFLIGHT_RECEIPT="$preflight" \
          bash "$tool_root/tools/build_fully_quantized_kpack_discovery_bundle.sh" || return $?
      fi
      manifest="$out/partition-bundle.json"
      PPU_SDK="$sdk" python3 -B "$tool_root/tools/kpack_discovery_build_partitions.py" verify \
        --root "$out" --manifest "$manifest" || return 2

      if [ -e "$publish_dir" ] || [ -L "$publish_dir" ]; then
        verify_published_partition_and_maybe_remove_local \
          "$tool_root" "$sdk" "$local_root" "$out" "$publish_dir" \
          "$route" "$partition" "$remove_local" || return $?
        continue
      fi
      stage="$publish_base/$route/.p$(printf '%02d' "$partition").worker-$worker_id.current.$$"
      [ ! -e "$stage" ] || { fail "publish staging path already exists: $stage"; return $?; }
      cp -a "$out" "$stage" || return 2
      PPU_SDK="$sdk" python3 -B "$tool_root/tools/kpack_discovery_build_partitions.py" verify \
        --root "$stage" --manifest "$stage/partition-bundle.json" || return 2
      mv "$stage" "$publish_dir" || return 2
      printf '[kpack-build-worker] PUBLISHED %s\n' "$publish_dir"
      verify_published_partition_and_maybe_remove_local \
        "$tool_root" "$sdk" "$local_root" "$out" "$publish_dir" \
        "$route" "$partition" "$remove_local" || return $?
  done <<<"$assigned_tasks"

  list_path="$publish_base/worker-$worker_id-of-$worker_count.manifests"
  list_current="$publish_base/.worker-$worker_id-of-$worker_count.manifests.current.$$"
  : >"$list_current" || return 2
  while IFS=$'\t' read -r route partition; do
      manifest="$publish_base/$route/p$(printf '%02d' "$partition")/partition-bundle.json"
      [ -f "$manifest" ] && [ ! -L "$manifest" ] || {
        fail "published manifest is missing: $manifest"; return $?; }
      printf '%s\n' "$manifest" >>"$list_current" || return 2
  done <<<"$assigned_tasks"
  if [ -e "$list_path" ]; then
    cmp -s "$list_current" "$list_path" || {
      fail "worker completion manifest differs: $list_path"; return $?; }
    [ -f "$list_current" ] && [ ! -L "$list_current" ] && \
      [ "$(dirname -- "$list_current")" = "$publish_base" ] || {
        fail "worker completion staging file escaped publish directory"; return $?; }
    unlink "$list_current" || return 2
  else
    mv "$list_current" "$list_path" || return 2
  fi
  printf '[kpack-build-worker] COMPLETE worker=%s/%s manifests=%s list=%s\n' \
    "$worker_id" "$worker_count" "$(wc -l <"$list_path")" "$list_path"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
