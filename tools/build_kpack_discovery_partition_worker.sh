#!/usr/bin/env bash
# Build one disjoint share of the exhaustive K-pack discovery partitions.
#
# Every machine uses the same immutable partition plan.  A deterministic
# greedy-LPT assignment keeps whole partitions on one worker while balancing
# their compiled-parent counts.  Builds use the machine-local fast disk; only
# completed, revalidated partition bundles are atomically published.

set -uo pipefail

fail() { printf '[kpack-build-worker] FAIL: %s\n' "$*" >&2; return 2; }

main() {
  [ "$#" -eq 0 ] || { fail "no positional arguments are accepted"; return $?; }
  local root local_parent local_root publish_root plan sdk worker_id worker_count jobs
  local source_sha partition_count preflight route partition out resume
  local publish_base publish_dir stage manifest list_path list_current
  local assigned_tasks

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 2
  worker_id="${WORKER_ID:-}"
  worker_count="${WORKER_COUNT:-}"
  plan="${KPACK_BUILD_PARTITION_PLAN:-}"
  publish_root="${KPACK_PARTITION_PUBLISH_ROOT:-}"
  local_root="${KPACK_PARTITION_LOCAL_ROOT:-}"
  jobs="${JOBS:-32}"
  sdk="${PPU_SDK:-${PPU_HOME:-}}"

  case "$worker_id:$worker_count:$jobs" in
    *[!0-9:]*) fail "WORKER_ID, WORKER_COUNT and JOBS must be integers"; return $?;;
    :*|*::*) fail "WORKER_ID and WORKER_COUNT are required"; return $?;;
  esac
  [ "$worker_count" -gt 0 ] && [ "$worker_id" -lt "$worker_count" ] && \
    [ "$jobs" -gt 0 ] || {
      fail "require 0 <= WORKER_ID < WORKER_COUNT and JOBS > 0"; return $?; }
  [ -n "$plan" ] && [ -n "$publish_root" ] && [ -n "$local_root" ] && [ -n "$sdk" ] || {
    fail "set KPACK_BUILD_PARTITION_PLAN, KPACK_PARTITION_PUBLISH_ROOT, KPACK_PARTITION_LOCAL_ROOT and PPU_SDK"
    return $?
  }
  plan="$(realpath -e -- "$plan")" || return 2
  sdk="$(realpath -e -- "$sdk")" || return 2
  [ -x "$sdk/bin/hgcc" ] && [ -x "$sdk/bin/hgobjdump" ] || {
    fail "PPU_SDK lacks hgcc/hgobjdump"; return $?; }

  local_parent="$(realpath -e /root/autodl-tmp)" || {
    fail "/root/autodl-tmp is required for machine-local build output"; return $?; }
  if [ -e "$local_root" ]; then
    [ -d "$local_root" ] && [ ! -L "$local_root" ] || {
      fail "KPACK_PARTITION_LOCAL_ROOT is not a regular directory"; return $?; }
    local_root="$(realpath -e -- "$local_root")" || return 2
  else
    case "$local_root" in "$local_parent"/*) ;; *)
      fail "KPACK_PARTITION_LOCAL_ROOT must be a strict /root/autodl-tmp child"; return $?;;
    esac
    mkdir -p "$local_root" || return 2
    local_root="$(realpath -e -- "$local_root")" || return 2
  fi
  case "$local_root" in "$local_parent"/*) ;; *)
    fail "KPACK_PARTITION_LOCAL_ROOT escaped /root/autodl-tmp"; return $?;;
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
      resume=0; [ ! -e "$out" ] || resume=1
      printf '[kpack-build-worker] worker=%s/%s route=%s partition=%s/%s resume=%s\n' \
        "$worker_id" "$worker_count" "$route" "$partition" "$partition_count" "$resume"
      if [ "$route" = scalefirst ]; then
        OUT="$out" RESUME="$resume" JOBS="$jobs" PPU_SDK="$sdk" \
          KPACK_BUILD_PARTITION_PLAN="$plan" KPACK_BUILD_PARTITION_ID="$partition" \
          KPACK_GLOBAL_PREFLIGHT_RECEIPT="$preflight" \
          bash "$root/tools/build_scalefirst_kpack_discovery_bundle.sh" || return $?
      else
        OUT="$out" RESUME="$resume" JOBS="$jobs" PPU_SDK="$sdk" \
          KPACK_BUILD_PARTITION_PLAN="$plan" KPACK_BUILD_PARTITION_ID="$partition" \
          KPACK_GLOBAL_PREFLIGHT_RECEIPT="$preflight" \
          bash "$root/tools/build_fully_quantized_kpack_discovery_bundle.sh" || return $?
      fi
      manifest="$out/partition-bundle.json"
      PPU_SDK="$sdk" python3 -B "$root/tools/kpack_discovery_build_partitions.py" verify \
        --root "$out" --manifest "$manifest" || return 2

      publish_dir="$publish_base/$route/p$(printf '%02d' "$partition")"
      if [ -e "$publish_dir" ]; then
        [ -d "$publish_dir" ] && [ ! -L "$publish_dir" ] || {
          fail "published partition path is not a regular directory: $publish_dir"; return $?; }
        PPU_SDK="$sdk" python3 -B "$root/tools/kpack_discovery_build_partitions.py" verify \
          --root "$publish_dir" --manifest "$publish_dir/partition-bundle.json" || return 2
        cmp -s "$manifest" "$publish_dir/partition-bundle.json" || {
          fail "published partition authority differs: $publish_dir"; return $?; }
        continue
      fi
      stage="$publish_base/$route/.p$(printf '%02d' "$partition").worker-$worker_id.current.$$"
      [ ! -e "$stage" ] || { fail "publish staging path already exists: $stage"; return $?; }
      cp -a "$out" "$stage" || return 2
      PPU_SDK="$sdk" python3 -B "$root/tools/kpack_discovery_build_partitions.py" verify \
        --root "$stage" --manifest "$stage/partition-bundle.json" || return 2
      mv "$stage" "$publish_dir" || return 2
      printf '[kpack-build-worker] PUBLISHED %s\n' "$publish_dir"
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

main "$@"
