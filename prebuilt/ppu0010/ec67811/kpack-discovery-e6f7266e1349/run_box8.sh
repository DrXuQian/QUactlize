#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[ec67811-box8] FAIL: %s\n' "$*" >&2
  exit 2
}

runner_root="$(realpath -e -- "${QUACTLIZE_ROOT:-/workspace/quactlize-runner-5919afa}")" ||
  fail 'set QUACTLIZE_ROOT to the pinned runner checkout'
campaign="$(realpath -e -- "${CAMPAIGN:-/workspace/campaign-ec67811}")" ||
  fail 'set CAMPAIGN to the extracted campaign directory'
run="$(realpath -m -- "${RUN:-/workspace/kpack-discovery-ec67811}")" ||
  fail 'cannot resolve RUN'
sdk="$(realpath -e -- "${PPU_SDK:-/nonexistent}")" ||
  fail 'set PPU_SDK to the compatible SDK root'
correctness_repeats="${CORRECTNESS_REPEATS:-256}"
case "$correctness_repeats" in
  ''|*[!0-9]*) fail 'CORRECTNESS_REPEATS must be a positive integer' ;;
esac
test "$correctness_repeats" -ge 1 ||
  fail 'CORRECTNESS_REPEATS must be a positive integer'

source_sha=ec67811bd709eace941daf3c650d45df574b1a87
runner_sha=5919afa07d57ecb21bc2a5c73ce5b78f5c929648
catalog="$campaign/control/distributed-catalog.json"
execution="$campaign/control/execution-8"
plan="$execution/route-plan.json"
master="$execution/master.json"
assignment="$execution/assignment.json"
published="$campaign/published/$source_sha"

test "$(git -C "$runner_root" rev-parse HEAD)" = "$runner_sha" ||
  fail "runner checkout must be exactly $runner_sha"
test -x "$sdk/bin/hgobjdump" || fail 'PPU_SDK lacks bin/hgobjdump'
test -s "$catalog" || fail 'distributed catalog is missing'
test -s "$assignment" || fail '8-worker assignment is missing'
test -d "$published" || fail 'published payload tree is missing'
case "$run" in /workspace/*) ;; *) fail 'RUN must be below /workspace';; esac

# shellcheck source=/dev/null
source "$sdk/envsetup.sh"
export PPU_SDK="$sdk"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib64:$sdk/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$runner_root"
mkdir -p "$run"/{devices,logs,workers}

python3 -B tools/kpack_discovery_build_partitions.py validate-catalog \
  --catalog "$catalog"
python3 -B tools/kpack_discovery_worker_plan.py validate-assignment \
  --bundle "$catalog" --plan "$plan" --master "$master" \
  --assignment "$assignment"

load_artifact_args() {
  local selection="$1"
  mapfile -t artifact_root_args < <(
    python3 -B - "$catalog" "$selection" "$published" <<'PY'
import json
import pathlib
import sys

catalog_path, selection_path, published = map(pathlib.Path, sys.argv[1:])
catalog = json.loads(catalog_path.read_text())
selection = json.loads(selection_path.read_text())
by_id = {row["artifact_id"]: row for row in catalog["partitions"]}
if not selection["artifact_ids"] or not set(selection["artifact_ids"]) <= set(by_id):
    raise SystemExit("selection artifact IDs differ from catalog")
for artifact_id in selection["artifact_ids"]:
    row = by_id[artifact_id]
    root = (published / row["route"] / f"p{row['partition_id']:02d}").resolve(strict=True)
    print("--artifact-root")
    print(f"{artifact_id}={root}")
PY
  )
  test "${#artifact_root_args[@]}" -eq 16 ||
    fail "$(basename "$selection") does not own exactly eight artifacts"
}

probe_pids=()
for worker in $(seq 0 7); do
  selection="$execution/selections/worker-$worker.json"
  (
    load_artifact_args "$selection"
    CUDA_VISIBLE_DEVICES="$worker" \
      python3 -B tools/run_kpack_discovery_worker.py probe-device \
        --bundle "$catalog" --plan "$plan" --master "$master" \
        --assignment "$assignment" --selection "$selection" \
        --worker-id "$worker" --output "$run/devices/worker-$worker.json" \
        "${artifact_root_args[@]}"
  ) >"$run/logs/probe-$worker.log" 2>&1 &
  probe_pids+=("$!")
done

probe_rc=0
for pid in "${probe_pids[@]}"; do
  wait "$pid" || probe_rc=1
done
if test "$probe_rc" -ne 0; then
  tail -80 "$run"/logs/probe-*.log >&2
  fail 'one or more device probes failed'
fi

python3 -B - "$runner_root" "$run" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

runner, run = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(runner / "tools"))
import box_identity_schema

workers = []
common = []
for worker in range(8):
    path = run / "devices" / f"worker-{worker}.json"
    raw = path.read_bytes()
    doc = json.loads(raw)
    values, _ = box_identity_schema.values_and_sources(doc)
    probe = doc["device_probe"]
    if probe["status"] != "measured" or len(probe["candidates"]) != 1:
        raise SystemExit(f"worker {worker} has no unique measured device")
    candidate = probe["candidates"][0]
    common.append({
        "device_model": values["device_model"],
        "compute_capability": candidate["compute_capability"],
        "compute_units": candidate["compute_units"],
        "driver_version": values["driver_version"],
        "sdk_compiler_identity": values["sdk_compiler_identity"],
    })
    workers.append({
        "worker_id": worker,
        "identity_sha256": hashlib.sha256(raw).hexdigest(),
    })

canonical = {
    json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    for value in common
}
if len(canonical) != 1:
    raise SystemExit("the eight selected devices are not homogeneous")
if len({row["identity_sha256"] for row in workers}) != 8:
    raise SystemExit("device identity evidence was reused")
homogeneity_key = hashlib.sha256(next(iter(canonical)).encode("ascii")).hexdigest()
for row in workers:
    row["homogeneity_key"] = homogeneity_key
document = {
    "schema": "quactlize.kpack-discovery-device-homogeneity.v1",
    "workers": workers,
}
encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
output = run / "device-homogeneity.json"
if output.exists():
    if output.read_text() != encoded:
        raise SystemExit("resumed device-homogeneity authority differs")
else:
    temporary = output.with_name(f".{output.name}.current.{os.getpid()}")
    temporary.write_text(encoded)
    os.replace(temporary, output)
PY

for worker in $(seq 0 7); do
  selection="$execution/selections/worker-$worker.json"
  output="$run/workers/worker-$worker"
  pid_file="$run/worker-$worker.pid"
  if test -s "$pid_file" && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    printf '[ec67811-box8] worker=%d already-running pid=%s\n' \
      "$worker" "$(cat "$pid_file")"
    continue
  fi
  load_artifact_args "$selection"
  resume=()
  test ! -d "$output" || resume=(--resume)
  nohup env CUDA_VISIBLE_DEVICES="$worker" PYTHONUNBUFFERED=1 \
    python3 -B tools/run_kpack_discovery_worker.py run \
      --bundle "$catalog" --plan "$plan" --master "$master" \
      --assignment "$assignment" --selection "$selection" \
      --worker-id "$worker" \
      --device-identity "$run/devices/worker-$worker.json" \
      --device-homogeneity "$run/device-homogeneity.json" \
      --output "$output" --phase all --screen-iterations 5 \
      --confirm-iterations 11 --confirm-rounds 3 \
      --correctness-repeats "$correctness_repeats" --warmups 3 \
      "${resume[@]}" "${artifact_root_args[@]}" \
      >>"$run/logs/worker-$worker.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf '[ec67811-box8] worker=%d started pid=%d\n' "$worker" "$pid"
done

sleep 3
for worker in $(seq 0 7); do
  pid="$(cat "$run/worker-$worker.pid")"
  kill -0 "$pid" 2>/dev/null || {
    tail -80 "$run/logs/worker-$worker.log" >&2
    fail "worker $worker exited during startup"
  }
done
printf '[ec67811-box8] STARTED workers=8 correctness_repeats=%s run=%s\n' \
  "$correctness_repeats" "$run"
