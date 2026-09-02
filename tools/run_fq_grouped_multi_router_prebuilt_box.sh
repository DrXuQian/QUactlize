#!/usr/bin/env bash
# Execute an exact prebuilt grouped multi-router bundle; this runner never builds.
set -uo pipefail

fail() {
  printf '[fq-grouped-multi-router] FAIL: %s\n' "$*" >&2
  return 2
}

main() {
  [ "$#" -eq 0 ] || { fail 'no positional arguments'; return 2; }
  local root workspace out bundle manifest plan iterations warmups rounds sdk release
  local q round binary log rc
  local -a case_args
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-grouped-multi-router-$(date -u +%Y%m%dT%H%M%SZ)-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  [ ! -e "$out" ] || { fail 'OUT already exists'; return 2; }
  [ "${RESUME:-0}" = 0 ] || { fail 'RESUME is not implemented'; return 2; }
  case "${CUDA_VISIBLE_DEVICES:-}" in
    ''|*,*|*[!0-9]*) fail 'CUDA_VISIBLE_DEVICES must name exactly one ordinal'; return 2;;
  esac
  bundle="$(realpath -e -- "${FQ_GROUPED_MULTI_ROUTER_BUNDLE:-/nonexistent}")" || {
    fail 'exact prebuilt bundle is required'; return 2;
  }
  case "$bundle" in "$workspace"/*) ;; *) fail 'bundle must be a strict /workspace child'; return 2;; esac
  manifest="$bundle/manifest.json"
  [ -f "$manifest" ] && [ ! -L "$manifest" ] || { fail 'manifest is missing/symlinked'; return 2; }
  iterations="${PERF_ITERATIONS:-11}"
  warmups="${PERF_WARMUPS:-3}"
  rounds="${PERF_ROUNDS:-3}"
  case "$iterations:$warmups:$rounds" in
    *[!0-9:]*|0:*|*:0:*|*:*:0) fail 'timing controls must be positive integers'; return 2;;
  esac
  sdk="${PPU_SDK:-${PPU_HOME:-}}"
  [ -d "$sdk" ] && [ ! -L "$sdk" ] || { fail 'real box PPU SDK is required'; return 2; }
  release="$sdk/release.yaml"
  [ -f "$release" ] || { fail 'box SDK release.yaml is required'; return 2; }
  mkdir -p "$out/inputs" "$out/runs" "$out/results" || return 2
  plan="$out/inputs/plan.json"
  python3 -B "$root/tools/probe_box_identity.py" resolve \
    --output "$out/inputs/box-identity.json" || {
      fail 'runtime one-device identity probe failed'; return 2;
    }
  python3 -B - "$out/inputs/box-identity.json" <<'PY' || {
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    probe = json.load(stream)["device_probe"]
assert probe["status"] in ("measured", "properties-unavailable")
assert probe["device_count"] == 1
PY
    fail 'measured runtime one-device evidence is required'
    return 2
  }
  python3 -B "$root/tools/plan_fq_grouped_multi_router.py" self-test || return 2
  python3 -B "$root/tools/analyze_fq_grouped_multi_router.py" self-test || return 2
  python3 -B "$root/tools/plan_fq_grouped_multi_router.py" materialize \
    --output "$plan" || return 2
  mapfile -t case_args < <(python3 -B - "$plan" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1]))
for name, row in plan["routers"].items():
    values = [
        name, "512", "2048", str(row["total_rows"]), str(row["max_rows"]),
        str(row["active"]), str(row["zero"]), str(row["work_tm16"]),
        str(row["work_tm32"]), str(row["work_tm128"]),
        row["rows_hash"],
        ",".join(map(str, row["rows"])),
    ]
    print("--case=" + ":".join(values))
PY
  ) || return 2
  [ "${#case_args[@]}" -eq 6 ] || { fail 'plan-to-profile denominator differs'; return 2; }
  python3 -B "$root/tools/fq_grouped_multi_router_manifest.py" self-test || return 2
  python3 -B "$root/tools/fq_grouped_multi_router_manifest.py" verify \
    --bundle "$bundle" --manifest "$manifest" --source-root "$root" \
    --release "$release" || return 2
  git -C "$root" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain)"' || {
      fail 'box recursive submodule worktree is dirty'
      return 2
    }
  cp -- "$manifest" "$out/inputs/bundle-manifest.json" || return 2
  manifest="$out/inputs/bundle-manifest.json"
  {
    printf 'runner_source=%s\n' "$(git -C "$root" rev-parse HEAD)"
    printf 'device_ordinal=%s\n' "$CUDA_VISIBLE_DEVICES"
    sha256sum "$manifest" "$plan" "$out/inputs/box-identity.json" \
      "$root/tools/probe_box_identity.py" \
      "$root/tools/fq_grouped_multi_router.py" \
      "$root/tools/plan_fq_grouped_multi_router.py" \
      "$root/tools/analyze_fq_grouped_multi_router.py" \
      "$root/tools/fq_grouped_multi_router_manifest.py" \
      "$root/tools/run_fq_grouped_multi_router_prebuilt_box.sh"
  } > "$out/inputs/result-authority.txt" || return 2
  for q in 10 11 12 13 14; do
    binary="$(python3 -B - "$bundle" "$manifest" "$q" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
row = json.load(open(sys.argv[2]))["binaries"][sys.argv[3]]
print((root / row["path"]).resolve())
PY
    )" || return 2
    for round in $(seq 1 "$rounds"); do
      log="$out/runs/q$q-round$round.log"
      LD_LIBRARY_PATH="$(dirname "$binary")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$binary" --round="$round" --iterations="$iterations" \
        --warmups="$warmups" "${case_args[@]}" > "$log" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
        tail -n 120 "$log" >&2
        fail "q$q round$round rc=$rc"
        return "$rc"
      fi
      grep -Fqx \
        "FQ_GROUPED_ROUTER_RUN schema=grouped-kpack-multi-router-v1 q=$q round=$round layout=kpack iterations=$iterations warmups=$warmups cells=6 status=PASS" \
        "$log" || { fail 'completion marker differs'; return 2; }
      sha256sum "$log" > "$log.sha256" || return 2
    done
  done
  python3 -B "$root/tools/analyze_fq_grouped_multi_router.py" analyze \
    --plan "$plan" --runs "$out/runs" --output "$out/results" \
    --rounds "$rounds" --iterations "$iterations" --warmups "$warmups" || return 2
  printf '[fq-grouped-multi-router] DIAGNOSTIC_COMPLETE artifacts=%s\n' "$out"
}

main "$@"
