#!/usr/bin/env bash
# Execute the prebuilt FQ Split-K reducer census. This script never builds.
set -euo pipefail

fail() {
  printf '[fq-splitk-reducer-box] FAIL: %s\n' "$*" >&2
  exit 2
}

main() {
  [[ $# = 0 ]] || fail 'no positional arguments are accepted'
  local root artifact_root out bundle_input bundle sdk identity_input homogeneity_input
  local worker_id manifest plan binary probe identity homogeneity execution round seed log temp rc

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  artifact_root="$(realpath -e -- "${FQ_SPLITK_REDUCER_ROOT:-/workspace}")" ||
    fail 'artifact root is absent'
  out="$(realpath -m -- "${OUT:-$artifact_root/fq-splitk-reducer-run-$(date -u +%Y%m%dT%H%M%SZ)-$$}")"
  case "$out" in "$artifact_root"/*) ;; *) fail 'OUT must be a strict artifact-root child' ;; esac
  [[ ! -e "$out" && ! -L "$out" ]] || fail 'OUT already exists'
  case "${CUDA_VISIBLE_DEVICES:-}" in
    ''|*,*|*[!0-9]*) fail 'CUDA_VISIBLE_DEVICES must name exactly one numeric ordinal' ;;
  esac
  [[ "${RESUME:-0}" = 0 ]] || fail 'RESUME is not implemented'
  [[ -z "${PPU_DEFS:-}${PPU_EXTRA_DEFS:-}" ]] ||
    fail 'ambient build definitions are forbidden'
  for variable in PERF_ITERATIONS PERF_WARMUPS PERF_ROUNDS FQ_REDUCER_CASE_BEGIN FQ_REDUCER_CASE_END; do
    [[ -z "${!variable:-}" ]] || fail "$variable cannot change the frozen census"
  done

  bundle_input="${FQ_SPLITK_REDUCER_BUNDLE:-/nonexistent}"
  [[ -d "$bundle_input" && ! -L "$bundle_input" ]] ||
    fail 'a plain prebuilt bundle directory is required'
  bundle="$(realpath -e -- "$bundle_input")"
  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" ||
    fail 'the exact build SDK is required'
  identity_input="$(realpath -e -- "${KPACK_DEVICE_IDENTITY:-/nonexistent}")" ||
    fail 'KPACK_DEVICE_IDENTITY is required'
  homogeneity_input="$(realpath -e -- "${KPACK_DEVICE_HOMOGENEITY:-/nonexistent}")" ||
    fail 'KPACK_DEVICE_HOMOGENEITY is required'
  [[ -f "$identity_input" && ! -L "$identity_input" &&
     -f "$homogeneity_input" && ! -L "$homogeneity_input" ]] ||
    fail 'device identity/homogeneity must be plain files'
  worker_id="${KPACK_WORKER_ID:-}"
  [[ "$worker_id" =~ ^[0-9]+$ ]] || fail 'KPACK_WORKER_ID must be nonnegative'

  mkdir -p "$out/inputs" "$out/runs"
  python3 -B "$root/tools/plan_fq_splitk_reducer_lookup.py" self-test
  python3 -B "$root/tools/analyze_fq_splitk_reducer_lookup.py" self-test
  python3 -B "$root/tools/fq_splitk_reducer_prebuilt.py" verify \
    --bundle "$bundle" --source-root "$root" --sdk "$sdk"
  manifest="$out/inputs/bundle-manifest.json"
  plan="$out/inputs/reducer-plan.json"
  identity="$out/inputs/device-identity.json"
  homogeneity="$out/inputs/device-homogeneity.json"
  cp -- "$bundle/manifest.json" "$manifest"
  cp -- "$bundle/reducer-plan.json" "$plan"
  cp -- "$homogeneity_input" "$homogeneity"
  probe="$bundle/box_identity_probe"
  python3 -B "$root/tools/probe_box_identity.py" resolve \
    --runtime-probe-binary "$probe" --output "$identity"
  cmp -s -- "$identity_input" "$identity" ||
    fail 'live device identity differs from the campaign worker identity'

  binary="$bundle/test_fq_splitk_reducer_lookup"
  export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib64:$sdk/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  execution="$out/inputs/execution-authority.json"
  python3 -B "$root/tools/analyze_fq_splitk_reducer_lookup.py" \
    write-execution-authority --manifest "$manifest" --plan "$plan" \
    --device-identity "$identity" --device-homogeneity "$homogeneity" \
    --binary "$binary" --worker-id "$worker_id" \
    --visible-device "$CUDA_VISIBLE_DEVICES" --output "$execution"

  for round in 1 2 3; do
    seed="$(python3 -B "$root/tools/plan_fq_splitk_reducer_lookup.py" \
      round-seed --round "$round")"
    log="$out/runs/round-$round.log"
    temp="$out/runs/.round-$round.current.$$"
    [[ ! -e "$temp" && ! -L "$temp" ]] || fail 'round temporary log exists'
    set +e
    "$binary" --round="$round" --schedule-seed="$seed" \
      --warmups=3 --samples=11 --case-begin=0 --case-end=1035 \
      --plant-output-fault=0 >"$temp" 2>&1
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      mv -- "$temp" "$log.failed"
      tail -n 160 "$log.failed" >&2
      fail "round $round failed rc=$rc"
    fi
    mv -- "$temp" "$log"
    grep -Fq "FQ_REDUCER_LOOKUP_DONE " "$log" ||
      fail "round $round has no completion marker"
  done

  python3 -B "$root/tools/analyze_fq_splitk_reducer_lookup.py" analyze \
    --plan "$plan" --manifest "$manifest" \
    --execution-authority "$execution" --runs "$out/runs" \
    --output "$out/results"
  cat "$out/results/verdict.log"
  printf '[fq-splitk-reducer-box] PASS cases=1035 rounds=3 samples_per_case=33 artifacts=%s\n' "$out"
}

main "$@"
