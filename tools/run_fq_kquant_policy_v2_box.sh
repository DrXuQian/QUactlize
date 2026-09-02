#!/usr/bin/env bash
# Execute the source-bound Q12 policy-v2 prebuilt; this runner never builds.
set -euo pipefail

fail() {
  printf '[fq-kquant-policy-v2-box] FAIL: %s\n' "$*" >&2
  exit 2
}

main() {
  [[ $# = 0 ]] || fail 'no positional arguments are accepted'
  local root artifact_root out bundle bundle_input sdk manifest plan binary library iterations warmups rounds r log
  local -a dense_args
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  artifact_root="$(realpath -e -- "${FQ_KQUANT_POLICY_V2_ROOT:-/root/autodl-tmp}")" || fail 'artifact root is absent'
  out="$(realpath -m -- "${OUT:-$artifact_root/fq-kquant-policy-v2-run-$(date -u +%Y%m%dT%H%M%SZ)-$$}")"
  case "$out" in "$artifact_root"/*) ;; *) fail 'OUT must be a strict artifact-root child';; esac
  [[ ! -e "$out" && ! -L "$out" ]] || fail "refusing existing OUT: $out"
  case "${CUDA_VISIBLE_DEVICES:-}" in ''|*,*|*[!0-9]*) fail 'CUDA_VISIBLE_DEVICES must name exactly one numeric ordinal';; esac
  [[ "${RESUME:-0}" = 0 ]] || fail 'RESUME is not implemented'
  [[ -z "${PPU_DEFS:-}${PPU_EXTRA_DEFS:-}" ]] || fail 'ambient build definitions are forbidden'
  bundle_input="${FQ_KQUANT_POLICY_V2_BUNDLE:-/nonexistent}"
  [[ -d "$bundle_input" && ! -L "$bundle_input" ]] || fail 'bundle must be a regular non-symlink directory'
  bundle="$(realpath -e -- "$bundle_input")" || fail 'exact prebuilt bundle is required'
  case "$bundle" in "$artifact_root"/*) ;; *) fail 'bundle must be a strict artifact-root child';; esac
  manifest="$bundle/manifest.json"
  [[ -f "$manifest" && ! -L "$manifest" ]] || fail 'manifest is missing or symlinked'
  sdk="${PPU_SDK:-${PPU_HOME:-}}"
  [[ -n "$sdk" ]] || fail 'PPU_SDK is required'
  sdk="$(realpath -e -- "$sdk")"
  iterations="${PERF_ITERATIONS:-11}"
  warmups="${PERF_WARMUPS:-3}"
  rounds="${PERF_ROUNDS:-3}"
  case "$iterations:$warmups:$rounds" in *[!0-9:]*|0:*|*:0:*|*:*:0) fail 'timing controls must be positive integers';; esac
  [[ "$rounds" -ge 2 ]] || fail 'PERF_ROUNDS must be at least 2'
  [[ "${SWEEP_PROFILE:-kpack-policy-v2}" = kpack-policy-v2 && "${SWEEP_CONFIGS:-1}" = 1 ]] || fail 'profile identity differs'
  mkdir -p "$out/inputs" "$out/runs" "$out/results"
  python3 -B "$root/tools/fq_kquant_policy_v2_prebuilt.py" self-test
  python3 -B "$root/tools/plan_fq_kquant_policy_v2.py" self-test
  python3 -B "$root/tools/analyze_fq_kquant_policy_v2.py" self-test
  python3 -B "$root/tools/fq_kquant_policy_v2_prebuilt.py" verify \
    --bundle "$bundle" --source-root "$root" --sdk "$sdk" \
    --execution-sdk-compatible
  python3 -B "$root/tools/probe_box_identity.py" resolve \
    --output "$out/inputs/box-identity.json" || fail 'runtime one-device probe failed'
  python3 -B - "$out/inputs/box-identity.json" <<'PY' || fail 'measured runtime one-device evidence is required'
import json,sys
probe=json.load(open(sys.argv[1]))['device_probe']
assert probe['status'] in ('measured','properties-unavailable')
assert probe['device_count']==1
PY
  plan="$out/inputs/plan.json"
  python3 -B "$root/tools/plan_fq_kquant_policy_v2.py" materialize --output "$plan"
  cp -- "$manifest" "$out/inputs/bundle-manifest.json"
  binary="$bundle/test_fq_kquant_layout_perf"; library="$bundle/libquactlize_ppu.so"
  [[ -x "$binary" && -f "$binary" && ! -L "$binary" && -f "$library" && ! -L "$library" ]] || fail 'verified artifacts changed'
  {
    printf 'runner_source=%s\n' "$(git -C "$root" rev-parse HEAD)"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
    sha256sum "$manifest" "$binary" "$library" "$plan" "$out/inputs/box-identity.json" \
      "$root/tools/fq_kquant_policy_v2_prebuilt.py" \
      "$root/tools/plan_fq_kquant_policy_v2.py" \
      "$root/tools/analyze_fq_kquant_policy_v2.py" \
      "$root/tools/probe_box_identity.py" "$root/tools/run_fq_kquant_policy_v2_box.sh"
  } >"$out/inputs/result-authority.txt"
  mapfile -t dense_args < <(python3 -B - "$plan" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))['dense']:
 print(f"--dense={row['m']},{row['n']},{row['k']}")
PY
  )
  [[ ${#dense_args[@]} = 64 ]] || fail 'plan denominator differs'
  for r in $(seq 1 "$rounds"); do
    log="$out/runs/q12-round$r.log"
    LD_LIBRARY_PATH="$bundle${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$binary" --iterations="$iterations" --warmups="$warmups" --round="$r" \
      --order=kpack-first --all-configs=1 --profile=kpack-policy-v2 \
      "${dense_args[@]}" >"$log" 2>&1 || { tail -n 160 "$log" >&2; fail "round$r failed"; }
    grep -Fqx "FQ_KQUANT_POLICY_RUN schema=kpack-policy-v2 q=12 round=$r layout=kpack order=kpack-first iterations=$iterations warmups=$warmups all_configs=1 dense_cases=64 grouped_cases=0 status=PASS" "$log" || fail "round$r completion marker differs"
    sha256sum "$log" >"$log.sha256"
  done
  python3 -B "$root/tools/analyze_fq_kquant_policy_v2.py" analyze \
    --plan "$plan" --runs "$out/runs" --output "$out/results" \
    --rounds "$rounds" --iterations "$iterations" --warmups "$warmups"
  printf '[fq-kquant-policy-v2-box] DIAGNOSTIC_COMPLETE artifacts=%s\n' "$out"
}
main "$@"
