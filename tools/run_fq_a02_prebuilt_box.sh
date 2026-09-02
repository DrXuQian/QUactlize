#!/usr/bin/env bash
# Execute-only A02 typed diagnostics. Grouped product evidence is referenced
# from the strict A01 result and is never re-labelled as execution by this run.
set -euo pipefail

fail() {
  printf '[fq-a02-box] FAIL: %s\n' "$*" >&2
  exit 2
}

main() {
  [[ $# -eq 0 ]] || fail 'no positional arguments are accepted'
  local root artifact_root out raw_bundle bundle sdk a01 a01_copy repeats iterations
  local q4_binary q3_binary

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  artifact_root="$(realpath -e -- "${FQ_A02_ROOT:-/root/autodl-tmp}")" ||
    fail 'artifact root is absent'
  out="$(realpath -m -- "${OUT:-$artifact_root/fq-a02-run-$(date -u +%Y%m%dT%H%M%SZ)-$$}")"
  case "$out" in "$artifact_root"/*) ;; *) fail 'OUT is outside the artifact root' ;; esac
  [[ ! -e "$out" && ! -L "$out" ]] || fail 'OUT already exists'
  case "${CUDA_VISIBLE_DEVICES:-}" in
    ''|*,*|*[!0-9]*) fail 'CUDA_VISIBLE_DEVICES must name one numeric ordinal' ;;
  esac

  raw_bundle="${FQ_A02_BUNDLE:-/nonexistent}"
  [[ -d "$raw_bundle" && ! -L "$raw_bundle" ]] ||
    fail 'a regular non-symlink prebuilt bundle is required'
  bundle="$(realpath -e -- "$raw_bundle")"
  case "$bundle" in "$artifact_root"/*) ;; *) fail 'bundle is outside the artifact root' ;; esac
  sdk="$(realpath -e -- "${PPU_SDK:-${PPU_HOME:-/nonexistent}}")" ||
    fail 'same-release runtime SDK is required'
  a01="${A01_PRODUCT_GATE_RESULT:-/nonexistent}"
  [[ -f "$a01" && ! -L "$a01" ]] || fail 'regular A01 result.json is required'

  repeats="${CORRECTNESS_REPEATS:-4096}"
  iterations="${ITERATIONS:-3}"
  [[ "$repeats" =~ ^[0-9]+$ && "$repeats" -ge 1024 ]] ||
    fail 'CORRECTNESS_REPEATS must be at least 1024'
  [[ "$iterations" =~ ^[1-9][0-9]*$ ]] || fail 'ITERATIONS must be positive'

  mkdir -p "$out/inputs" "$out/results"
  a01_copy="$out/inputs/a01-product-gate-result.json"
  cp -- "$a01" "$a01_copy"
  python3 -B "$root/tools/fq_a02_prebuilt.py" verify \
    --bundle "$bundle" --source-root "$root" --sdk "$sdk"
  python3 -B "$root/tools/fq_a02_prebuilt.py" verify-a01 \
    --result "$a01_copy" --summary "$out/inputs/a01-reference-summary.json" \
    --minimum-q4-repeats 1024
  python3 -B "$root/tools/probe_box_identity.py" resolve \
    --output "$out/inputs/box-identity.json"
  python3 -B - "$out/inputs/box-identity.json" <<'PY' ||
    fail 'measured one-device runtime probe is required'
import json
import sys
probe = json.load(open(sys.argv[1], encoding="utf-8"))["device_probe"]
assert probe["status"] in ("measured", "properties-unavailable")
assert probe["device_count"] == 1
PY

  cp -- "$bundle/manifest.json" "$out/inputs/bundle-manifest.json"
  q4_binary="$bundle/test_fully_quantized_internal_sweep"
  q3_binary="$bundle/test_fq_a02_q3_bchunk_aggregate"
  "$q4_binary" --shape=1x1024x5120 --iterations="$iterations" \
    --correctness-repeats="$repeats" --only-split=1 --bc-mode=skip \
    >"$out/results/q4.log" 2>&1 || {
      tail -n 120 "$out/results/q4.log" >&2
      fail 'Q4 diagnostic failed'
    }
  "$q3_binary" --shape=1x1024x5120 --iterations="$iterations" \
    --correctness-repeats="$repeats" --only-split=1 --bc-mode=skip \
    >"$out/results/q3.log" 2>&1 || {
      tail -n 120 "$out/results/q3.log" >&2
      fail 'Q3 aggregate diagnostic failed'
    }
  python3 -B "$root/tools/check_fq_a02_typed_diagnostics.py" \
    --q4-log "$out/results/q4.log" --q3-log "$out/results/q3.log" \
    --repeats "$repeats" | tee "$out/results/verdict.txt"
  sha256sum \
    "$bundle/manifest.json" "$q4_binary" "$q3_binary" \
    "$bundle/q4.isa.txt" "$bundle/q3.isa.txt" \
    "$a01_copy" "$out/inputs/a01-reference-summary.json" \
    "$out/inputs/box-identity.json" "$out/results/q4.log" \
    "$out/results/q3.log" "$out/results/verdict.txt" \
    >"$out/results/authority.sha256"
  printf '[fq-a02-box] PASS product_cells=1 nonproduct_typed_cells=3 grouped_execution=NONE a01_summary_sha=%s artifacts=%s\n' \
    "$(sha256sum "$out/inputs/a01-reference-summary.json" | awk '{print $1}')" "$out"
}

main "$@"
