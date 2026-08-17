#!/usr/bin/env bash
# First real-format prefill smoke sweep: q/k/v/o of layer 0 in the actual
# Qwen3.5-35B-A3B GGUF, for M=64 and M=2048.  qtype comes from the GGUF
# tensor header.  The finite denominator is the explicitly tagged semantic
# rows in test_scalefirst_bench; this script does not call that a full generated
# tactic sweep and does not time the scale prepass or direct-FQ arm.
set -uo pipefail

main() {
  local root spec gguf sha short stamp out build_root build_log plan binary rc
  local -a bins
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  spec="${PREFILL_SPEC:-$root/benchmarks/prefill_qwen35_a3b_smoke.json}"
  gguf="${GGUF:-/sim/eec/shared/AI_workspace/llm-models/Qwen3.5-35B-A3B-Q4_K_M-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf}"
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="${OUT:-/workspace/quactlize-prefill-smoke-${short}-${stamp}}"
  out="$(realpath -m -- "$out")" || return 2
  case "$out" in
    /workspace/*) ;;
    *) printf '[prefill-sweep] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[prefill-sweep] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  if [ ! -f "$gguf" ]; then
    printf '[prefill-sweep] FAIL: GGUF not found: %s\n' "$gguf" >&2
    return 2
  fi
  mkdir -p "$out" || return 2
  plan="$out/plan.json"
  build_root="$out/build"
  build_log="$out/build.log"

  python3 "$root/tools/prefill_sweep.py" self-test || return 2
  python3 "$root/tools/prefill_sweep.py" plan --spec "$spec" --gguf "$gguf" --output "$plan" \
    2>&1 | tee "$out/plan.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then return "$rc"; fi

  {
    printf 'root_sha=%s\n' "$sha"
    printf 'root_status_begin\n'; git -C "$root" status --porcelain=v1; printf 'root_status_end\n'
    printf 'actlize_sha=%s\n' "$(git -C "$root/third_party/actlize" rev-parse HEAD)"
    printf 'gguf=%s\n' "$gguf"
    printf 'spec=%s\n' "$spec"
    printf 'timing_scope=ScaleFirst-GEMM-only;prepass-and-direct-FQ-excluded\n'
    printf 'candidate_scope=finite-manual-test_scalefirst_bench-row-families\n'
  } >"$out/provenance.txt"

  env PPU_BUILD_DIR="$build_root" PPU_ARCHS=ppu0010 TARGET=test_scalefirst_bench \
    JOBS="${JOBS:-16}" "$root/build.sh" 2>&1 | tee "$build_log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '[prefill-sweep] FAIL: test_scalefirst_bench build returned rc=%d\n' "$rc" >&2
    printf '[prefill-sweep] artifacts: %s\n' "$out" >&2
    return "$rc"
  fi
  mapfile -t bins < <(find "$build_root" -type f -name test_scalefirst_bench -perm -u+x -print)
  if [ "${#bins[@]}" -ne 1 ]; then
    printf '[prefill-sweep] FAIL: expected exactly one benchmark binary, found %d\n' "${#bins[@]}" >&2
    printf '  %s\n' "${bins[@]:-<none>}" >&2
    return 2
  fi
  binary="${bins[0]}"
  printf 'binary=%s\nbinary_sha256=%s\n' "$binary" "$(sha256sum "$binary" | awk '{print $1}')" \
    >>"$out/provenance.txt"

  python3 "$root/tools/prefill_sweep.py" measure --plan "$plan" --bin "$binary" \
    --out "$out/results" --repeats "${BENCH_REPS:-3}" --timeout "${SHAPE_TIMEOUT:-1800}" \
    --peak-tflops "${PEAK_TFLOPS:-500}" --hbm-gbs "${HBM_GBS:-2766}" \
    2>&1 | tee "$out/measure.log"
  rc=${PIPESTATUS[0]}
  printf '[prefill-sweep] artifacts: %s\n' "$out"
  printf '[prefill-sweep] summary: %s\n' "$out/results/summary.tsv"
  printf '[prefill-sweep] offline layout plan: %s\n' "$out/results/offline_layout_plan.json"
  return "$rc"
}

main "$@"
