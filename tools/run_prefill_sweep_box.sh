#!/usr/bin/env bash
# Real-format q/k/v/o sweep for the M values named by the committed spec.
# qtype comes from the GGUF tensor header. The bounded denominator is the
# explicitly tagged semantic rows in test_scalefirst_bench plus Q8's shared
# candidate manifest. This is a performance envelope, not a claim that every
# generated tactic exists, and it does not time the scale prepass/direct-FQ arm.
set -uo pipefail

main() {
  local root spec spec_rel gguf sha short stamp out build_root build_log plan binary rc proof_log proof_evidence
  local -a bins
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  spec="${PREFILL_SPEC:-$root/benchmarks/prefill_qwen35_a3b_smoke.json}"
  spec="$(realpath -e -- "$spec")" || {
    printf '[prefill-sweep] FAIL: spec does not exist: %s\n' "${PREFILL_SPEC:-$root/benchmarks/prefill_qwen35_a3b_smoke.json}" >&2
    return 2
  }
  case "$spec" in
    "$root"/*) spec_rel="${spec#"$root"/}" ;;
    *) printf '[prefill-sweep] FAIL: spec must be a committed repository file: %s\n' "$spec" >&2; return 2 ;;
  esac
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
  proof_log="$out/l208-q8-layout.log"
  proof_evidence="$out/l208-q8-layout.expected.txt"

  # This result must bind to the named SHA, not merely to whatever source
  # bytes happened to be in a dirty checkout. A partial include-graph list
  # cannot establish that property: a dirty transitive collective or CMake
  # file changes the binary just as surely as this runner does. Require the
  # complete tracked root and actlize trees to match their named commits.
  if ! git -C "$root" diff --quiet HEAD -- ||
     ! git -C "$root" diff --cached --quiet HEAD --; then
    printf '[prefill-sweep] FAIL: tracked source differs from root SHA %s\n' "$sha" >&2
    git -C "$root" status --short >&2
    return 2
  fi
  if ! git -C "$root/third_party/actlize" diff --quiet HEAD -- ||
     ! git -C "$root/third_party/actlize" diff --cached --quiet HEAD --; then
    printf '[prefill-sweep] FAIL: tracked actlize source differs from its named SHA\n' >&2
    git -C "$root/third_party/actlize" status --short >&2
    return 2
  fi
  local -a authorities=(
    tools/prefill_sweep.py tools/run_prefill_sweep_box.sh
    "$spec_rel"
    benchmarks/test_scalefirst_bench.cu benchmarks/prefill_q8_candidates.inc
    dev/fold_derivation/l208_q8_emit_layout.cu
    dev/fold_derivation/run_l208_q8_emit_layout.sh
    dev/fold_derivation/l208_q8_emit_layout.expected.txt
    ci/check_l208_q8_committed_evidence.py
    quactlize/include/actlize_extensions/cutlass/quactlize_mix_gemm_convert.h
    quactlize/include/xplane_offline.hpp quactlize/include/ppu_format_config.inc
  )
  if ! git -C "$root" diff --quiet HEAD -- "${authorities[@]}" ||
     ! git -C "$root" diff --cached --quiet HEAD -- "${authorities[@]}"; then
    printf '[prefill-sweep] FAIL: a prefill authority differs from committed SHA %s\n' "$sha" >&2
    git -C "$root" status --short -- "${authorities[@]}" >&2
    return 2
  fi
  for authority in "${authorities[@]}"; do
    git -C "$root" ls-files --error-unmatch "$authority" >/dev/null 2>&1 || {
      printf '[prefill-sweep] FAIL: authority is not tracked by SHA %s: %s\n' "$sha" "$authority" >&2
      return 2
    }
  done
  local vendor_converter=include/cutlass/fast_numeric_conversion_for_mix_gemm.h
  if ! git -C "$root/third_party/actlize" diff --quiet HEAD -- "$vendor_converter" ||
     ! git -C "$root/third_party/actlize" diff --cached --quiet HEAD -- "$vendor_converter" ||
     ! git -C "$root/third_party/actlize" ls-files --error-unmatch "$vendor_converter" >/dev/null 2>&1; then
    printf '[prefill-sweep] FAIL: actlize int8 converter authority is dirty or untracked: %s\n' \
      "$vendor_converter" >&2
    return 2
  fi

  python3 "$root/tools/prefill_sweep.py" self-test || return 2
  # L208 is a host/CUDA compile-time oracle.  The PPU box's `nvcc` delegates
  # device preprocessing to ppu_clang++, where the repository's stub headers
  # cannot be mixed with the real SDK (all-stub lacks hggc_fp8; all-real hits
  # the CUDA/GCC13 seam; mixing them hits __assert).  Consume the exact local
  # evidence from the result SHA instead.  The production target below is
  # still compiled fresh by hgcc, so this does not replace device admission.
  git -C "$root" show "$sha:dev/fold_derivation/l208_q8_emit_layout.expected.txt" \
    >"$proof_evidence" || {
    printf '[prefill-sweep] FAIL: result SHA lacks committed L208 evidence\n' >&2
    return 2
  }
  {
    printf '[prefill-sweep] l208-evidence=committed-local-oracle source-sha=%s fresh-box-execution=0\n' "$sha"
    python3 "$root/ci/check_l208_q8_committed_evidence.py" \
      --committed-only --evidence "$proof_evidence"
  } 2>&1 | tee "$proof_log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '[prefill-sweep] FAIL: committed Q8 layout authority L208 returned rc=%d; no binary was built\n' "$rc" >&2
    return "$rc"
  fi
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
    printf 'cache_scope=same-address-warm-upper-envelope;distinct-MBU-is-a-traffic-model-not-a-counter\n'
    printf 'candidate_scope=bounded-performance-envelope;q8-authority=benchmarks/prefill_q8_candidates.inc\n'
    printf 'host=%s\n' "$(hostname)"
    printf 'kernel=%s\n' "$(uname -srmo)"
    printf 'hgcc=%s\n' "$(command -v hgcc 2>/dev/null || printf UNAVAILABLE)"
    printf 'hgcc_version_begin\n'; hgcc --version 2>&1 || true; printf 'hgcc_version_end\n'
    printf 'l208_evidence=%s\n' "$proof_evidence"
    printf 'l208_evidence_sha256=%s\n' "$(sha256sum "$proof_evidence" | awk '{print $1}')"
    printf 'l208_log=%s\n' "$proof_log"
    printf 'l208_log_sha256=%s\n' "$(sha256sum "$proof_log" | awk '{print $1}')"
  } >"$out/provenance.txt"

  # Admission is deliberately after plan + provenance are durable and before
  # build.  A checkpoint for which every selected format lacks a proved row
  # family must leave useful evidence while compiling/running exactly nothing.
  python3 "$root/tools/prefill_sweep.py" admit --plan "$plan" 2>&1 | tee "$out/admission.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf '[prefill-sweep] NO_SUPPORTED_CELLS: no binary was built or measured\n' >&2
    printf '[prefill-sweep] artifacts: %s\n' "$out" >&2
    return "$rc"
  fi

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
