#!/usr/bin/env bash
set -uo pipefail

main() {
  local repo base out compiler
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  base="${QUACTLIZE_L224_OUT:-/workspace/quactlize-l224-q4-f1-virtual-f2}"
  out="${base}/run-$$"
  mkdir -p "$out" || return 2
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [ -z "$compiler" ]; then
    printf '[l224-runner] FAIL: nvcc is unavailable; virtual-fold proof did not run\n' >&2
    return 2
  fi
  "$repo/dev/fold_derivation/nvidia_nvcc_or_skip.sh" "$compiler" l224-runner || return $?
  "$compiler" -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w \
    -I "$repo/dev/fold_derivation/stub_inc" \
    -I "$repo/third_party/actlize/include" \
    -I "$repo/quactlize/include" \
    "$repo/dev/fold_derivation/l224_q4_f1_virtual_f2.cu" \
    -o "$out/l224_q4_f1_virtual_f2" >"$out/build.log" 2>&1 || {
      printf '[l224-runner] FAIL: virtual-fold proof did not compile\n' >&2
      tail -n 120 "$out/build.log" >&2
      return 2
    }
  "$out/l224_q4_f1_virtual_f2" | tee "$out/run.log" || return 1
  grep -Fqx \
    'L224_Q4_F1_VIRTUAL_F2 verdict=PASS positives=4 negatives=2 artifact=A64/F1 compute=F2 weight_byte_multiplier=1 runtime_branches=0 t32=DEFERRED_MACROSTEP' \
    "$out/run.log" || {
      printf '[l224-runner] FAIL: exact virtual-fold census changed\n' >&2
      return 1
    }
  printf '[l224-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
