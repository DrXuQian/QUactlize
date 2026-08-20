#!/usr/bin/env bash
set -uo pipefail

main() {
  local repo base out compiler
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  base="${QUACTLIZE_L210_OUT:-/workspace/quactlize-l210-q4-a32-consumer-layout}"
  out="${base}/run-$$"
  mkdir -p "$out" || return 2
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [ -z "$compiler" ]; then
    printf '[l210-runner] FAIL: nvcc is unavailable; cross-layout proof did not run\n' >&2
    return 2
  fi
  "$compiler" -std=c++17 -O2 -x cu -arch=sm_80 \
    --expt-relaxed-constexpr -w \
    -I "$repo/dev/fold_derivation/stub_inc" \
    -I "$repo/third_party/actlize/include" \
    -I "$repo/quactlize/include" \
    "$repo/dev/fold_derivation/l210_q4_a32_consumer_layout.cu" \
    -o "$out/l210_q4_a32_consumer_layout" \
    >"$out/build.log" 2>&1 || {
      printf '[l210-runner] FAIL: cross-layout proof did not compile\n' >&2
      tail -n 120 "$out/build.log" >&2
      return 2
    }
  "$out/l210_q4_a32_consumer_layout" | tee "$out/run.log" || return 1
  grep -Fqx \
    '[l210] PASS rows=10 positives=4 negatives=6 exact_device_byte_bad=32768/32768 exact_device_code_bad=65536/65536 one_bit_corruption_bad=1' \
    "$out/run.log" || {
      printf '[l210-runner] FAIL: expected exact positive/negative census was not produced\n' >&2
      return 1
    }
  printf '[l210-runner] PASS artifacts=%s\n' "$out"
}

main "$@"
