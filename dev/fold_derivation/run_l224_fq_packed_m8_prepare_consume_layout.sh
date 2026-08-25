#!/usr/bin/env bash
set -uo pipefail

main() {
  local repo out binary
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  out="${QUACTLIZE_L224_OUT:-/tmp/quactlize-l224}"
  mkdir -p "$out" || return 2
  binary="$out/l224_fq_packed_m8_prepare_consume_layout"
  nvcc -std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr \
    -I "$repo/dev/fold_derivation/stub_inc" \
    -I "$repo/third_party/actlize/include" \
    -I "$repo/third_party/actlize/tools/util/include" \
    -I "$repo/quactlize/include" \
    -o "$binary" \
    "$repo/dev/fold_derivation/l224_fq_packed_m8_prepare_consume_layout.cu" \
    || return 2
  "$binary" || return 1
  printf '[l224] PASS: exact A block/register lifetime and stale-sb13 incident fingerprint\n'
}

main "$@"
