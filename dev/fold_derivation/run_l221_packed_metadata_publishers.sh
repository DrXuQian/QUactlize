#!/usr/bin/env bash
set -uo pipefail

main() {
  local repo out compiler
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  out="${QUACTLIZE_L221_OUT:-/workspace/quactlize-l221-packed-metadata-publishers}/run-$$"
  mkdir -p "$out" || return 2
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [ -z "$compiler" ]; then
    printf '[l221-runner] FAIL: nvcc unavailable\n' >&2
    return 2
  fi
  "$compiler" -std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w \
    -I "$repo/dev/fold_derivation/stub_inc" \
    -I "$repo/third_party/actlize/include" \
    -I "$repo/quactlize/include" \
    "$repo/dev/fold_derivation/l221_packed_metadata_publishers.cu" \
    -o "$out/oracle" >"$out/build.log" 2>&1 || {
      tail -n 120 "$out/build.log" >&2
      return 2
    }
  "$out/oracle" | tee "$out/oracle.log" || return 1
  grep -Fqx \
    'L221_PUBLISHERS variant=legacy-modulo-all tile_n=64 cta=128 owners=64 unique=1024 visits=2048 duplicate_visits=1024 hits=2..2 decoder_read_bytes=1024 read_duplicate_writer_overlap=1024 first_second_decoder_warp_column=32' \
    "$out/oracle.log" || return 1
  grep -Fqx \
    'L221_PUBLISHERS variant=owner-only tile_n=64 cta=128 owners=64 unique=1024 visits=1024 duplicate_visits=0 hits=1..1 decoder_read_bytes=1024 read_duplicate_writer_overlap=0 first_second_decoder_warp_column=32' \
    "$out/oracle.log" || return 1
  grep -Fqx \
    'L221_SUMMARY legacy=RED candidate=EXACT verdict=PASS' \
    "$out/oracle.log" || return 1
  printf '[l221-runner] PASS: exact CuTe TN64/CTA128 duplicate-publisher negative RED; owner-only exact; artifacts=%s\n' "$out"
}

main "$@"
