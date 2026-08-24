#!/usr/bin/env bash
set -uo pipefail

main() {
  local repo out source
  local -a common

  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  out="${QUACTLIZE_L223_OUT:-/tmp/quactlize_l223}"
  mkdir -p "$out" || return 2

  common=(
    -std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr
    -I "$repo/dev/fold_derivation/stub_inc"
    -I "$repo/third_party/actlize/include"
    -I "$repo/third_party/actlize/tools/util/include"
    -I "$repo/quactlize/include"
  )
  source="$repo/dev/fold_derivation/l223_fq_splitk_shared_epilogue_layout.cu"

  nvcc "${common[@]}" -o "$out/green" "$source" || return 2
  "$out/green" || return 2

  nvcc "${common[@]}" -DL223_BAD_R2S_ROTATE=1 \
    -o "$out/bad-r2s" "$source" || return 2
  if "$out/bad-r2s"; then
    echo '[l223] FAIL: rotated R2S value-coordinate plant stayed green' >&2
    return 2
  fi

  nvcc "${common[@]}" -DL223_BAD_S2R_THREAD_MODULO=1 \
    -o "$out/bad-s2r" "$source" || return 2
  if "$out/bad-s2r"; then
    echo '[l223] FAIL: duplicate S2R thread-owner plant stayed green' >&2
    return 2
  fi

  echo '[l223] PASS: exact legacy R2S/S2R map plus value-coordinate and owner negatives'
}

main "$@"
