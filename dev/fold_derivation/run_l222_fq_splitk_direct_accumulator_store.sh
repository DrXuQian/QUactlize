#!/usr/bin/env bash
set -uo pipefail

main() {
  local repo out source
  local -a common

  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  out="${QUACTLIZE_L222_OUT:-/tmp/quactlize_l222}"
  mkdir -p "$out" || return 2

  common=(
    -std=c++17 -x cu -arch=sm_80 -w --expt-relaxed-constexpr
    -I "$repo/dev/fold_derivation/stub_inc"
    -I "$repo/third_party/actlize/include"
    -I "$repo/third_party/actlize/tools/util/include"
    -I "$repo/quactlize/include"
  )
  source="$repo/dev/fold_derivation/l222_fq_splitk_direct_accumulator_store.cu"

  nvcc "${common[@]}" -o "$out/green" "$source" || return 2
  "$out/green" || return 2

  nvcc "${common[@]}" -DL222_BAD_THREAD_MODULO=1 \
    -o "$out/bad-thread" "$source" || return 2
  if "$out/bad-thread"; then
    echo '[l222] FAIL: bad-thread ownership plant stayed green' >&2
    return 2
  fi

  nvcc "${common[@]}" -DL222_BAD_FRAGMENT_ROTATE=1 \
    -o "$out/bad-fragment" "$source" || return 2
  if "$out/bad-fragment"; then
    echo '[l222] FAIL: bad-fragment coordinate plant stayed green' >&2
    return 2
  fi

  echo '[l222] PASS: exact direct store plus duplicate-owner and register-coordinate negatives'
}

main "$@"
