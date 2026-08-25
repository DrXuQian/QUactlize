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
  source="$repo/dev/fold_derivation/l223_fq_splitk_partial_abi.cu"

  nvcc "${common[@]}" -o "$out/green" "$source" || return 2
  "$out/green" || return 2

  local plant macro
  for plant in bad-plane-stride bad-plane-select bad-reducer-pitch; do
    case "$plant" in
      bad-plane-stride) macro=L223_BAD_PLANE_STRIDE ;;
      bad-plane-select) macro=L223_BAD_PLANE_SELECT ;;
      bad-reducer-pitch) macro=L223_BAD_REDUCER_PITCH ;;
    esac
    nvcc "${common[@]}" -D"$macro"=1 -o "$out/$plant" "$source" || return 2
    if "$out/$plant"; then
      echo "[l223] FAIL: $plant stayed green" >&2
      return 2
    fi
  done
  echo '[l223] PASS: CuTe producer/plane/reducer equivalence plus three negatives'
}

main "$@"
