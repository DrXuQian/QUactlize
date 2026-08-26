#!/usr/bin/env bash
# Prove that an executable named nvcc is a complete NVIDIA CUDA compiler before
# using repository PPU stubs.  On the PPU box nvcc may delegate device
# preprocessing to ppu_clang++; the name/version then lies about this fixture's
# capability and float8.h fails on hggc_fp8.h.
set -uo pipefail

main() {
  local compiler="${1:-}" tag="${2:-nvcc-host-oracle}" tmp src bin log rc diagnostic
  [[ -n "$compiler" && -x "$compiler" ]] || {
    printf '[%s] SKIP: nvcc is unavailable\n' "$tag" >&2
    return 3
  }
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-nvidia-nvcc-probe.XXXXXX")" || return 2
  case "$tmp" in "${TMPDIR:-/tmp}"/quactlize-nvidia-nvcc-probe.*) ;; *) return 2 ;; esac
  src="$tmp/probe.cu"; bin="$tmp/probe"; log="$tmp/probe.log"
  trap 'rm -rf -- "$tmp"' RETURN
  printf '%s\n' \
    '#include <cuda_fp16.h>' \
    '__global__ void k(__half* p){ *p = __hadd(p[threadIdx.x], p[blockIdx.x]); }' \
    'int main(){ return 0; }' >"$src" || return 2
  "$compiler" -std=c++17 -arch=sm_80 -w "$src" -o "$bin" >"$log" 2>&1
  rc=$?
  if [[ $rc -ne 0 || ! -s "$bin" ]]; then
    diagnostic="$(grep -m1 -E ': (fatal )?error:' "$log" || tail -n 1 "$log")"
    [[ -n "$diagnostic" ]] || diagnostic="compiler probe failed without a diagnostic"
    printf '[%s] SKIP: this nvcc cannot compile the NVIDIA CUDA/CUTLASS fixture; use SHA-bound committed evidence (%s)\n' \
      "$tag" "${diagnostic:0:180}" >&2
    return 3
  fi
  return 0
}

main "$@"
