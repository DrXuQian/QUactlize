#!/usr/bin/env bash
# Build the Q12 policy-v2 executable/library locally and seal a prebuilt bundle.
set -euo pipefail

fail() {
  printf '[fq-kquant-policy-v2-build] FAIL: %s\n' "$*" >&2
  exit 2
}

main() {
  [[ $# -le 1 ]] || fail "usage: $0 [BUNDLE_DIR]"
  local root parent out sdk jobs work log build_make binary_count library_count
  local -a build_makes
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  parent="${FQ_KQUANT_PREBUILT_ROOT:-/root/autodl-tmp}"
  mkdir -p "$parent"
  parent="$(realpath -e -- "$parent")"
  out="$(realpath -m -- "${1:-$parent/fq-kquant-policy-v2-prebuilt-$(git -C "$root" rev-parse --short=8 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)}")"
  case "$out" in "$parent"/*) ;; *) fail 'bundle must be a strict artifact-root child';; esac
  [[ ! -e "$out" && ! -L "$out" ]] || fail "refusing existing bundle: $out"
  sdk="${PPU_SDK:-${PPU_HOME:-}}"
  [[ -n "$sdk" ]] || fail 'PPU_SDK is required'
  sdk="$(realpath -e -- "$sdk")"
  [[ -x "$sdk/bin/hgcc" && ! -L "$sdk/bin/hgcc" && -x "$sdk/bin/hgobjdump" && ! -L "$sdk/bin/hgobjdump" ]] || fail 'regular SDK compiler/inspector are required'
  [[ "$("$sdk/bin/hgcc" --version 2>&1 | head -n1)" != *stub* ]] || fail 'stub hgcc is forbidden'
  jobs="${JOBS:-2}"
  [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail 'JOBS must be positive'
  python3 -B "$root/tools/fq_kquant_policy_v2_prebuilt.py" self-test
  git -C "$root" diff --quiet --ignore-submodules=none HEAD -- || fail 'tracked source/submodule state is dirty'
  [[ -z "$(git -C "$root" status --porcelain -- \
    build.sh CMakeLists.txt quactlize benchmarks/test_fq_kquant_layout_perf.cu \
    tools/plan_fq_kquant_policy_v2.py tools/analyze_fq_kquant_policy_v2.py \
    tools/fq_kquant_policy_v2_prebuilt.py tools/build_fq_kquant_policy_v2_prebuilt.sh \
    tools/run_fq_kquant_policy_v2_box.sh)" ]] || fail 'build authority is dirty or untracked'
  work="$(mktemp -d "$parent/.fq-kquant-policy-v2-build.XXXXXX")"
  log="$work/build.log"
  printf '[fq-kquant-policy-v2-build] source=%s work=%s\n' "$(git -C "$root" rev-parse HEAD)" "$work"
  env -u CC -u CXX -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
    PPU_SDK="$sdk" PPU_BUILD_DIR="$work/q12" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    TARGET=test_fq_kquant_layout_perf FQ_KQUANT_PERF_QTYPE=12 \
    PPU_DEFS='PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=0 QUACTLIZE_DENSE_ONLY=12' \
    bash "$root/build.sh" >"$log" 2>&1 || { tail -n 160 "$log" >&2; fail "build failed; preserved $work"; }
  binary_count="$(find -P "$work/q12" -type f -name test_fq_kquant_layout_perf -perm -u+x | wc -l)"
  library_count="$(find -P "$work/q12" -type f -name libquactlize_ppu.so | wc -l)"
  [[ "$binary_count" = 1 && "$library_count" = 1 ]] || fail 'build produced a non-singleton binary/library set'
  mapfile -t build_makes < <(find -P "$work/q12" -type f -path '*test_fq_kquant_layout_perf.dir/build.make')
  [[ ${#build_makes[@]} = 1 ]] || fail 'target build.make is not unique'
  build_make="${build_makes[0]}"
  python3 -B "$root/tools/fq_kquant_policy_v2_prebuilt.py" create \
    --bundle "$out" --build "$work/q12" --sdk "$sdk" --build-log "$log" \
    --cmake-log "$work/q12/cmake.log" --build-make "$build_make"
  printf '[fq-kquant-policy-v2-build] PASS bundle=%s work=%s\n' "$out" "$work"
}
main "$@"
