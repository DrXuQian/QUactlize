#!/usr/bin/env bash
# Isolated Q4 dense K-pack-only M=1..64 policy-v2 pilot.
set -uo pipefail

fail() { printf '[fq-kquant-policy-v2] FAIL: %s\n' "$*" >&2; return 2; }

main() {
  [ "$#" -eq 0 ] || { fail 'no positional arguments are accepted'; return 2; }
  local root workspace sha short stamp out jobs iterations warmups rounds sdk plan build log binary library target_make r rc visible_device
  local -a dense_args authority_paths dirty_paths
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2; short="${sha:0:8}"; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-kquant-policy-v2-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  [ ! -e "$out" ] || { fail "refusing existing OUT: $out"; return 2; }
  jobs="${JOBS:-16}"; iterations="${PERF_ITERATIONS:-11}"; warmups="${PERF_WARMUPS:-3}"; rounds="${PERF_ROUNDS:-3}"
  case "$jobs:$iterations:$warmups:$rounds" in *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0) fail 'numeric controls must be positive integers'; return 2;; esac
  [ "$rounds" -ge 2 ] || { fail 'PERF_ROUNDS must be at least 2'; return 2; }
  [ "${SWEEP_PROFILE:-kpack-policy-v2}" = kpack-policy-v2 ] || { fail 'profile identity differs'; return 2; }
  [ "${SWEEP_CONFIGS:-1}" = 1 ] || { fail 'policy-v2 requires SWEEP_CONFIGS=1'; return 2; }
  [ "${RESUME:-0}" = 0 ] || { fail 'policy-v2 does not implement RESUME; RESUME must be 0'; return 2; }
  [ -z "${PPU_DEFS:-}" ] && [ -z "${PPU_EXTRA_DEFS:-}" ] || { fail 'ambient PPU definitions are forbidden'; return 2; }
  visible_device="${CUDA_VISIBLE_DEVICES:-}"
  [[ "$visible_device" =~ ^[0-9]+$ ]] || { fail 'CUDA_VISIBLE_DEVICES must name exactly one numeric device ordinal'; return 2; }
  sdk="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
  [ -x "$sdk/bin/hgcc" ] && [ -x "$sdk/bin/hgobjdump" ] || { fail 'real inherited PPU SDK is required'; return 2; }
  [[ "$("$sdk/bin/hgcc" --version 2>&1 | head -n 1 || true)" != *stub* ]] || { fail 'stub hgcc is forbidden'; return 2; }
  mkdir -p "$out"/build "$out"/runs "$out"/results "$out"/inputs || return 2
  plan="$out/inputs/plan.json"
  python3 -B "$root/tools/plan_fq_kquant_policy_v2.py" self-test || return 2
  python3 -B "$root/tools/analyze_fq_kquant_policy_v2.py" self-test || return 2
  python3 -B "$root/tools/plan_fq_kquant_policy_v2.py" materialize --output "$plan" || return 2
  mapfile -t dense_args < <(python3 -B - "$plan" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))['dense']:
 print(f"--dense={row['m']},{row['n']},{row['k']}")
PY
  ) || return 2
  [ "${#dense_args[@]}" -eq 64 ] || { fail 'plan-to-CLI denominator differs'; return 2; }
  authority_paths=(
    build.sh CMakeLists.txt quactlize/csrc/CMakeLists.txt
    benchmarks/test_fq_kquant_layout_perf.cu
    quactlize/csrc/device/ppu_dense_backend.cu
    quactlize/include/ppu_dense_configs.inc
    quactlize/include/ppu_grouped_configs.inc
    quactlize/include/ppu_q4_kpack4_shipping_policy.hpp
    quactlize/include/kquant_kpack_offline.hpp
    tools/plan_fq_kquant_policy_v2.py
    tools/analyze_fq_kquant_policy_v2.py
    tools/run_fq_kquant_policy_v2_box.sh
    tools/run_fq_kquant_kpack_perf_box.sh
  )
  # The source SHA only binds tracked files. Reject changes anywhere in the
  # product source tree (including untracked headers), not merely in the small
  # list hashed below for human-readable authority.
  dirty_paths=(
    build.sh CMakeLists.txt quactlize
    benchmarks/test_fq_kquant_layout_perf.cu
    third_party/actlize third_party/cutlass
    tools/plan_fq_kquant_policy_v2.py
    tools/analyze_fq_kquant_policy_v2.py
    tools/run_fq_kquant_policy_v2_box.sh
    tools/run_fq_kquant_kpack_perf_box.sh
  )
  [ -z "$(git -C "$root" status --porcelain -- "${dirty_paths[@]}")" ] || { fail 'tracked/staged/untracked build input is dirty'; return 2; }
  {
    printf 'source_sha=%s\n' "$sha"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$visible_device"
    printf 'sdk_root=%s\n' "$(realpath -e "$sdk")"
    printf 'hgcc_version=%s\n' "$("$sdk/bin/hgcc" --version 2>&1 | head -n 1)"
    printf '%s\n' 'submodules:'
    git -C "$root" submodule status --recursive
    printf '%s\n' 'sdk_files:'
    if [ -f "$sdk/release.yaml" ]; then sha256sum "$sdk/release.yaml"; else printf '%s\n' 'release_yaml=ABSENT'; fi
    sha256sum "$sdk/bin/hgcc" "$sdk/bin/hgobjdump"
    printf '%s\n' 'build_inputs:'
    for path in "${authority_paths[@]}"; do sha256sum "$root/$path"; done
  } > "$out/source-authority.txt" || return 2
  build="$out/build/q12"; log="$out/results/build-q12.log"
  printf '[fq-kquant-policy-v2] build q=12 packed_format=0\n'
  (cd "$root" && env -u CC -u CXX -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
    PPU_BUILD_DIR="$build" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    TARGET=test_fq_kquant_layout_perf FQ_KQUANT_PERF_QTYPE=12 \
    PPU_DEFS='PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=0 QUACTLIZE_DENSE_ONLY=12' ./build.sh) > "$log" 2>&1
  rc=$?; [ "$rc" -eq 0 ] || { tail -n 160 "$log" >&2; fail "build rc=$rc artifacts=$out"; return "$rc"; }
  binary="$(find "$build" -type f -name test_fq_kquant_layout_perf -perm -u+x -print -quit)"
  library="$(find "$build" -type f -name libquactlize_ppu.so -print -quit)"
  [ -x "$binary" ] && [ -f "$library" ] && [ ! -L "$binary" ] && [ ! -L "$library" ] || { fail 'exact binary/library missing or symlinked'; return 2; }
  target_make="$(find "$build" -type f -path '*test_fq_kquant_layout_perf.dir/build.make' -print -quit)"
  grep -Fqx '[build.sh] FQ_KQUANT_PERF_QTYPE=12' "$log" && \
    grep -F 'PPU_PACKED_FORMAT=0' "$log" >/dev/null && \
    grep -F 'QUACTLIZE_DENSE_ONLY=12' "$log" >/dev/null && \
    grep -F 'FullyQuantized K-quant layout perf: qtype=12 carrier=production-C-ABI' "$build/cmake.log" >/dev/null && \
    [ -n "$target_make" ] && grep -F -- '-DFQ_KQUANT_PERF_QTYPE=12' "$target_make" >/dev/null || { fail 'Q4 build identity did not reach build.sh/CMake/target'; return 2; }
  sha256sum "$binary" "$library" > "$out/results/binaries.sha256" || return 2
  for r in $(seq 1 "$rounds"); do
    log="$out/runs/q12-round$r.log"
    printf '[fq-kquant-policy-v2] run q=12 round=%s layout=kpack\n' "$r"
    LD_LIBRARY_PATH="$(dirname "$library")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$binary" --iterations="$iterations" --warmups="$warmups" --round="$r" \
      --order=kpack-first --all-configs=1 --profile=kpack-policy-v2 \
      "${dense_args[@]}" > "$log" 2>&1
    rc=$?; [ "$rc" -eq 0 ] || { tail -n 160 "$log" >&2; fail "round$r rc=$rc"; return "$rc"; }
    grep -Fqx "FQ_KQUANT_POLICY_RUN schema=kpack-policy-v2 q=12 round=$r layout=kpack order=kpack-first iterations=$iterations warmups=$warmups all_configs=1 dense_cases=64 grouped_cases=0 status=PASS" "$log" || { fail "round$r completion marker differs"; return 2; }
    sha256sum "$log" > "$log.sha256" || return 2
  done
  python3 -B "$root/tools/analyze_fq_kquant_policy_v2.py" analyze --plan "$plan" --runs "$out/runs" --output "$out/results" --rounds "$rounds" --iterations "$iterations" --warmups "$warmups" || return 2
  printf '[fq-kquant-policy-v2] DIAGNOSTIC_COMPLETE sha=%s artifacts=%s\n' "$sha" "$out"
}
main "$@"
