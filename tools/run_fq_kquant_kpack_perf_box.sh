#!/usr/bin/env bash
# Production-C-ABI real-shape performance A/B for Q2/Q3/Q4/Q5/Q6.
set -uo pipefail

fail() {
  printf '[fq-kquant-perf] FAIL: %s\n' "$*" >&2
  return 2
}

atomic_text() {
  local path="$1" value="$2" current="${1}.current.$$"
  printf '%s\n' "$value" > "$current" || return 2
  mv -f -- "$current" "$path"
}

run_committed() {
  local log="$1" hash="${1}.sha256" current failed rc digest
  shift
  if [ -s "$log" ] && [ -s "$hash" ]; then
    [ "${RESUME:-0}" = 1 ] || { fail "phase exists without RESUME=1: $log"; return 2; }
    digest="$(sha256sum "$log" | awk '{print $1}')" || return 2
    [ "$(cat "$hash")" = "$digest" ] || { fail "committed phase changed: $log"; return 2; }
    printf '[fq-kquant-perf] reuse phase=%s\n' "$log"
    return 0
  fi
  if [ -e "$log" ] || [ -e "$hash" ]; then
    [ "${RESUME:-0}" = 1 ] || { fail "phase residue exists without RESUME=1: $log"; return 2; }
    failed="${log}.uncommitted.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    [ ! -e "$log" ] || mv -- "$log" "$failed" || return 2
    [ ! -e "$hash" ] || mv -- "$hash" "${failed}.sha256" || return 2
  fi
  current="${log}.current.$$"
  "$@" > "$current" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    failed="${log}.failed.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mv -- "$current" "$failed" || return 2
    tail -n 160 "$failed" >&2
    fail "phase rc=$rc preserved=$failed"
    return "$rc"
  fi
  mv -- "$current" "$log" || return 2
  digest="$(sha256sum "$log" | awk '{print $1}')" || return 2
  atomic_text "$hash" "$digest"
}

main() {
  [ "$#" -eq 0 ] || { fail 'no positional arguments are accepted'; return 2; }
  local root workspace sha short stamp out resume jobs iterations warmups rounds
  local threshold all_configs profile actual_profile sdk_root plan planner analyzer fitter q fmt format_defs build build_log
  local dense_count grouped_count
  local binary library target_make round order log rc
  local -a dense_args grouped_args run_args

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-kquant-layout-perf-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  resume="${RESUME:-0}"; jobs="${JOBS:-16}"
  iterations="${PERF_ITERATIONS:-11}"; warmups="${PERF_WARMUPS:-3}"
  rounds="${PERF_ROUNDS:-3}"; threshold="${REGRESSION_THRESHOLD_PCT:-3.0}"
  all_configs="${SWEEP_CONFIGS:-0}"
  profile="${SWEEP_PROFILE:-layout-ab}"
  case "$resume:$all_configs" in 0:0|0:1|1:0|1:1) ;; *) fail 'RESUME/SWEEP_CONFIGS must be 0 or 1'; return 2;; esac
  case "$profile" in
    layout-ab|heuristic) ;;
    kpack-policy-v2)
      exec "$root/tools/run_fq_kquant_policy_v2_box.sh"
      ;;
    *) fail 'SWEEP_PROFILE must be layout-ab, heuristic, or kpack-policy-v2'; return 2;;
  esac
  if [ "$profile" = heuristic ] && [ "$all_configs" != 1 ]; then
    fail 'SWEEP_PROFILE=heuristic requires SWEEP_CONFIGS=1'; return 2
  fi
  case "$jobs:$iterations:$warmups:$rounds" in
    *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0) fail 'JOBS/iterations/warmups/rounds must be positive integers'; return 2;;
  esac
  [ "$rounds" -ge 2 ] || { fail 'PERF_ROUNDS must be at least 2 for alternating A/B order'; return 2; }
  python3 -B - "$threshold" <<'PY' || return 2
import sys
x=float(sys.argv[1]); assert 0 < x < 100
PY
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS changes the registered A/B'; return 2
  fi
  if [ -e "$out" ] && [ "$resume" != 1 ]; then
    fail "refusing existing OUT without RESUME=1: $out"; return 2
  fi
  if [ ! -e "$out" ] && [ "$resume" = 1 ]; then
    fail "RESUME=1 requires an existing OUT: $out"; return 2
  fi
  mkdir -p "$out/build" "$out/runs" "$out/results" "$out/inputs" || return 2

  sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
  [ -n "$sdk_root" ] && [ -x "$sdk_root/bin/hgcc" ] && [ -x "$sdk_root/bin/hgobjdump" ] || {
    fail 'real inherited PPU_SDK with hgcc/hgobjdump is required'; return 2; }
  [[ "$($sdk_root/bin/hgcc --version 2>&1 | head -n 1 || true)" != *stub* ]] || {
    fail 'stub hgcc is forbidden'; return 2; }

  planner="$root/tools/plan_fq_kquant_kpack_perf.py"
  analyzer="$root/tools/analyze_fq_kquant_kpack_perf.py"
  fitter="$root/tools/fit_fq_kquant_config_heuristic.py"
  plan="$out/plan.json"
  python3 -B "$planner" self-test || return 2
  python3 -B "$analyzer" self-test >/dev/null || return 2
  python3 -B "$fitter" self-test >/dev/null || return 2
  if [ -s "$plan" ]; then
    [ "$resume" = 1 ] || { fail 'plan exists without RESUME=1'; return 2; }
    python3 -B "$planner" validate --plan "$plan" || return 2
  else
    [ "$resume" != 1 ] || { fail 'resume bundle lost plan.json'; return 2; }
    python3 -B "$planner" materialize --profile "$profile" --output "$plan" || return 2
  fi
  actual_profile="$(python3 -B - "$plan" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(p.get('profile','layout-ab'))
PY
  )" || return 2
  [ "$actual_profile" = "$profile" ] || {
    fail "plan profile=$actual_profile differs from requested $profile"; return 2; }
  mapfile -t dense_args < <(python3 -B - "$plan" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
for r in p['dense']: print(f"--dense={r['m']},{r['n']},{r['k']}")
PY
  ) || return 2
  mapfile -t grouped_args < <(python3 -B - "$plan" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
for r in p['grouped']:
 print(f"--grouped={r['tokens']},{r['n']},{r['k']},{r['experts']},{r['topk']}")
PY
  ) || return 2
  read -r dense_count grouped_count < <(python3 -B - "$plan" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(len(p['dense']), len(p['grouped']))
PY
  ) || return 2
  [ "${#dense_args[@]}" -eq "$dense_count" ] && \
    [ "${#grouped_args[@]}" -eq "$grouped_count" ] || {
    fail 'plan-to-CLI denominator differs'; return 2; }

  if [ "$resume" != 1 ]; then
    git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2
    {
      printf '%s\n' "$sha"
      git -C "$root/third_party/actlize" rev-parse HEAD
      sha256sum "$root/benchmarks/test_fq_kquant_layout_perf.cu" \
        "$root/benchmarks/workloads.py" "$root/benchmarks/moe_router_fixture.hpp" \
        "$root/tools/gguf_internal_shape_inventory.py" \
        "$root/quactlize/csrc/device/ppu_dense_backend.cu" \
        "$root/quactlize/include/kquant_kpack_offline.hpp" \
        "$root/quactlize/include/ppu_dense_configs.inc" \
        "$root/quactlize/include/ppu_grouped_configs.inc" \
        "$planner" "$analyzer" "$fitter" \
        "$root/tools/run_fq_kquant_kpack_perf_box.sh"
    } > "$out/source-authority.sha256" || return 2
  else
    [ -s "$out/source-authority.sha256" ] || { fail 'resume source authority is missing'; return 2; }
    [ "$(sed -n '1p' "$out/source-authority.sha256")" = "$sha" ] || {
      fail 'RESUME requires the exact measurement HEAD'; return 2; }
  fi

  for q in 10 11 12 13 14; do
    case "$q" in 10) fmt=2;; 11) fmt=3;; 12) fmt=0;; 13) fmt=1;; 14) fmt=4;; esac
    format_defs="PPU_PACKED_SCALE=1 PPU_PACKED_FORMAT=$fmt QUACTLIZE_DENSE_ONLY=$q"
    build="$out/build/q$q"; build_log="$out/results/build-q$q.log"
    if [ "$resume" = 1 ] && [ -s "$out/results/binary-q$q.path" ] && \
       [ -s "$out/results/binary-q$q.sha256" ]; then
      binary="$(cat "$out/results/binary-q$q.path")"
      sha256sum -c "$out/results/binary-q$q.sha256" >/dev/null || {
        fail "q$q resume binary/hash differs"; return 2; }
      library="$(cat "$out/results/library-q$q.path")"
      sha256sum -c "$out/results/library-q$q.sha256" >/dev/null || {
        fail "q$q resume library/hash differs"; return 2; }
      printf '[fq-kquant-perf] reuse q=%s binary=%s\n' "$q" "$binary"
    else
      printf '[fq-kquant-perf] build q=%s packed_format=%s\n' "$q" "$fmt"
      (cd "$root" && env -u CC -u CXX -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
        PPU_BUILD_DIR="$build" PPU_ARCHS=ppu0010 JOBS="$jobs" \
        TARGET=test_fq_kquant_layout_perf FQ_KQUANT_PERF_QTYPE="$q" \
        PPU_DEFS="$format_defs" \
        ./build.sh) > "$build_log" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
        tail -n 180 "$build_log" >&2
        fail "q$q build rc=$rc artifacts=$out"; return "$rc"
      fi
      binary="$(find "$build" -type f -name test_fq_kquant_layout_perf -perm -u+x -print -quit)"
      library="$(find "$build" -type f -name 'libquactlize_ppu.so' -print -quit)"
      [ -x "$binary" ] && [ -f "$library" ] && [ ! -L "$binary" ] && [ ! -L "$library" ] || {
        fail "q$q exact binary/library is missing or symlinked"; return 2; }
      target_make="$(find "$build" -type f -path '*test_fq_kquant_layout_perf.dir/build.make' -print -quit)"
      grep -Fqx "[build.sh] FQ_KQUANT_PERF_QTYPE=$q" "$build_log" && \
      grep -F "PPU_PACKED_FORMAT=$fmt" "$build_log" >/dev/null && \
      grep -F "QUACTLIZE_DENSE_ONLY=$q" "$build_log" >/dev/null && \
      grep -F "FullyQuantized K-quant layout perf: qtype=$q carrier=production-C-ABI" \
        "$build/cmake.log" >/dev/null && [ -n "$target_make" ] && \
      grep -F -- "-DFQ_KQUANT_PERF_QTYPE=$q" "$target_make" >/dev/null || {
          fail "q$q build identity did not reach CMake/target"; return 2; }
      printf '%s\n' "$binary" > "$out/results/binary-q$q.path"
      printf '%s\n' "$library" > "$out/results/library-q$q.path"
      sha256sum "$binary" > "$out/results/binary-q$q.sha256" || return 2
      sha256sum "$library" > "$out/results/library-q$q.sha256" || return 2
    fi
    for round in $(seq 1 "$rounds"); do
      if ((round % 2)); then order=xplane-first; else order=kpack-first; fi
      log="$out/runs/q$q-round$round.log"
      run_args=("--iterations=$iterations" "--warmups=$warmups" "--round=$round"
                "--order=$order" "--all-configs=$all_configs")
      if [ "$q" = 12 ]; then
        run_args+=("${grouped_args[@]}")
      else
        run_args+=("${dense_args[@]}" "${grouped_args[@]}")
      fi
      printf '[fq-kquant-perf] run q=%s round=%s order=%s\n' "$q" "$round" "$order"
      run_committed "$log" env \
        LD_LIBRARY_PATH="$(dirname "$library")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$binary" "${run_args[@]}" || return $?
      grep '^FQ_KQUANT_LAYOUT_RUN ' "$log" || {
        fail "q$q round$round lacks completion marker"; return 2; }
    done
  done

  python3 -B "$analyzer" analyze --plan "$plan" --runs "$out/runs" \
    --output "$out/results" --rounds "$rounds" --iterations "$iterations" \
    --threshold-pct "$threshold" --all-configs "$all_configs" || return 2
  if [ "$all_configs" = 1 ]; then
    python3 -B "$fitter" fit \
      --summary "$out/results/summary.json" \
      --output "$out/results/config-heuristic.json" \
      --regret-threshold-pct "$threshold" \
      --max-leaves "${HEURISTIC_MAX_LEAVES:-8}" \
      --min-leaf-rows "${HEURISTIC_MIN_LEAF_ROWS:-2}" \
      --min-leaf-families "${HEURISTIC_MIN_LEAF_FAMILIES:-1}" \
      | tee "$out/results/config-heuristic.log" || return 2
    sha256sum "$out/results/summary.json" \
      "$out/results/config-heuristic.json" \
      > "$out/results/config-heuristic.sha256" || return 2
  fi
  printf '[fq-kquant-perf] DIAGNOSTIC_COMPLETE sha=%s artifacts=%s\n' "$sha" "$out"
  printf '[fq-kquant-perf] summary=%s\n' "$out/results/summary.tsv"
}

main "$@"
