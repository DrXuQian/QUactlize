#!/usr/bin/env bash
# Same-metadata/same-persistent-driver Q4 prefill A/B.
#
# Arms:
#   Xplane A64 + FP16 ScaleFirst metadata
#   K-pack4      + FP16 ScaleFirst metadata
#
# Only M=2048/4096 and PERSISTENT full-output are measured.  M=64 and every
# Split-K board are deliberately outside this causal experiment.
set -uo pipefail

fail() {
  printf '[sf-kpack4-prefill-ab] FAIL: %s\n' "$*" >&2
  return 2
}

atomic_text() {
  local destination="$1" value="$2" current
  current="${destination}.current.$$"
  printf '%s\n' "$value" > "$current" || return 2
  mv -f -- "$current" "$destination"
}

run_phase() {
  local log="$1" commit="${1}.sha256" current failed digest rc
  shift
  if [ -s "$log" ] && [ -s "$commit" ]; then
    digest="$(sha256sum "$log" | awk '{print $1}')" || return 2
    [ "$(cat "$commit")" = "$digest" ] || {
      fail "committed phase changed: $log"; return 2; }
    [ "${RESUME:-0}" = 1 ] || {
      fail "phase exists without RESUME=1: $log"; return 2; }
    printf '[sf-kpack4-prefill-ab] reuse phase=%s\n' "$log"
    return 0
  fi
  if [ -e "$log" ] || [ -e "$commit" ]; then
    [ "${RESUME:-0}" = 1 ] || {
      fail "phase residue exists without RESUME=1: $log"; return 2; }
    failed="${log}.uncommitted.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    [ ! -e "$log" ] || mv -- "$log" "$failed" || return 2
    [ ! -e "$commit" ] || mv -- "$commit" "${failed}.sha256" || return 2
  fi
  current="${log}.current.$$"
  "$@" > "$current" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    failed="${log}.failed.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mv -- "$current" "$failed" || return 2
    tail -n 180 "$failed" >&2
    fail "phase rc=$rc preserved=$failed"
    return "$rc"
  fi
  mv -- "$current" "$log" || return 2
  digest="$(sha256sum "$log" | awk '{print $1}')" || return 2
  atomic_text "$commit" "$digest"
}

main() {
  local root workspace sha short stamp out jobs iterations rounds repeats
  local threshold selector analyzer generated arm layout artifact build_dir
  local build_log target_make binary rc shape_key m n k round order name log
  local source_state saved_state
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-sf-q4k-kpack4-prefill-ab-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  case "${RESUME:-0}" in 0|1) ;; *) fail 'RESUME must be 0 or 1'; return 2;; esac
  if [ -e "$out" ] && [ "${RESUME:-0}" != 1 ]; then
    fail "refusing existing OUT without RESUME=1: $out"; return 2
  fi
  if [ ! -e "$out" ] && [ "${RESUME:-0}" = 1 ]; then
    fail "RESUME=1 requires an existing OUT: $out"; return 2
  fi
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ] || \
     [ -n "${SCALEFIRST_SWEEP_WEIGHT_LAYOUT:-}" ]; then
    fail 'ambient PPU definitions/layout override changes the A/B'; return 2
  fi
  jobs="${JOBS:-16}"
  iterations="${PERF_ITERATIONS:-21}"
  rounds="${PERF_ROUNDS:-2}"
  repeats="${CORRECTNESS_REPEATS:-4}"
  threshold="${REGRESSION_THRESHOLD_PCT:-3.0}"
  case "$jobs:$iterations:$rounds:$repeats" in
    *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0)
      fail 'JOBS/iterations/rounds/repeats must be positive integers'; return 2;;
  esac
  python3 -B - "$threshold" <<'PY' || return 2
import sys
value=float(sys.argv[1])
assert 0 < value < 100
PY

  selector="$root/tools/select_scalefirst_q4k_kpack4_prefill_ab.py"
  analyzer="$root/tools/analyze_scalefirst_q4k_kpack4_prefill_ab.py"
  generated="$out/generated"
  mkdir -p "$generated" "$out/build" "$out/runs" "$out/results" || return 2

  python3 -B "$selector" self-test || return 2
  python3 -B "$analyzer" self-test >/dev/null || return 2
  python3 -B "$root/ci/check_scalefirst_q4k_kpack4_prefill_ab.py" || return 2
  if [ ! -s "$generated/manifest.json" ]; then
    python3 -B "$selector" materialize --out-dir "$generated" || return 2
  fi
  python3 -B - "$generated/manifest.json" <<'PY' || return 2
import json,sys
value=json.load(open(sys.argv[1]))
assert value['schema']=='quactlize.scalefirst-q4k-kpack4-prefill-ab.v1'
assert value['shapes']==[[2048,1024,5120],[4096,1024,5120]]
assert len(value['arms'])==2
assert all(a['metadata']=='scalefirst-fp16-scale-zero' and
           a['algorithms']==['PERSISTENT'] and
           len(a['typed_rows'])==3 for a in value['arms'])
PY
  source_state="$({
    git -C "$root" rev-parse HEAD
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum "$generated/manifest.json" "$generated"/*/manifest.json \
      "$selector" "$analyzer" \
      "$root/benchmarks/scalefirst_internal_sweep_bench.hpp" \
      "$root/benchmarks/test_scalefirst_internal_sweep.cu" \
      "$root/quactlize/csrc/scalefirst_internal_sweep.cmake.in" \
      "$root/tools/run_scalefirst_q4k_kpack4_prefill_ab_box.sh"
  } | sha256sum | awk '{print $1}')" || return 2
  saved_state="$out/source-state.sha256"
  if [ -s "$saved_state" ] && [ "$(cat "$saved_state")" != "$source_state" ]; then
    fail 'source/generated authority changed on resume'; return 2
  fi
  atomic_text "$saved_state" "$source_state" || return 2

  for arm in xplane q4-kpack4; do
    if [ "$arm" = xplane ]; then layout=0; artifact=64
    else layout=1; artifact=0
    fi
    build_dir="$out/build/$arm"
    build_log="$out/results/build-$arm.log"
    if [ "${RESUME:-0}" = 1 ] && \
       [ -s "$out/results/binary-$arm.path" ] && \
       [ -s "$out/results/binary-$arm.sha256" ]; then
      binary="$(cat "$out/results/binary-$arm.path")"
      [ -x "$binary" ] && [ ! -L "$binary" ] && \
        sha256sum -c "$out/results/binary-$arm.sha256" >/dev/null || {
          fail "$arm resumed binary/hash differs"; return 2; }
      printf '[sf-kpack4-prefill-ab] reuse build arm=%s\n' "$arm"
    else
      printf '[sf-kpack4-prefill-ab] build arm=%s layout=%s A=%s rows=3\n' \
        "$arm" "$layout" "$artifact"
      (cd "$root" && env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
        -u CC -u CXX PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 \
        JOBS="$jobs" TARGET=test_scalefirst_internal_sweep \
        SCALEFIRST_SWEEP_GENERATED_DIR="$generated/$arm" \
        SCALEFIRST_SWEEP_QTYPE=12 SCALEFIRST_SWEEP_ARTIFACT_TK="$artifact" \
        SCALEFIRST_SWEEP_BCHUNK=0 SCALEFIRST_SWEEP_WEIGHT_LAYOUT="$layout" \
        PPU_DEFS= PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
        ./build.sh) > "$build_log" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
        tail -n 180 "$build_log" >&2
        fail "$arm build rc=$rc artifacts=$out"; return "$rc"
      fi
      target_make="$(find "$build_dir" -type f \
        -path '*test_scalefirst_internal_sweep.dir/build.make' -print -quit)"
      binary="$(find "$build_dir" -type f -name test_scalefirst_internal_sweep \
        -perm -u+x -print -quit)"
      [ -n "$target_make" ] && [ -n "$binary" ] && [ ! -L "$binary" ] || {
        fail "$arm build identity is incomplete"; return 2; }
      grep -Fqx "[build.sh] SCALEFIRST_SWEEP_WEIGHT_LAYOUT=$layout" "$build_log" && \
        grep -F "ScaleFirst internal sweep: q=12 A=$artifact bc=0 layout=$layout units=1" \
          "$build_dir/cmake.log" >/dev/null && \
        grep -Eq "^SCALEFIRST_SWEEP_WEIGHT_LAYOUT(:[^=]*)?=$layout$" \
          "$build_dir/CMakeCache.txt" && \
        grep -Eq -- "(^|[[:space:]])-DSCALEFIRST_SWEEP_WEIGHT_LAYOUT=$layout([[:space:]]|$)" \
          "$target_make" && \
        grep -Eq -- '(^|[[:space:]])-DPPU_PACKED_SCALE=0([[:space:]]|$)' \
          "$target_make" || {
        fail "$arm ScaleFirst/layout build ABI differs"; return 2; }
      printf '%s\n' "$binary" > "$out/results/binary-$arm.path" || return 2
      sha256sum "$binary" > "$out/results/binary-$arm.sha256" || return 2
      sha256sum "$target_make" "$generated/$arm/manifest.json" \
        "$generated/$arm/units/"*.cu \
        > "$out/results/build-identity-$arm.sha256" || return 2
    fi
  done

  {
    git -C "$root" rev-parse HEAD
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum "$generated/manifest.json" "$selector" "$analyzer" \
      "$root/benchmarks/scalefirst_internal_sweep_bench.hpp" \
      "$root/benchmarks/test_scalefirst_internal_sweep.cu" \
      "$root/quactlize/csrc/scalefirst_internal_sweep.cmake.in" \
      "$root/tools/run_scalefirst_q4k_kpack4_prefill_ab_box.sh"
  } > "$out/source-authority.sha256" || return 2
  git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2

  while IFS=$'\t' read -r shape_key m n k; do
    mkdir -p "$out/runs/$shape_key" || return 2
    for round in $(seq 1 "$rounds"); do
      if [ $((round % 2)) -eq 1 ]; then
        order='xplane q4-kpack4'
      else
        order='q4-kpack4 xplane'
      fi
      for name in $order; do
        binary="$(cat "$out/results/binary-$name.path")"
        log="$out/runs/$shape_key/round-$round-$name.log"
        printf '[sf-kpack4-prefill-ab] shape=%sx%sx%s round=%s arm=%s algorithm=PERSISTENT\n' \
          "$m" "$n" "$k" "$round" "$name"
        run_phase "$log" "$binary" --shape="${m}x${n}x${k}" \
          --iterations="$iterations" --correctness-repeats="$repeats" \
          --algorithm=persistent --fixture=exact || return $?
      done
    done
  done <<'EOF'
m2048_n1024_k5120	2048	1024	5120
m4096_n1024_k5120	4096	1024	5120
EOF

  python3 -B "$analyzer" analyze --runs "$out/runs" \
    --rounds "$rounds" --iterations "$iterations" \
    --threshold-pct "$threshold" \
    --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" | tee "$out/results/summary.log" || return 2
  sha256sum "$out/results/summary.json" "$out/results/summary.tsv" \
    "$out"/runs/*/*.log > "$out/results/authority.sha256" || return 2
  printf '[sf-kpack4-prefill-ab] PASS sha=%s shapes=M2048,M4096 metadata=ScaleFirst-FP16 algorithm=PERSISTENT artifacts=%s\n' \
    "$sha" "$out"
  printf '[sf-kpack4-prefill-ab] summary=%s\n' "$out/results/summary.tsv"
}

main "$@"
