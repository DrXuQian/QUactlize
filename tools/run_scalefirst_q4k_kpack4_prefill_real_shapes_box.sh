#!/usr/bin/env bash
# Product-path Q4_K prefill A/B on the complete real-shape denominator.
#
# Both arms use ScaleFirst FP16 scale/zero metadata, the same seven tactics,
# PERSISTENT full-output execution and the exact fixture.  Only the resident
# weight byte map differs: historical Xplane A64 versus K-pack4 transpose-v1.
set -uo pipefail

fail() {
  printf '[sf-kpack4-prefill-real] FAIL: %s\n' "$*" >&2
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
    printf '[sf-kpack4-prefill-real] reuse phase=%s\n' "$log"
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
  if [ "$#" -ne 0 ]; then
    fail 'no positional arguments are accepted'; return 2
  fi
  local root workspace sha short stamp out resume jobs iterations rounds repeats
  local threshold inventory frozen_inventory master frozen_master plan shape_rows
  local selector analyzer generated source_state saved_state arm layout artifact
  local build_dir build_log target_make binary rc shape_key m n k round order name log

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-sf-q4k-kpack4-prefill-real-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  resume="${RESUME:-0}"
  case "$resume" in 0|1) ;; *) fail 'RESUME must be 0 or 1'; return 2;; esac
  if [ -e "$out" ] && [ "$resume" != 1 ]; then
    fail "refusing existing OUT without RESUME=1: $out"; return 2
  fi
  if [ ! -e "$out" ] && [ "$resume" = 1 ]; then
    fail "RESUME=1 requires an existing OUT: $out"; return 2
  fi
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ] || \
     [ -n "${SCALEFIRST_SWEEP_WEIGHT_LAYOUT:-}" ]; then
    fail 'ambient PPU definitions/layout override changes the A/B'; return 2
  fi

  inventory="${INTERNAL_SWEEP_SPEC:-}"
  if [ -z "$inventory" ] || [ ! -s "$inventory" ]; then
    fail 'INTERNAL_SWEEP_SPEC must name the COMPLETE inventory-v2 JSON'; return 2
  fi
  inventory="$(realpath -e -- "$inventory")" || return 2
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

  selector="$root/tools/select_scalefirst_q4k_kpack4_prefill_real_shapes.py"
  analyzer="$root/tools/analyze_scalefirst_q4k_kpack4_prefill_real_shapes.py"
  master="$root/benchmarks/scalefirst_q4k_real_shapes_pruned_policy.json"
  mkdir -p "$out/inputs" "$out/policies" "$out/generated" "$out/build" \
    "$out/runs" "$out/results" || return 2
  frozen_inventory="$out/inputs/inventory-v2.json"
  frozen_master="$out/inputs/scalefirst_q4k_real_shapes_pruned_policy.json"
  plan="$out/plan.json"
  shape_rows="$out/shape-rows.tsv"
  generated="$out/generated"

  python3 -B - "$inventory" "$frozen_inventory" "$master" "$frozen_master" \
    "$resume" <<'PY' || return 2
import os,pathlib,sys
resume=sys.argv[5]=='1'
for source_name,frozen_name,label in (
    (sys.argv[1],sys.argv[2],'inventory-v2'),
    (sys.argv[3],sys.argv[4],'master policy')):
    source=pathlib.Path(source_name); frozen=pathlib.Path(frozen_name)
    data=source.read_bytes()
    if not data: raise SystemExit(f'{label} is empty')
    if frozen.exists():
        if frozen.read_bytes()!=data:
            raise SystemExit(f'{label} differs from frozen bundle')
    elif resume:
        raise SystemExit(f'resume bundle lost frozen {label}')
    else:
        temporary=frozen.with_name(f'.{frozen.name}.current.{os.getpid()}')
        temporary.write_bytes(data); os.replace(temporary,frozen)
PY

  python3 -B "$selector" self-test || return 2
  python3 -B "$analyzer" self-test >/dev/null || return 2
  python3 -B "$root/ci/check_scalefirst_q4k_kpack4_prefill_real_shapes.py" || return 2
  python3 -B "$root/tools/plan_scalefirst_q4k_real_shapes.py" self-test >/dev/null || return 2
  if [ ! -s "$plan" ]; then
    [ "$resume" != 1 ] || { fail 'resume bundle lost plan.json'; return 2; }
    python3 -B "$root/tools/plan_scalefirst_q4k_real_shapes.py" materialize \
      --inventory "$frozen_inventory" --master-policy "$frozen_master" \
      --output "$plan" --policies-dir "$out/policies" || return 2
  fi
  python3 -B - "$plan" "$shape_rows" <<'PY' || return 2
import json,pathlib,sys
plan=json.load(open(sys.argv[1])); out=pathlib.Path(sys.argv[2])
families=((1024,5120),(5120,8192),(5120,25600),(8192,5120),(25600,5120))
expected={(m,n,k) for n,k in families for m in (64,2048,4096)}
if plan.get('schema')!='quactlize.scalefirst_q4k_real_shapes_plan.v1':
    raise SystemExit('real-shape plan schema differs')
observed={(int(x['m']),int(x['n']),int(x['k'])) for x in plan.get('shapes',[])}
if observed!=expected or plan.get('shape_count')!=15:
    raise SystemExit(f'real prefill shape denominator differs missing={sorted(expected-observed)} extra={sorted(observed-expected)}')
lines=[]
for m,n,k in sorted(observed, key=lambda x:(x[1],x[2],x[0])):
    lines.append(f'm{m}_n{n}_k{k}\t{m}\t{n}\t{k}')
out.write_text('\n'.join(lines)+'\n')
PY
  if [ ! -s "$generated/manifest.json" ]; then
    [ "$resume" != 1 ] || { fail 'resume bundle lost generated manifest'; return 2; }
    python3 -B "$selector" materialize --out-dir "$generated" || return 2
  fi
  python3 -B - "$generated/manifest.json" "$shape_rows" <<'PY' || return 2
import json,sys
value=json.load(open(sys.argv[1]))
rows=[line.rstrip('\n').split('\t') for line in open(sys.argv[2])]
shapes=[[int(row[1]),int(row[2]),int(row[3])] for row in rows]
assert value['schema']=='quactlize.scalefirst-q4k-kpack4-prefill-real.v2'
assert value['shapes']==shapes and len(shapes)==15
assert len(value['arms'])==2
assert all(a['metadata']=='scalefirst-fp16-scale-zero' and
           a['algorithms']==['PERSISTENT'] and
           len(a['typed_rows'])==7 for a in value['arms'])
PY

  source_state="$({
    git -C "$root" rev-parse HEAD
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum "$frozen_inventory" "$frozen_master" "$plan" "$shape_rows" \
      "$generated/manifest.json" "$generated"/*/manifest.json \
      "$selector" "$analyzer" \
      "$root/tools/select_scalefirst_q4k_kpack4_prefill_ab.py" \
      "$root/tools/analyze_scalefirst_q4k_kpack4_prefill_ab.py" \
      "$root/tools/plan_scalefirst_q4k_real_shapes.py" \
      "$root/ci/check_scalefirst_q4k_kpack4_prefill_real_shapes.py" \
      "$root/benchmarks/scalefirst_internal_sweep_bench.hpp" \
      "$root/benchmarks/test_scalefirst_internal_sweep.cu" \
      "$root/quactlize/csrc/scalefirst_internal_sweep.cmake.in" \
      "$root/tools/run_scalefirst_q4k_kpack4_prefill_real_shapes_box.sh"
  } | sha256sum | awk '{print $1}')" || return 2
  saved_state="$out/source-state.sha256"
  if [ -s "$saved_state" ] && [ "$(cat "$saved_state")" != "$source_state" ]; then
    fail 'source/generated/inventory authority changed on resume'; return 2
  fi
  atomic_text "$saved_state" "$source_state" || return 2

  for arm in xplane q4-kpack4; do
    if [ "$arm" = xplane ]; then layout=0; artifact=64
    else layout=1; artifact=0
    fi
    build_dir="$out/build/$arm"
    build_log="$out/results/build-$arm.log"
    if [ "$resume" = 1 ] && [ -s "$out/results/binary-$arm.path" ] && \
       [ -s "$out/results/binary-$arm.sha256" ]; then
      binary="$(cat "$out/results/binary-$arm.path")"
      [ -x "$binary" ] && [ ! -L "$binary" ] && \
        sha256sum -c "$out/results/binary-$arm.sha256" >/dev/null || {
          fail "$arm resumed binary/hash differs"; return 2; }
      printf '[sf-kpack4-prefill-real] reuse build arm=%s\n' "$arm"
    else
      printf '[sf-kpack4-prefill-real] build arm=%s layout=%s A=%s rows=7\n' \
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
    sha256sum "$frozen_inventory" "$plan" "$generated/manifest.json" \
      "$selector" "$analyzer" \
      "$root/tools/run_scalefirst_q4k_kpack4_prefill_real_shapes_box.sh"
  } > "$out/source-authority.sha256" || return 2
  git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2

  while IFS=$'\t' read -r shape_key m n k; do
    mkdir -p "$out/runs/$shape_key" || return 2
    for round in $(seq 1 "$rounds"); do
      if [ $((round % 2)) -eq 1 ]; then order='xplane q4-kpack4'
      else order='q4-kpack4 xplane'
      fi
      for name in $order; do
        binary="$(cat "$out/results/binary-$name.path")"
        log="$out/runs/$shape_key/round-$round-$name.log"
        printf '[sf-kpack4-prefill-real] shape=%sx%sx%s round=%s arm=%s algorithm=PERSISTENT\n' \
          "$m" "$n" "$k" "$round" "$name"
        run_phase "$log" "$binary" --shape="${m}x${n}x${k}" \
          --iterations="$iterations" --correctness-repeats="$repeats" \
          --algorithm=persistent --fixture=exact || return $?
      done
    done
  done < "$shape_rows"

  python3 -B "$analyzer" analyze --runs "$out/runs" \
    --rounds "$rounds" --iterations "$iterations" \
    --threshold-pct "$threshold" \
    --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" | tee "$out/results/summary.log" || return 2
  sha256sum "$out/results/summary.json" "$out/results/summary.tsv" \
    "$out/results/family-summary.tsv" "$out"/runs/*/*.log \
    > "$out/results/authority.sha256" || return 2
  printf '[sf-kpack4-prefill-real] PASS sha=%s shapes=15 families=5 M=64,2048,4096 metadata=ScaleFirst-FP16 algorithm=PERSISTENT artifacts=%s\n' \
    "$sha" "$out"
  printf '[sf-kpack4-prefill-real] summary=%s\n' "$out/results/summary.tsv"
}

main "$@"
