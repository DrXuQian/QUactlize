#!/usr/bin/env bash
# Fixed Split-K scheduler A/B for the isomorphic Q4_K layout/provider rows.
#
# Factorial axes:
#   layout:       xplane / q4-kpack4
#   A provider:   standard-aiu / packed-row
#   grid spelling native (M,N,S) / N-on-x (N,M,S)
#
# Tactic and runtime are frozen at 8x64x256_w8x16_s2, S=4.  The target is
# M1,N8192,K5120; balanced and K-heavy shapes are controls.
set -uo pipefail

fail() {
  printf '[fq-grid-order-ab] FAIL: %s\n' "$*" >&2
  return 2
}

main() {
  local root workspace sha short stamp out jobs per_unit iterations rounds threshold
  local source_x source_k selected plan analyzer selector base layout artifact ap
  local schedule defs name generated build_dir build_log binary target_make rc
  local symbol_file unit registry shape_key m n k role round order log
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-grid-order-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  [ ! -e "$out" ] || { fail "refusing existing OUT: $out"; return 2; }
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS changes the factorial'; return 2
  fi
  jobs="${JOBS:-16}"
  per_unit="${FQ_CONFIGS_PER_UNIT:-144}"
  iterations="${PERF_ITERATIONS:-101}"
  rounds="${PERF_ROUNDS:-2}"
  threshold="${MATERIAL_THRESHOLD:-0.02}"
  case "$jobs:$per_unit:$iterations:$rounds" in
    *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0)
      fail 'JOBS/FQ_CONFIGS_PER_UNIT/PERF_ITERATIONS/PERF_ROUNDS must be positive'; return 2;;
  esac
  python3 -B - "$threshold" <<'PY' || return 2
import sys
x=float(sys.argv[1])
assert 0 < x < 1
PY

  analyzer="$root/tools/analyze_fq_q4k_grid_order_ab.py"
  selector="$root/tools/select_fq_q4k_kpack4_xplane_isomorphic_ab.py"
  source_x="$out/generated/source-xplane"
  source_k="$out/generated/source-kpack4"
  selected="$out/generated/ab"
  plan="$out/plan.json"
  mkdir -p "$source_x" "$source_k" "$selected" "$out/build" \
    "$out/runs" "$out/results" || return 2

  python3 -B "$root/ci/check_fq_fixed_splitk_n_on_x.py" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_generator.py" || return 2
  python3 -B "$selector" self-test || return 2
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_xplane_isomorphic_ab.py" \
    self-test || return 2
  python3 -B "$analyzer" self-test || return 2

  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --weight-layout xplane \
    --tile-m-filter 8 --per-unit "$per_unit" --out-dir "$source_x" || return 2
  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 0 --bchunk 0 --weight-layout q4-kpack4 \
    --tile-m-filter 8 --per-unit "$per_unit" --out-dir "$source_k" || return 2
  python3 -B "$selector" materialize --xplane-dir "$source_x" \
    --kpack4-dir "$source_k" --out-dir "$selected" || return 2
  python3 -B "$analyzer" plan --output "$plan" || return 2

  {
    printf '%s\n' "$sha"
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum \
      "$root/benchmarks/test_fully_quantized_internal_sweep.cu" \
      "$root/benchmarks/fully_quantized_splitk_producer_bench.hpp" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp" \
      "$root/quactlize/include/q4_kpack4_offline.hpp" \
      "$selector" "$analyzer" \
      "$root/ci/check_fq_fixed_splitk_n_on_x.py" \
      "$root/tools/run_fq_q4k_grid_order_ab_box.sh"
  } > "$out/source-authority.sha256" || return 2
  git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2

  for base in xplane-ap0 kpack4-ap0 xplane-ap1 kpack4-ap1; do
    case "$base" in
      xplane-*) layout=0; artifact=64;;
      kpack4-*) layout=1; artifact=0;;
    esac
    case "$base" in *-ap0) ap=0;; *-ap1) ap=1;; esac
    generated="$selected/$base"
    symbol_file="$out/results/row-symbol-${base}.txt"
    unit="$(python3 -B - "$generated/manifest.json" "$symbol_file" \
      "$artifact" "$layout" "$ap" <<'PY'
import json,pathlib,sys
manifest=pathlib.Path(sys.argv[1]); symbol=pathlib.Path(sys.argv[2])
artifact,layout,ap=map(int,sys.argv[3:])
v=json.loads(manifest.read_text()); row=v['row']
assert v['selection_denominator']==1 and v['artifact_tile_k']==artifact
assert v['weight_layout']==layout and v['a_provider_id']==ap
assert (row['tile_m'],row['tile_n'],row['tactic_tile_k'],row['warp_m'],
        row['warp_n'],row['stages'],row['bchunk'])==(8,64,256,8,16,2,0)
assert len(v['units'])==1
u=pathlib.Path(v['units'][0]); assert u.is_file()
symbol.write_text(row['symbol']+'\n')
print(u)
PY
    )" || return 2
    registry="$generated/fq_tc_registry.inc"
    [ -s "$unit" ] && [ -s "$registry" ] && [ -s "$symbol_file" ] || {
      fail "$base generated identity is incomplete"; return 2; }

    for schedule in native-grid n-on-x; do
      name="$base-$schedule"
      defs=""
      [ "$schedule" = n-on-x ] && defs="PPU_FIXED_SPLITK_N_ON_X=1"
      build_dir="$out/build/$name"
      build_log="$out/results/build-${name}.log"
      mkdir -p "$build_dir" || return 2
      printf '[fq-grid-order-ab] build arm=%s layout=%s provider=AP%s schedule=%s\n' \
        "$name" "$layout" "$ap" "$schedule"
      (cd "$root" && env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE \
        -u CC -u CXX PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 \
        JOBS="$jobs" TARGET=test_fully_quantized_internal_sweep \
        FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
        FQ_SWEEP_ARTIFACT_TK="$artifact" FQ_SWEEP_BCHUNK=0 \
        FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT="$layout" \
        PPU_DEFS="$defs" PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
        ./build.sh) > "$build_log" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
        tail -n 160 "$build_log" >&2
        fail "$name build rc=$rc artifacts=$out"; return "$rc"
      fi
      binary="$(find "$build_dir" -type f \
        -name test_fully_quantized_internal_sweep -perm -u+x -print -quit)"
      [ -n "$binary" ] && [ ! -L "$binary" ] || {
        fail "$name binary missing or symlinked"; return 2; }
      target_make="$(find "$build_dir" -type f \
        -path '*test_fully_quantized_internal_sweep.dir/build.make' -print -quit)"
      [ -n "$target_make" ] || { fail "$name target build.make missing"; return 2; }
      if [ "$schedule" = n-on-x ]; then
        grep -Eq -- '(^|[[:space:]])-DPPU_FIXED_SPLITK_N_ON_X=1([[:space:]]|$)' \
          "$target_make" || { fail "$name compile define missing"; return 2; }
      elif grep -F -- '-DPPU_FIXED_SPLITK_N_ON_X' "$target_make" >/dev/null; then
        fail "$name native control inherited N-on-x define"; return 2
      fi
      printf '%s\n' "$binary" > "$out/results/binary-${name}.path"
      sha256sum "$binary" "$target_make" "$generated/manifest.json" \
        "$unit" "$registry" > "$out/results/build-identity-${name}.sha256" || return 2
    done
  done

  while IFS=$'\t' read -r shape_key m n k role; do
    for ap in 0 1; do
      for layout in xplane kpack4; do
        base="$layout-ap$ap"
        mkdir -p "$out/runs/$shape_key/ap$ap/$layout" || return 2
        for round in $(seq 1 "$rounds"); do
          if [ $((round % 2)) -eq 1 ]; then
            order="native-grid n-on-x"
          else
            order="n-on-x native-grid"
          fi
          for schedule in $order; do
            name="$base-$schedule"
            binary="$(cat "$out/results/binary-${name}.path")"
            symbol_file="$out/results/row-symbol-${base}.txt"
            log="$out/runs/$shape_key/ap$ap/$layout/round-${round}-${schedule}.log"
            printf '[fq-grid-order-ab] timing shape=%s role=%s provider=AP%s layout=%s round=%s schedule=%s\n' \
              "$shape_key" "$role" "$ap" "$layout" "$round" "$schedule"
            "$binary" --shape="${m}x${n}x${k}" --iterations="$iterations" \
              --correctness-repeats=8 --only-split=4 --tm8-max-m=8 \
              --symbols-file="$symbol_file" --bc-mode=skip > "$log" 2>&1
            rc=$?
            if [ "$rc" -ne 0 ]; then
              tail -n 100 "$log" >&2
              fail "timing arm=$name shape=$shape_key rc=$rc"; return "$rc"
            fi
          done
        done
      done
    done
  done < <(python3 -B - "$plan" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))['cases']:
 print(row['shape_key'],*row['shape'],row['role'],sep='\t')
PY
  )

  python3 -B "$analyzer" analyze --master "$selected/manifest.json" \
    --plan "$plan" --runs-root "$out/runs" --iterations "$iterations" \
    --rounds "$rounds" --threshold "$threshold" \
    --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" | tee "$out/results/summary.log" || return 2

  find "$out" -type f ! -name bundle.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > "$out/bundle.sha256" || return 2
  sed -n '1,14p' "$out/results/summary.tsv"
  printf '[fq-grid-order-ab] PASS sha=%s comparisons=12 artifacts=%s\n' "$sha" "$out"
}

main "$@"
