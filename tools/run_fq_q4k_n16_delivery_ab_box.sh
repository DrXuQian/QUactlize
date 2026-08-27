#!/usr/bin/env bash
# Q4_K K-pack4 native N64/8KiB versus four-way N16/2KiB delivery A/B.
set -uo pipefail

fail() { printf '[fq-n16-delivery-ab] FAIL: %s\n' "$*" >&2; return 2; }

main() {
  local root workspace sha short stamp out jobs per_unit iterations rounds threshold
  local source_x source_k selected plan selector analyzer base layout artifact ap
  local variant defs name generated build_dir build_log binary target_make rc
  local symbol_file unit registry shape_key m n k role round order log
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"; stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-n16-delivery-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  [ ! -e "$out" ] || { fail "refusing existing OUT: $out"; return 2; }
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS changes the A/B'; return 2
  fi
  jobs="${JOBS:-16}"; per_unit="${FQ_CONFIGS_PER_UNIT:-144}"
  iterations="${PERF_ITERATIONS:-101}"; rounds="${PERF_ROUNDS:-2}"
  threshold="${MATERIAL_THRESHOLD:-0.02}"
  case "$jobs:$per_unit:$iterations:$rounds" in
    *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0)
      fail 'JOBS/FQ_CONFIGS_PER_UNIT/PERF_ITERATIONS/PERF_ROUNDS must be positive'; return 2;;
  esac
  python3 -B - "$threshold" <<'PY' || return 2
import sys
assert 0 < float(sys.argv[1]) < 1
PY

  selector="$root/tools/select_fq_q4k_kpack4_xplane_isomorphic_ab.py"
  analyzer="$root/tools/analyze_fq_q4k_n16_delivery_ab.py"
  source_x="$out/generated/source-xplane"; source_k="$out/generated/source-kpack4"
  selected="$out/generated/ab"; plan="$out/plan.json"
  mkdir -p "$source_x" "$source_k" "$selected" "$out/build" \
    "$out/runs" "$out/results" || return 2

  python3 -B "$root/ci/check_fq_q4k_kpack4_generator.py" || return 2
  python3 -B "$selector" self-test || return 2
  python3 -B "$analyzer" self-test || return 2
  QUACTLIZE_L231_OUT="$out/host-l231" \
    bash "$root/dev/fold_derivation/run_l231_q4_kpack4_production_fragment.sh" || return 2
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
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp" \
      "$root/quactlize/include/q4_kpack4_offline.hpp" \
      "$root/dev/fold_derivation/l231_q4_kpack4_production_fragment.cu" \
      "$root/dev/fold_derivation/run_l231_q4_kpack4_production_fragment.sh" \
      "$selector" "$analyzer" "$root/tools/run_fq_q4k_n16_delivery_ab_box.sh"
  } > "$out/source-authority.sha256" || return 2
  git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2

  for base in xplane-ap0 xplane-ap1 kpack4-ap0 kpack4-ap1; do
    case "$base" in xplane-*) layout=0; artifact=64;; kpack4-*) layout=1; artifact=0;; esac
    case "$base" in *-ap0) ap=0;; *-ap1) ap=1;; esac
    generated="$selected/$base"; symbol_file="$out/results/row-symbol-${base}.txt"
    unit="$(python3 -B - "$generated/manifest.json" "$symbol_file" \
      "$artifact" "$layout" "$ap" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
artifact,layout,ap=map(int,sys.argv[3:]); v=json.loads(p.read_text()); r=v['row']
assert v['selection_denominator']==1 and v['artifact_tile_k']==artifact
assert v['weight_layout']==layout and v['a_provider_id']==ap
assert (r['tile_m'],r['tile_n'],r['tactic_tile_k'],r['warp_m'],r['warp_n'],r['stages'])==(8,64,256,8,16,2)
assert len(v['units'])==1 and pathlib.Path(v['units'][0]).is_file()
out.write_text(r['symbol']+'\n'); print(v['units'][0])
PY
    )" || return 2
    registry="$generated/fq_tc_registry.inc"
    if [ "$layout" -eq 0 ]; then variants="native"; else variants="native n16"; fi
    for variant in $variants; do
      defs=""; [ "$variant" = n16 ] && defs="PPU_Q4_KPACK4_N16_DELIVERY=1"
      name="$base-$variant"; build_dir="$out/build/$name"
      build_log="$out/results/build-${name}.log"; mkdir -p "$build_dir" || return 2
      printf '[fq-n16-delivery-ab] build arm=%s layout=%s provider=AP%s variant=%s\n' \
        "$name" "$layout" "$ap" "$variant"
      (cd "$root" && env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE -u CC -u CXX \
        PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
        TARGET=test_fully_quantized_internal_sweep \
        FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
        FQ_SWEEP_ARTIFACT_TK="$artifact" FQ_SWEEP_BCHUNK=0 \
        FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT="$layout" \
        PPU_DEFS="$defs" PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
        ./build.sh) > "$build_log" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then tail -n 180 "$build_log" >&2; fail "$name build rc=$rc artifacts=$out"; return "$rc"; fi
      binary="$(find "$build_dir" -type f -name test_fully_quantized_internal_sweep -perm -u+x -print -quit)"
      target_make="$(find "$build_dir" -type f -path '*test_fully_quantized_internal_sweep.dir/build.make' -print -quit)"
      [ -n "$binary" ] && [ ! -L "$binary" ] && [ -n "$target_make" ] || {
        fail "$name binary/build identity missing"; return 2; }
      if [ "$variant" = n16 ]; then
        grep -Eq -- '(^|[[:space:]])-DPPU_Q4_KPACK4_N16_DELIVERY=1([[:space:]]|$)' \
          "$target_make" || { fail "$name compile define missing"; return 2; }
      elif grep -F -- '-DPPU_Q4_KPACK4_N16_DELIVERY' "$target_make" >/dev/null; then
        fail "$name native control inherited N16 define"; return 2
      fi
      printf '%s\n' "$binary" > "$out/results/binary-${name}.path"
      sha256sum "$binary" "$target_make" "$generated/manifest.json" "$unit" "$registry" \
        > "$out/results/build-identity-${name}.sha256" || return 2
    done
  done

  while IFS=$'\t' read -r shape_key m n k role; do
    for ap in 0 1; do
      mkdir -p "$out/runs/$shape_key/ap$ap" || return 2
      for round in $(seq 1 "$rounds"); do
        if [ $((round % 2)) -eq 1 ]; then
          order="xplane kpack4-native kpack4-n16"
        else
          order="kpack4-n16 kpack4-native xplane"
        fi
        for variant in $order; do
          case "$variant" in
            xplane) base="xplane-ap$ap"; name="$base-native";;
            kpack4-native) base="kpack4-ap$ap"; name="$base-native";;
            kpack4-n16) base="kpack4-ap$ap"; name="$base-n16";;
          esac
          binary="$(cat "$out/results/binary-${name}.path")"
          symbol_file="$out/results/row-symbol-${base}.txt"
          log="$out/runs/$shape_key/ap$ap/round-${round}-${variant}.log"
          printf '[fq-n16-delivery-ab] timing shape=%s role=%s provider=AP%s arm=%s round=%s\n' \
            "$shape_key" "$role" "$ap" "$variant" "$round"
          "$binary" --shape="${m}x${n}x${k}" --iterations="$iterations" \
            --correctness-repeats=16 --only-split=4 --tm8-max-m=8 \
            --symbols-file="$symbol_file" --bc-mode=skip > "$log" 2>&1
          rc=$?
          if [ "$rc" -ne 0 ]; then tail -n 120 "$log" >&2; fail "timing $name/$shape_key rc=$rc"; return "$rc"; fi
        done
      done
    done
  done < <(python3 -B - "$plan" <<'PY'
import json,sys
for r in json.load(open(sys.argv[1]))['cases']:
 print(r['shape_key'],*r['shape'],r['role'],sep='\t')
PY
  )

  python3 -B "$analyzer" analyze --master "$selected/manifest.json" --plan "$plan" \
    --runs-root "$out/runs" --iterations "$iterations" --rounds "$rounds" \
    --threshold "$threshold" --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" | tee "$out/results/summary.log" || return 2
  find "$out" -type f ! -name bundle.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > "$out/bundle.sha256" || return 2
  cat "$out/results/summary.tsv"
  printf '[fq-n16-delivery-ab] PASS sha=%s comparisons=6 artifacts=%s\n' "$sha" "$out"
}

main "$@"
