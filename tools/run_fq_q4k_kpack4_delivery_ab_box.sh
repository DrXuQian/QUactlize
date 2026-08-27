#!/usr/bin/env bash
# Compare matched K-pack4 resident delivery caps auto64/D32/D16 on PPU0010.
set -uo pipefail

fail() {
  printf '[fq-kpack4-delivery-ab] FAIL: %s\n' "$*" >&2
  return 2
}

resolve_executable() {
  local candidate="$1"
  [ -n "$candidate" ] || return 1
  if [[ "$candidate" == */* ]]; then
    [ -x "$candidate" ] && { readlink -f "$candidate"; return 0; }
  else
    command -v "$candidate" 2>/dev/null && return 0
  fi
  return 1
}

main() {
  local root workspace sha short stamp out jobs per_unit iterations rounds repeats
  local threshold run_acu sdk_root hgcc hgobjdump acu selector analyzer base_analyzer
  local source_x source_k selected ap delivery tag arm generated build_dir build_log
  local binary target_make symbol_file unit registry list_elf symbol demangled line resource
  local tmp_codegen round order log rc report_base report details raw_details raw_value
  local -a acu_cmd reports

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-kpack4-delivery-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  [ ! -e "$out" ] || { fail "refusing existing OUT: $out"; return 2; }
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS changes the factorial'; return 2
  fi

  jobs="${JOBS:-16}"
  per_unit="${FQ_CONFIGS_PER_UNIT:-144}"
  iterations="${PERF_ITERATIONS:-201}"
  rounds="${PERF_ROUNDS:-3}"
  repeats="${CORRECTNESS_REPEATS:-64}"
  threshold="${MATERIAL_THRESHOLD:-0.02}"
  run_acu="${RUN_ACU:-1}"
  case "$jobs:$per_unit:$iterations:$rounds:$repeats" in
    *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0:*|*:*:*:*:0)
      fail 'JOBS/FQ_CONFIGS_PER_UNIT/PERF_ITERATIONS/PERF_ROUNDS/CORRECTNESS_REPEATS must be positive'; return 2;;
  esac
  [ "$rounds" -eq 3 ] || { fail 'PERF_ROUNDS is frozen to 3 for cyclic order'; return 2; }
  case "$run_acu" in 0|1) ;; *) fail 'RUN_ACU must be 0 or 1'; return 2;; esac
  python3 -B - "$threshold" <<'PY' || return 2
import sys
assert 0 < float(sys.argv[1]) < 1
PY

  sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
  hgcc="$(resolve_executable "${HGCC:-${sdk_root:+$sdk_root/bin/hgcc}}" || true)"
  hgobjdump="$(resolve_executable "${HGOBJDUMP:-${sdk_root:+$sdk_root/bin/hgobjdump}}" || true)"
  [ -n "$hgcc" ] || hgcc="$(resolve_executable "$(command -v hgcc 2>/dev/null || true)" || true)"
  [ -n "$hgobjdump" ] || hgobjdump="$(resolve_executable "$(command -v hgobjdump 2>/dev/null || true)" || true)"
  if [ -z "$hgcc" ] || [ -z "$hgobjdump" ]; then
    fail 'real hgcc and hgobjdump are required'; return 2
  fi
  [ -n "$sdk_root" ] || sdk_root="$(cd "$(dirname "$hgcc")/.." && pwd)"
  sdk_root="$(cd "$sdk_root" && pwd)" || return 2
  if [ "$(resolve_executable "$sdk_root/bin/hgcc" || true)" != "$hgcc" ] ||
     [ "$(resolve_executable "$sdk_root/bin/hgobjdump" || true)" != "$hgobjdump" ]; then
    fail "hgcc/hgobjdump are not both owned by PPU_SDK=$sdk_root"; return 2
  fi
  if [[ "$($hgcc --version 2>&1 | head -n 1 || true)" == *stub* ]] ||
     [[ "$($hgobjdump --version 2>&1 | head -n 1 || true)" == *stub* ]]; then
    fail 'stub compiler/disassembler is forbidden'; return 2
  fi

  selector="$root/tools/select_fq_q4k_kpack4_xplane_isomorphic_ab.py"
  analyzer="$root/tools/analyze_fq_q4k_kpack4_delivery_ab.py"
  base_analyzer="$root/tools/analyze_fq_q4k_kpack4_xplane_isomorphic_ab.py"
  source_x="$out/generated/source-xplane"
  source_k="$out/generated/source-kpack4"
  selected="$out/generated/ab"
  mkdir -p "$source_x" "$source_k" "$selected" "$out/build" "$out/codegen" \
    "$out/runs" "$out/results" "$out/acu" || return 2

  python3 -B "$root/ci/check_fq_q4k_kpack4_generator.py" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_delivery_ab.py" || return 2
  python3 -B "$selector" self-test || return 2
  python3 -B "$analyzer" self-test || return 2
  python3 -B "$root/ci/local_gates.py" \
    -k l229_q4_kpack4_production_type --strict || return 2
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

  git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2
  {
    printf '%s\n' "$sha"
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum "$hgcc" "$hgobjdump" \
      "$root/benchmarks/test_fully_quantized_internal_sweep.cu" \
      "$root/benchmarks/fully_quantized_splitk_producer_bench.hpp" \
      "$root/benchmarks/fully_quantized_splitk_producer_unit.inc" \
      "$root/ci/local_gates.py" \
      "$root/quactlize/include/ppu_mixed_policy.hpp" \
      "$root/quactlize/include/fpA_intB_ppu.cuh" \
      "$root/quactlize/include/q4_kpack4_offline.hpp" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp" \
      "$root/dev/fold_derivation/l229_q4_kpack4_production_type.cu" \
      "$root/dev/fold_derivation/l231_q4_kpack4_production_fragment.cu" \
      "$root/dev/fold_derivation/run_l231_q4_kpack4_production_fragment.sh" \
      "$selector" "$base_analyzer" "$analyzer" \
      "$root/ci/check_fq_q4k_kpack4_delivery_ab.py" \
      "$root/tools/run_fq_q4k_kpack4_delivery_ab_box.sh"
  } > "$out/source-authority.sha256" || return 2

  for ap in 0 1; do
    generated="$selected/kpack4-ap$ap"
    symbol_file="$out/results/row-symbol-ap$ap.txt"
    unit="$(python3 -B - "$generated/manifest.json" "$symbol_file" "$ap" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); ap=int(sys.argv[3])
v=json.loads(p.read_text()); r=v['row']
provider='packed-row' if ap else 'standard-aiu'
assert v['schema']=='quactlize.fq-q4k-kpack4-xplane-isomorphic-arm.v1'
assert v['name']==f'kpack4-ap{ap}' and v['selection_denominator']==1
assert v['artifact_tile_k']==0 and v['weight_layout']==1
assert v['a_provider_id']==ap and v['a_provider']==provider
assert (r['tile_m'],r['tile_n'],r['tactic_tile_k'],r['warp_m'],r['warp_n'],r['stages'])==(8,64,256,8,16,2)
assert r['a_provider']==provider and len(v['units'])==1
unit=pathlib.Path(v['units'][0]); assert unit.is_file()
out.write_text(r['symbol']+'\n'); print(unit)
PY
    )" || return 2
    registry="$generated/fq_tc_registry.inc"
    for delivery in 0 32 16; do
      case "$delivery" in 0) tag=auto64;; 32) tag=d32;; 16) tag=d16;; esac
      arm="kpack4-ap${ap}-${tag}"
      build_dir="$out/build/$arm"
      build_log="$out/results/build-${arm}.log"
      mkdir -p "$build_dir" "$out/codegen/$arm" || return 2
      printf '[fq-kpack4-delivery-ab] build arm=%s delivery_cap_n=%s\n' "$arm" "$delivery"
      (cd "$root" && env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE -u CC -u CXX \
        PPU_SDK="$sdk_root" PPU_HOME= PPU_SDK_SITE_DEFAULT= \
        PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
        TARGET=test_fully_quantized_internal_sweep \
        FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
        FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0 \
        FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1 \
        PPU_DEFS="FQ_TC_KPACK4_DELIVERY_N=$delivery" PPU_EXTRA_DEFS= \
        CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= ./build.sh) > "$build_log" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
        tail -n 180 "$build_log" >&2
        fail "$arm build rc=$rc artifacts=$out"; return "$rc"
      fi
      binary="$(find "$build_dir" -type f -name test_fully_quantized_internal_sweep -perm -u+x -print -quit)"
      target_make="$(find "$build_dir" -type f -path '*test_fully_quantized_internal_sweep.dir/build.make' -print -quit)"
      if [ -z "$binary" ] || [ -L "$binary" ] || [ -z "$target_make" ]; then
        fail "$arm binary/build identity is missing"; return 2
      fi
      if [ "$(grep -Eo -- '-DFQ_TC_KPACK4_DELIVERY_N=[0-9]+' "$target_make" | sort -u | tr '\n' ' ')" != \
           "-DFQ_TC_KPACK4_DELIVERY_N=$delivery " ]; then
        fail "$arm delivery compile ABI differs"; return 2
      fi
      if ! grep -Fqx '[build.sh] FQ_SWEEP_WEIGHT_LAYOUT=1' "$build_log" ||
         ! grep -F 'FullyQuantized internal sweep: q=12 A=0 bc=0 format=0 layout=1 units=1' \
           "$build_dir/cmake.log" >/dev/null ||
         ! grep -Eq '^FQ_SWEEP_WEIGHT_LAYOUT(:[^=]*)?=1$' "$build_dir/CMakeCache.txt" ||
         ! grep -Eq -- '(^|[[:space:]])-DFQ_SWEEP_WEIGHT_LAYOUT=1([[:space:]]|$)' "$target_make" ||
         ! grep -F "$(basename "$unit")" "$target_make" >/dev/null; then
        fail "$arm generated-row/layout build ABI differs"; return 2
      fi
      printf '%s\n' "$binary" > "$out/results/binary-${arm}.path"
      sha256sum "$binary" "$target_make" "$generated/manifest.json" "$unit" "$registry" \
        > "$out/results/build-identity-${arm}.sha256" || return 2

      list_elf="$out/codegen/$arm/list-elf.txt"
      "$hgobjdump" -lelf "$binary" > "$list_elf" \
        2> "$out/codegen/$arm/list-elf.err" || { fail "$arm hgobjdump -lelf"; return 2; }
      symbol="$out/codegen/$arm/kernel-symbol.txt"
      demangled="$out/codegen/$arm/kernel-symbol-demangled.txt"
      python3 -B "$base_analyzer" select-symbol --list-elf "$list_elf" \
        --symbol-output "$symbol" --demangled-output "$demangled" || return 2
      line="$out/codegen/$arm/kernel-line.txt"
      resource="$out/codegen/$arm/resource-usage.txt"
      "$hgobjdump" -line "-func=$(cat "$symbol")" "$binary" > "$line" \
        2> "$out/codegen/$arm/kernel-line.err" || { fail "$arm line disassembly"; return 2; }
      "$hgobjdump" "-res-usage=$(cat "$symbol")" "$binary" > "$resource" \
        2> "$out/codegen/$arm/resource-usage.err" || { fail "$arm resource report"; return 2; }
      tmp_codegen="$out/codegen/$arm/base.json"
      python3 -B "$base_analyzer" codegen --arm-manifest "$generated/manifest.json" \
        --line "$line" --resource "$resource" --binary "$binary" \
        --symbol "$symbol" --demangled "$demangled" --output "$tmp_codegen" || return 2
      python3 -B - "$tmp_codegen" "$out/codegen/ap${ap}-${tag}.json" "$delivery" <<'PY' || return 2
import json,pathlib,sys
source=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2]); delivery=int(sys.argv[3])
value=json.loads(source.read_text()); value['delivery_cap_n']=delivery
output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
    done
  done

  for ap in 0 1; do
    mkdir -p "$out/runs/ap$ap" || return 2
    for round in 1 2 3; do
      case "$round" in
        1) order="auto64 d32 d16";;
        2) order="d16 auto64 d32";;
        3) order="d32 d16 auto64";;
      esac
      for tag in $order; do
        arm="kpack4-ap${ap}-${tag}"
        binary="$(cat "$out/results/binary-${arm}.path")"
        symbol_file="$out/results/row-symbol-ap$ap.txt"
        log="$out/runs/ap$ap/round-${round}-${tag}.log"
        printf '[fq-kpack4-delivery-ab] timing provider=AP%s arm=%s round=%s\n' \
          "$ap" "$tag" "$round"
        "$binary" --shape=1x8192x5120 --iterations="$iterations" \
          --correctness-repeats="$repeats" --only-split=4 --tm8-max-m=8 \
          --symbols-file="$symbol_file" --bc-mode=skip > "$log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then
          tail -n 120 "$log" >&2
          fail "timing provider=AP$ap arm=$tag rc=$rc artifacts=$out"; return "$rc"
        fi
      done
    done
  done

  python3 -B "$analyzer" analyze --master "$selected/manifest.json" \
    --runs-root "$out/runs" --codegen-root "$out/codegen" \
    --iterations "$iterations" --rounds "$rounds" --threshold "$threshold" \
    --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" | tee "$out/results/summary.log" || return 2

  if [ "$run_acu" = 1 ]; then
    acu="$(resolve_executable "${ACU:-$(command -v acu 2>/dev/null || true)}" || true)"
    [ -n "$acu" ] || { fail 'RUN_ACU=1 but acu is unavailable; set ACU to its absolute executable'; return 2; }
    sha256sum "$acu" > "$out/results/acu-tool.sha256" || return 2
    printf 'provider\tdelivery\tarm\treport\tdetails\traw\n' > "$out/results/acu-index.tsv"
    for ap in 0 1; do
      for tag in auto64 d32 d16; do
        arm="kpack4-ap${ap}-${tag}"
        binary="$(cat "$out/results/binary-${arm}.path")"
        symbol_file="$out/results/row-symbol-ap$ap.txt"
        report_base="$out/acu/${arm}.report"
        log="$out/acu/${arm}.log"
        printf '[fq-kpack4-delivery-ab] ACU provider=AP%s arm=%s\n' "$ap" "$tag"
        acu_cmd=("$acu" -f -o "$report_base" --set full "$binary" \
          --shape=1x8192x5120 --iterations=1 --correctness-repeats=1 \
          --only-split=4 --tm8-max-m=8 --symbols-file="$symbol_file" \
          --bc-mode=skip --profile-subject-only)
        printf '%q ' "${acu_cmd[@]}" > "$out/acu/${arm}.command"
        printf '\n' >> "$out/acu/${arm}.command"
        "${acu_cmd[@]}" > "$log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ] ||
           [ "$(grep -c '^FQ_PROFILE_SUBJECT .* launches=1 reducer_launches=0$' "$log" || true)" -ne 1 ]; then
          tail -n 120 "$log" >&2
          fail "ACU subject closure failed arm=$arm rc=$rc"; return 2
        fi
        reports=()
        [ -s "$report_base" ] && reports+=("$report_base")
        [ -s "${report_base}.acurep" ] && reports+=("${report_base}.acurep")
        [ "${#reports[@]}" -eq 1 ] || {
          fail "ACU produced ${#reports[@]} reports for $arm"; return 2; }
        report="${reports[0]}"
        details="$out/acu/${arm}.details.csv"
        "$acu" --import "$report" --csv --page details > "$details" || return 2
        [ -s "$details" ] || { fail "empty ACU details for $arm"; return 2; }
        raw_details="$out/acu/${arm}.raw.csv"
        raw_value=NONE
        if "$acu" --import "$report" --csv --page raw > "$raw_details" \
             2> "$out/acu/${arm}.raw.err" && [ -s "$raw_details" ]; then
          raw_value="$raw_details"
        else
          printf '[fq-kpack4-delivery-ab] NOTE: ACU raw page unavailable for arm=%s; report remains authoritative\n' \
            "$arm" >&2
        fi
        printf 'AP%s\t%s\t%s\t%s\t%s\t%s\n' "$ap" "$tag" "$arm" "$report" "$details" "$raw_value" \
          >> "$out/results/acu-index.tsv"
      done
    done
    python3 -B "$analyzer" acu --index "$out/results/acu-index.tsv" \
      --output-json "$out/results/acu-summary.json" \
      --output-tsv "$out/results/acu-summary.tsv" | tee "$out/results/acu-summary.log" || return 2
  fi

  find "$out" -type f ! -name bundle.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > "$out/bundle.sha256" || return 2
  cat "$out/results/summary.tsv"
  if [ -s "$out/results/acu-summary.tsv" ]; then
    awk -F '\t' 'NR==1 || $NF==1' "$out/results/acu-summary.tsv"
  fi
  printf '[fq-kpack4-delivery-ab] PASS sha=%s shape=1x8192x5120 config=8x64x256_w8x16_s2 S=4 arms=6 artifacts=%s\n' \
    "$sha" "$out"
}

main "$@"
