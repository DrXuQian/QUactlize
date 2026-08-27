#!/usr/bin/env bash
# Exact Q4_K K-pack4 vs xplane A/B on PPU0010.
#
# Four one-row binaries form a 2x2 factorial:
#   weight layout = xplane A64 / canonical K-pack4
#   A provider    = standard-aiu / packed-row
#
# Timing fixes config=8x64x256_w8x16_s2 and S=4.  If a repeat-stable gap is
# at least ACU_GAP_THRESHOLD, one worst comparison per A provider is recaptured
# with exactly one producer launch per arm under ACU.
set -uo pipefail

fail() {
  printf '[fq-kpack4-xplane-ab] FAIL: %s\n' "$*" >&2
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
  local gap_threshold run_acu sdk_root hgcc hgobjdump acu analyzer selector
  local source_x source_k selected plan arm artifact layout generated build_dir
  local build_log binary symbol_file list_elf symbol demangled line resource rc
  local shape_key m n k ap round order name log report_base report details
  local resume measurement_sha changed target_make unit registry ap_id
  local -a acu_cmd reports
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-kpack4-xplane-ab-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  resume="${RESUME:-0}"
  case "$resume" in 0|1) ;; *) fail 'RESUME must be 0 or 1'; return 2;; esac
  if [ -e "$out" ] && [ "$resume" != 1 ]; then
    fail "refusing existing OUT without RESUME=1: $out"; return 2
  fi
  if [ ! -e "$out" ] && [ "$resume" = 1 ]; then
    fail "RESUME=1 requires an existing OUT: $out"; return 2
  fi
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS changes the A/B'; return 2
  fi
  jobs="${JOBS:-16}"
  per_unit="${FQ_CONFIGS_PER_UNIT:-144}"
  iterations="${PERF_ITERATIONS:-101}"
  rounds="${PERF_ROUNDS:-2}"
  repeats="${CORRECTNESS_REPEATS:-8}"
  gap_threshold="${ACU_GAP_THRESHOLD:-0.03}"
  run_acu="${RUN_ACU:-auto}"
  case "$jobs:$per_unit:$iterations:$rounds:$repeats" in
    *[!0-9:]*|0:*|*:0:*|*:*:0:*|*:*:*:0:*|*:*:*:*:0)
      fail 'JOBS/FQ_CONFIGS_PER_UNIT/PERF_ITERATIONS/PERF_ROUNDS/CORRECTNESS_REPEATS must be positive'; return 2;;
  esac
  case "$run_acu" in 0|1|auto) ;; *) fail 'RUN_ACU must be 0, 1, or auto'; return 2;; esac
  python3 -B - "$gap_threshold" <<'PY' || return 2
import sys
x=float(sys.argv[1])
assert 0 < x < 1
PY

  analyzer="$root/tools/analyze_fq_q4k_kpack4_xplane_isomorphic_ab.py"
  selector="$root/tools/select_fq_q4k_kpack4_xplane_isomorphic_ab.py"
  mkdir -p "$out/generated/source-xplane" "$out/generated/source-kpack4" \
    "$out/generated/ab" "$out/build" "$out/codegen" "$out/runs" \
    "$out/results" "$out/acu" || return 2

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
  sha256sum "$hgcc" "$hgobjdump" > "$out/results/sdk-tools.sha256" || return 2

  python3 -B "$root/ci/check_fq_q4k_kpack4_generator.py" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_xplane_isomorphic_ab.py" || return 2
  python3 -B "$selector" self-test || return 2
  python3 -B "$analyzer" self-test || return 2

  source_x="$out/generated/source-xplane"
  source_k="$out/generated/source-kpack4"
  selected="$out/generated/ab"
  plan="$out/plan.json"
  if [ "$resume" = 1 ]; then
    [ -s "$out/source-authority.sha256" ] || {
      fail 'resume source authority is missing'; return 2; }
    measurement_sha="$(sed -n '1p' "$out/source-authority.sha256")"
    [[ "$measurement_sha" =~ ^[0-9a-f]{40}$ ]] || {
      fail 'resume measurement SHA is malformed'; return 2; }
    git -C "$root" merge-base --is-ancestor "$measurement_sha" "$sha" || {
      fail 'resume measurement SHA is not an ancestor of current HEAD'; return 2; }
    while IFS= read -r changed; do
      case "$changed" in
        tools/analyze_fq_q4k_kpack4_xplane_isomorphic_ab.py|\
        tools/run_fq_q4k_kpack4_xplane_isomorphic_ab_box.sh|\
        ci/check_fq_q4k_kpack4_xplane_isomorphic_ab.py|\
        .codex/skills/ppu-cute-numeric-debug/references/q4-kpack4-fragment-destination.md) ;;
        *) fail "resume changed a measurement source: $changed"; return 2;;
      esac
    done < <(git -C "$root" diff --name-only "$measurement_sha..$sha")
    python3 -B "$analyzer" validate-inputs \
      --master "$selected/manifest.json" --plan "$plan" || return 2
    {
      printf 'measurement_sha=%s\nresume_sha=%s\n' "$measurement_sha" "$sha"
      sha256sum "$analyzer" \
        "$root/ci/check_fq_q4k_kpack4_xplane_isomorphic_ab.py" \
        "$root/tools/run_fq_q4k_kpack4_xplane_isomorphic_ab_box.sh" \
        "$root/.codex/skills/ppu-cute-numeric-debug/references/q4-kpack4-fragment-destination.md"
    } > "$out/analysis-resume.sha256" || return 2
    git -C "$root" diff --binary --no-ext-diff "$measurement_sha..$sha" -- \
      tools/analyze_fq_q4k_kpack4_xplane_isomorphic_ab.py \
      ci/check_fq_q4k_kpack4_xplane_isomorphic_ab.py \
      tools/run_fq_q4k_kpack4_xplane_isomorphic_ab_box.sh \
      .codex/skills/ppu-cute-numeric-debug/references/q4-kpack4-fragment-destination.md \
      > "$out/analysis-resume.patch" || return 2
    printf '[fq-kpack4-xplane-ab] resume measurement_sha=%s analysis_sha=%s\n' \
      "$measurement_sha" "$sha"
  else
    python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
      --qtype 12 --artifact-tk 64 --bchunk 0 --weight-layout xplane \
      --tile-m-filter 8 --per-unit "$per_unit" --out-dir "$source_x" || return 2
    python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
      --qtype 12 --artifact-tk 0 --bchunk 0 --weight-layout q4-kpack4 \
      --tile-m-filter 8 --per-unit "$per_unit" --out-dir "$source_k" || return 2
    python3 -B "$selector" materialize --xplane-dir "$source_x" \
      --kpack4-dir "$source_k" --out-dir "$selected" || return 2
    python3 -B "$analyzer" plan --output "$plan" || return 2
    git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2
    {
      printf '%s\n' "$sha"
      git -C "$root/third_party/actlize" rev-parse HEAD
      sha256sum \
        "$root/benchmarks/test_fully_quantized_internal_sweep.cu" \
        "$root/benchmarks/fully_quantized_splitk_producer_bench.hpp" \
        "$root/benchmarks/fully_quantized_splitk_producer_unit.inc" \
        "$root/quactlize/include/ppu_mixed_policy.hpp" \
        "$root/quactlize/include/fpA_intB_ppu.cuh" \
        "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp" \
        "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
        "$selector" "$analyzer" \
        "$root/ci/check_fq_q4k_kpack4_xplane_isomorphic_ab.py" \
        "$root/tools/run_fq_q4k_kpack4_xplane_isomorphic_ab_box.sh"
    } > "$out/source-authority.sha256" || return 2
  fi

  for arm in xplane-ap0 kpack4-ap0 xplane-ap1 kpack4-ap1; do
    case "$arm" in
      xplane-*) artifact=64; layout=0;;
      kpack4-*) artifact=0; layout=1;;
    esac
    case "$arm" in
      *-ap0) ap_id=0;;
      *-ap1) ap_id=1;;
    esac
    generated="$selected/$arm"
    build_dir="$out/build/$arm"
    build_log="$out/results/build-${arm}.log"
    mkdir -p "$build_dir" "$out/codegen/$arm" || return 2
    if [ "$resume" = 1 ] && [ -s "$out/results/binary-${arm}.path" ] && \
       [ -s "$out/results/binary-${arm}.sha256" ]; then
      binary="$(cat "$out/results/binary-${arm}.path")"
      [ -x "$binary" ] && [ ! -L "$binary" ] && \
        sha256sum -c "$out/results/binary-${arm}.sha256" >/dev/null || {
          fail "$arm resume binary/hash differs"; return 2; }
      printf '[fq-kpack4-xplane-ab] reuse arm=%s binary=%s\n' "$arm" "$binary"
    else
      printf '[fq-kpack4-xplane-ab] build arm=%s A=%s layout=%s\n' \
        "$arm" "$artifact" "$layout"
      (cd "$root" && env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE -u CC -u CXX \
        PPU_SDK="$sdk_root" PPU_HOME= PPU_SDK_SITE_DEFAULT= \
        PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
        TARGET=test_fully_quantized_internal_sweep \
        FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
        FQ_SWEEP_ARTIFACT_TK="$artifact" FQ_SWEEP_BCHUNK=0 \
        FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT="$layout" \
        PPU_DEFS= PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
        ./build.sh) > "$build_log" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
        tail -n 180 "$build_log" >&2
        fail "$arm build rc=$rc artifacts=$out"; return "$rc"
      fi
      binary="$(find "$build_dir" -type f -name test_fully_quantized_internal_sweep -perm -u+x -print -quit)"
      if [ -z "$binary" ] || [ -L "$binary" ]; then
        fail "$arm exact binary is missing or symlinked"; return 2
      fi
      printf '%s\n' "$binary" > "$out/results/binary-${arm}.path"
      sha256sum "$binary" > "$out/results/binary-${arm}.sha256" || return 2
    fi
    symbol_file="$out/results/row-symbol-${arm}.txt"
    unit="$(python3 -B - "$generated/manifest.json" "$symbol_file" \
      "$artifact" "$layout" "$arm" <<'PY'
import json,pathlib,sys
manifest=pathlib.Path(sys.argv[1])
symbol_out=pathlib.Path(sys.argv[2])
artifact,layout,arm=int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
ap=1 if arm.endswith('-ap1') else 0
provider='packed-row' if ap else 'standard-aiu'
value=json.loads(manifest.read_text())
assert value['schema']=='quactlize.fq-q4k-kpack4-xplane-isomorphic-arm.v1'
assert value['name']==arm and value['selection_denominator']==1
assert value['artifact_tile_k']==artifact and value['weight_layout']==layout
assert value['a_provider_id']==ap and value['a_provider']==provider
assert value['source_typed_denominator']==144
assert value['source_global_typed_denominator']==918
row=value['row']
expected={'qtype':12,'artifact_tile_k':artifact,'tile_m':8,'tile_n':64,
          'tactic_tile_k':256,'warp_m':8,'warp_n':16,'stages':2,
          'bchunk':0,'a_provider':provider}
assert all(row.get(k)==v for k,v in expected.items())
assert len(value['units'])==1
unit=pathlib.Path(value['units'][0])
registry=manifest.parent/'fq_tc_registry.inc'
macro=f"X({row['symbol']},12,{artifact},8,64,256,8,16,2,0,{ap})"
assert unit.is_file() and registry.is_file()
assert unit.read_text().count(macro)==1
assert registry.read_text().count(macro)==1
symbol_out.write_text(row['symbol']+'\n')
print(unit)
PY
    )" || return 2
    registry="$generated/fq_tc_registry.inc"
    target_make="$(find "$build_dir" -type f \
      -path '*test_fully_quantized_internal_sweep.dir/build.make' \
      -print -quit 2>/dev/null)"
    if ! grep -Fqx "[build.sh] FQ_SWEEP_WEIGHT_LAYOUT=$layout" "$build_log" ||
       ! grep -F "FullyQuantized internal sweep: q=12 A=$artifact bc=0 format=0 layout=$layout units=1" \
         "$build_dir/cmake.log" >/dev/null ||
       ! grep -Eq "^FQ_SWEEP_WEIGHT_LAYOUT(:[^=]*)?=$layout$" \
         "$build_dir/CMakeCache.txt" ||
       [ -z "$target_make" ] ||
       ! grep -Eq -- "(^|[[:space:]])-DFQ_SWEEP_WEIGHT_LAYOUT=$layout([[:space:]]|$)" \
         "$target_make" ||
       ! grep -F "$(basename "$unit")" "$target_make" >/dev/null; then
      fail "$arm generated-row/build ABI is not exact"
      grep -E 'FQ_SWEEP_WEIGHT_LAYOUT|FullyQuantized internal sweep:' \
        "$build_log" "$build_dir/cmake.log" "$build_dir/CMakeCache.txt" \
        "$target_make" 2>/dev/null >&2 || true
      return 2
    fi
    sha256sum "$generated/manifest.json" "$unit" "$registry" \
      "$target_make" "$build_dir/CMakeCache.txt" "$binary" \
      > "$out/results/build-identity-${arm}.sha256" || return 2
    printf '[fq-kpack4-xplane-ab] identity arm=%s provider=AP%s layout=%s generated-unit=1 target-abi=PASS\n' \
      "$arm" "$ap_id" "$layout"
    list_elf="$out/codegen/$arm/list-elf.txt"
    "$hgobjdump" -lelf "$binary" > "$list_elf" \
      2> "$out/codegen/$arm/list-elf.err" || { fail "$arm hgobjdump -lelf"; return 2; }
    symbol="$out/codegen/$arm/kernel-symbol.txt"
    demangled="$out/codegen/$arm/kernel-symbol-demangled.txt"
    python3 -B "$analyzer" select-symbol --list-elf "$list_elf" \
      --symbol-output "$symbol" --demangled-output "$demangled" || return 2
    line="$out/codegen/$arm/kernel-line.txt"
    resource="$out/codegen/$arm/resource-usage.txt"
    "$hgobjdump" -line "-func=$(cat "$symbol")" "$binary" > "$line" \
      2> "$out/codegen/$arm/kernel-line.err" || { fail "$arm line disassembly"; return 2; }
    "$hgobjdump" "-res-usage=$(cat "$symbol")" "$binary" > "$resource" \
      2> "$out/codegen/$arm/resource-usage.err" || { fail "$arm resource report"; return 2; }
    python3 -B "$analyzer" codegen --arm-manifest "$generated/manifest.json" \
      --line "$line" --resource "$resource" --binary "$binary" \
      --symbol "$symbol" --demangled "$demangled" \
      --output "$out/codegen/${arm}.json" || return 2
  done

  while IFS=$'\t' read -r shape_key m n k ap; do
    mkdir -p "$out/runs/$shape_key/ap$ap" || return 2
    for round in $(seq 1 "$rounds"); do
      if [ $((round % 2)) -eq 1 ]; then
        order="xplane-ap$ap kpack4-ap$ap"
      else
        order="kpack4-ap$ap xplane-ap$ap"
      fi
      for name in $order; do
        binary="$(cat "$out/results/binary-${name}.path")"
        symbol_file="$out/results/row-symbol-${name}.txt"
        log="$out/runs/$shape_key/ap$ap/round-${round}-${name}.log"
        printf '[fq-kpack4-xplane-ab] timing shape=%sx%sx%s provider=AP%s round=%s arm=%s\n' \
          "$m" "$n" "$k" "$ap" "$round" "$name"
        "$binary" --shape="${m}x${n}x${k}" --iterations="$iterations" \
          --correctness-repeats="$repeats" --only-split=4 --tm8-max-m=8 \
          --symbols-file="$symbol_file" --bc-mode=skip > "$log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then
          tail -n 100 "$log" >&2
          fail "timing arm=$name shape=$shape_key rc=$rc"; return "$rc"
        fi
      done
    done
  done < <(python3 -B - "$plan" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))['cases']:
 for ap in row['providers']:
  print(row['shape_key'],*row['shape'],ap,sep='\t')
PY
  )

  python3 -B "$analyzer" analyze --master "$selected/manifest.json" \
    --plan "$plan" --runs-root "$out/runs" --codegen-root "$out/codegen" \
    --iterations "$iterations" --rounds "$rounds" \
    --gap-threshold "$gap_threshold" --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" | tee "$out/results/summary.log" || return 2

  acu="$(resolve_executable "${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}" || true)"
  [ -n "$acu" ] || acu="$(resolve_executable "$(command -v acu 2>/dev/null || true)" || true)"
  python3 -B - "$out/results/summary.json" > "$out/results/acu-targets.tsv" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
by_provider={}
for row in value['acu_targets']:
 p=row['a_provider']
 if p not in by_provider or abs(row['delta']) > abs(by_provider[p]['delta']):
  by_provider[p]=row
for p,row in sorted(by_provider.items()):
 ap=int(p.removeprefix('AP'))
 print(row['shape_key'],*row['shape'],ap,*row['arms'],sep='\t')
PY
  if [ "$run_acu" = 1 ] || { [ "$run_acu" = auto ] && [ -s "$out/results/acu-targets.tsv" ]; }; then
    if [ -z "$acu" ]; then
      fail 'timing requires ACU follow-up but acu is unavailable; set ACU'; return 2
    fi
    sha256sum "$acu" > "$out/results/acu-tool.sha256" || return 2
    printf 'shape\tprovider\tarm\treport\tdetails\n' > "$out/results/acu-index.tsv" || return 2
    while IFS=$'\t' read -r shape_key m n k ap name_x name_k; do
      for name in "$name_x" "$name_k"; do
        binary="$(cat "$out/results/binary-${name}.path")"
        symbol_file="$out/results/row-symbol-${name}.txt"
        report_base="$out/acu/${shape_key}-ap${ap}-${name}.report"
        log="$out/acu/${shape_key}-ap${ap}-${name}.log"
        printf '[fq-kpack4-xplane-ab] ACU shape=%s provider=AP%s arm=%s\n' \
          "$shape_key" "$ap" "$name"
        acu_cmd=("$acu" -f -o "$report_base" --set full "$binary" \
          --shape="${m}x${n}x${k}" --iterations=1 --correctness-repeats=1 \
          --only-split=4 --tm8-max-m=8 --symbols-file="$symbol_file" \
          --bc-mode=skip --profile-subject-only)
        printf '%q ' "${acu_cmd[@]}" > "$out/acu/${shape_key}-ap${ap}-${name}.command" || return 2
        printf '\n' >> "$out/acu/${shape_key}-ap${ap}-${name}.command" || return 2
        "${acu_cmd[@]}" > "$log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ] ||
           [ "$(grep -c '^FQ_PROFILE_SUBJECT .* launches=1 reducer_launches=0$' "$log" || true)" -ne 1 ]; then
          tail -n 100 "$log" >&2
          fail "ACU subject closure failed arm=$name shape=$shape_key rc=$rc"; return 2
        fi
        reports=()
        [ -s "$report_base" ] && reports+=("$report_base")
        [ -s "${report_base}.acurep" ] && reports+=("${report_base}.acurep")
        if [ "${#reports[@]}" -ne 1 ]; then
          fail "ACU produced ${#reports[@]} report files for $name/$shape_key"; return 2
        fi
        report="${reports[0]}"
        details="$out/acu/${shape_key}-ap${ap}-${name}.details.csv"
        "$acu" --import "$report" --csv --page details > "$details" || return 2
        [ -s "$details" ] || { fail "empty ACU details for $name/$shape_key"; return 2; }
        printf '%s\tAP%s\t%s\t%s\t%s\n' "$shape_key" "$ap" "$name" \
          "$report" "$details" >> "$out/results/acu-index.tsv" || return 2
      done
    done < "$out/results/acu-targets.tsv"
  fi

  find "$out" -type f ! -name bundle.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > "$out/bundle.sha256" || return 2
  sed -n '1,12p' "$out/results/summary.tsv"
  printf '[fq-kpack4-xplane-ab] PASS sha=%s config=8x64x256_w8x16_s2 S=4 comparisons=8 artifacts=%s\n' \
    "$sha" "$out"
  if [ -s "$out/results/acu-targets.tsv" ]; then
    printf '[fq-kpack4-xplane-ab] ACU targets/reports=%s/acu\n' "$out"
  fi
}

main "$@"
