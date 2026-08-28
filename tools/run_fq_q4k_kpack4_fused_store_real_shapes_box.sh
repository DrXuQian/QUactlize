#!/usr/bin/env bash
# Matched D32 plain/fused-store K-pack4 sweep over all inventory-owned decode shapes.
set -uo pipefail

fail() {
  printf '[fq-kpack4-fused-store-real] FAIL: %s\n' "$*" >&2
  return 2
}

atomic_text() {
  local destination="$1" value="$2" current
  current="${destination}.current.$$"
  printf '%s\n' "$value" > "$current" || return 2
  mv -f -- "$current" "$destination" || return 2
}

run_phase() {
  local log="$1" commit="${1}.sha256" actual failed current rc
  shift
  if [ -s "$log" ] && [ -s "$commit" ]; then
    actual="$(sha256sum "$log" | awk '{print $1}')" || return 2
    [ "$(cat "$commit")" = "$actual" ] || {
      fail "committed phase changed: $log"; return 2; }
    [ "${RESUME:-0}" = 1 ] || {
      fail "phase exists without RESUME=1: $log"; return 2; }
    printf '[fq-kpack4-fused-store-real] resume phase=%s\n' "$log"
    return 0
  fi
  if [ -e "$log" ] || [ -e "$commit" ]; then
    [ "${RESUME:-0}" = 1 ] || {
      fail "phase residue exists without RESUME=1: $log"; return 2; }
    failed="${log}.uncommitted.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    [ ! -e "$log" ] || mv -- "$log" "$failed" || return 2
    [ ! -e "$commit" ] || mv -- "$commit" "${failed}.sha256" || return 2
    printf '[fq-kpack4-fused-store-real] preserved residue=%s\n' "$failed"
  fi
  current="${log}.current.$$"
  "$@" > "$current" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    failed="${log}.failed.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mv -- "$current" "$failed" || return 2
    tail -n 120 "$failed" >&2
    fail "phase rc=$rc preserved=$failed"
    return "$rc"
  fi
  mv -- "$current" "$log" || return 2
  actual="$(sha256sum "$log" | awk '{print $1}')" || return 2
  atomic_text "$commit" "$actual"
}

build_arm() {
  local root="$1" out="$2" generated="$3" jobs="$4" units="$5" arm="$6"
  local fused="$7" build_dir build_log target_make binary defs rc
  build_dir="$out/build/$arm"
  build_log="$out/results/build-$arm.log"
  if [ -s "$out/results/binary-$arm.path" ] && \
     [ -s "$out/results/build-$arm.sha256" ]; then
    [ "${RESUME:-0}" = 1 ] || {
      fail "build exists without RESUME=1: $arm"; return 2; }
    sha256sum -c "$out/results/build-$arm.sha256" >/dev/null || {
      fail "build authority changed: $arm"; return 2; }
    binary="$(cat "$out/results/binary-$arm.path")"
    [ -x "$binary" ] && [ ! -L "$binary" ] || {
      fail "binary is missing or linked: $arm"; return 2; }
    printf '[fq-kpack4-fused-store-real] resume build=%s\n' "$arm"
    return 0
  fi
  [ ! -e "$build_dir" ] || {
    fail "partial build residue requires a new OUT: $build_dir"; return 2; }
  mkdir -p "$build_dir" || return 2
  defs='FQ_TC_KPACK4_DELIVERY_N=32'
  [ "$fused" -eq 0 ] || defs="$defs PPU_PACKED_SCALE_FUSED=1"
  printf '[fq-kpack4-fused-store-real] build arm=%s D32 fused=%s typed=144\n' \
    "$arm" "$fused"
  (cd "$root" && env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE -u CC -u CXX \
    PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    TARGET=test_fully_quantized_internal_sweep \
    FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
    FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0 \
    FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1 \
    PPU_DEFS="$defs" PPU_EXTRA_DEFS= \
    CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= ./build.sh) > "$build_log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    tail -n 180 "$build_log" >&2
    fail "$arm build rc=$rc artifacts=$out"
    return "$rc"
  fi
  target_make="$(find "$build_dir" -type f \
    -path '*test_fully_quantized_internal_sweep.dir/build.make' -print -quit)"
  binary="$(find "$build_dir" -type f -name test_fully_quantized_internal_sweep \
    -perm -u+x -print -quit)"
  [ -n "$target_make" ] && [ -n "$binary" ] && [ ! -L "$binary" ] || {
    fail "$arm build identity is incomplete"; return 2; }
  grep -Fqx '[build.sh] FQ_SWEEP_WEIGHT_LAYOUT=1' "$build_log" && \
    grep -F "FullyQuantized internal sweep: q=12 A=0 bc=0 format=0 layout=1 units=$units" \
      "$build_dir/cmake.log" >/dev/null && \
    grep -Eq '^FQ_SWEEP_WEIGHT_LAYOUT(:[^=]*)?=1$' "$build_dir/CMakeCache.txt" && \
    grep -Eq -- '(^|[[:space:]])-DFQ_SWEEP_WEIGHT_LAYOUT=1([[:space:]]|$)' \
      "$target_make" || {
    fail "$arm generated/layout build ABI differs"; return 2; }
  [ "$(grep -Eo -- '-DFQ_TC_KPACK4_DELIVERY_N=[0-9]+' "$target_make" | \
      sort -u | tr '\n' ' ')" = '-DFQ_TC_KPACK4_DELIVERY_N=32 ' ] || {
    fail "$arm D32 compile ABI differs"; return 2; }
  if [ "$fused" -eq 1 ]; then
    [ "$(grep -Eo -- '-DPPU_PACKED_SCALE_FUSED=[0-9]+' "$target_make" | \
        sort -u | tr '\n' ' ')" = '-DPPU_PACKED_SCALE_FUSED=1 ' ] || {
      fail "$arm fused-store compile ABI differs"; return 2; }
  elif grep -Eq -- '-DPPU_PACKED_SCALE_FUSED(=|[[:space:]])' "$target_make"; then
    fail "$arm plain binary carries fused-store define"; return 2
  fi
  if grep -F 'PPU_PACKED_SCALE_FUSED_READ' "$target_make" >/dev/null; then
    fail "$arm retained deleted fused-load define"; return 2
  fi
  printf '%s\n' "$binary" > "$out/results/binary-$arm.path" || return 2
  sha256sum "$binary" "$target_make" "$generated/manifest.json" \
    > "$out/results/build-$arm.sha256" || return 2
}

main() {
  local root workspace sha short stamp out inventory policy jobs per_unit threshold
  local plan generated manifest units source_state saved_state ordinal shape_key m n k
  local arm fused binary directory shared screen_log scheduler_log confirm_log order
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-fused-store-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *) fail 'OUT must be a strict /workspace child'; return 2;; esac
  inventory="${INTERNAL_SWEEP_SPEC:-}"
  [ -n "$inventory" ] && [ -f "$inventory" ] || {
    fail 'INTERNAL_SWEEP_SPEC must name COMPLETE inventory-v2 JSON'; return 2; }
  inventory="$(realpath -e -- "$inventory")" || return 2
  policy="$(realpath -e -- "${FQ_Q4K_DECODE_POLICY:-$root/benchmarks/fq_q4k_decode_real_shapes_policy.json}")" || return 2
  cmp -s "$policy" "$root/benchmarks/fq_q4k_decode_real_shapes_policy.json" || {
    fail 'policy must be the registered decode policy'; return 2; }
  jobs="${JOBS:-16}"
  per_unit="${FQ_CONFIGS_PER_UNIT:-4}"
  threshold="${MATERIAL_THRESHOLD:-0.02}"
  case "$jobs:$per_unit" in *[!0-9:]*|0:*|*:0)
    fail 'JOBS/FQ_CONFIGS_PER_UNIT must be positive'; return 2;; esac
  python3 -B - "$threshold" <<'PY' || return 2
import sys
assert 0 < float(sys.argv[1]) < 1
PY
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS changes the factorial'; return 2
  fi
  if [ -e "$out" ] && [ "${RESUME:-0}" != 1 ]; then
    fail "refusing existing OUT without RESUME=1: $out"; return 2
  fi
  mkdir -p "$out/generated" "$out/build" "$out/raw/plain" \
    "$out/raw/store" "$out/raw/shared" "$out/results" || return 2

  python3 -B "$root/tools/plan_fq_q4k_decode_real_shapes.py" self-test || return 2
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" self-test \
    --policy "$policy" || return 2
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_fused_store_real_shapes.py" \
    self-test --policy "$policy" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_generator.py" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_fused_store_real_shapes_runner.py" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_delivery_committed_evidence.py" \
    --committed-only || return 2

  plan="$out/plan.json"
  if [ ! -s "$plan" ]; then
    python3 -B "$root/tools/plan_fq_q4k_decode_real_shapes.py" materialize \
      --inventory "$inventory" --policy "$policy" --output "$plan" || return 2
  fi
  python3 -B - "$plan" <<'PY' || return 2
import json,sys
value=json.load(open(sys.argv[1]))
assert value["family_count"] == 5 and value["shape_count"] == 20
assert value["decode_m"] == [1,2,4,8]
PY
  generated="$out/generated"
  manifest="$generated/manifest.json"
  if [ ! -s "$manifest" ]; then
    python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
      --qtype 12 --artifact-tk 0 --bchunk 0 --weight-layout q4-kpack4 \
      --tile-m-filter 8 --per-unit "$per_unit" --out-dir "$generated" || return 2
  fi
  python3 -B - "$manifest" <<'PY' || return 2
import json,sys
value=json.load(open(sys.argv[1]))
assert value["denominator"]["typed_rows"] == 144
assert value["denominator"]["source_typed_rows"] == 918
assert {row["a_provider"] for row in value["typed_rows"]} == {"standard-aiu","packed-row"}
assert value["weight_mapping"]["mapping_id"] == "0x51344b5034540001"
PY
  units="$(python3 -B -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["units"]))' "$manifest")" || return 2
  build_arm "$root" "$out" "$generated" "$jobs" "$units" plain 0 || return $?
  build_arm "$root" "$out" "$generated" "$jobs" "$units" store 1 || return $?

  source_state="$({
    git -C "$root" rev-parse HEAD
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum "$inventory" "$policy" "$plan" "$manifest" \
      "$(cat "$out/results/binary-plain.path")" \
      "$(cat "$out/results/binary-store.path")" \
      "$root/tools/analyze_fq_q4k_kpack4_pilot.py" \
      "$root/tools/analyze_fq_q4k_kpack4_fused_store_real_shapes.py" \
      "$root/tools/run_fq_q4k_kpack4_fused_store_real_shapes_box.sh" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
  } | sha256sum | awk '{print $1}')" || return 2
  saved_state="$out/source-state.sha256"
  if [ -s "$saved_state" ] && [ "$(cat "$saved_state")" != "$source_state" ]; then
    fail 'source/inventory/binary authority changed on resume'; return 2
  fi
  atomic_text "$saved_state" "$source_state" || return 2
  git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2

  ordinal=0
  while IFS=$'\t' read -r shape_key m n k; do
    ordinal=$((ordinal + 1))
    shared="$out/raw/shared/$shape_key"
    mkdir -p "$shared" "$out/raw/plain/$shape_key" "$out/raw/store/$shape_key" || return 2
    if [ $((ordinal % 2)) -eq 1 ]; then order='plain store'; else order='store plain'; fi
    for arm in $order; do
      [ "$arm" = store ] && fused=1 || fused=0
      binary="$(cat "$out/results/binary-$arm.path")"
      directory="$out/raw/$arm/$shape_key"
      screen_log="$directory/screen.log"
      printf '[fq-kpack4-fused-store-real] shape=%sx%sx%s arm=%s phase=screen typed=144\n' \
        "$m" "$n" "$k" "$arm"
      run_phase "$screen_log" "$binary" --shape="${m}x${n}x${k}" \
        --iterations=2 --correctness-repeats=1 --only-split=1 \
        --tm8-max-m=8 --bc-mode=all || return $?
      python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" screen \
        --shape="${m}x${n}x${k}" --manifest "$manifest" --log "$screen_log" \
        --policy "$policy" --scalezero-fused "$fused" --delivery-n 32 \
        --symbols-output "$directory/screen-symbols.txt" \
        --summary-output "$directory/screen.json" || return 2
    done
    python3 -B "$root/tools/analyze_fq_q4k_kpack4_fused_store_real_shapes.py" \
      union-symbols --manifest "$manifest" \
      --input "$out/raw/plain/$shape_key/screen-symbols.txt" \
      --input "$out/raw/store/$shape_key/screen-symbols.txt" \
      --output "$shared/screen-union-symbols.txt" || return 2

    if [ $((ordinal % 2)) -eq 1 ]; then order='store plain'; else order='plain store'; fi
    for arm in $order; do
      [ "$arm" = store ] && fused=1 || fused=0
      binary="$(cat "$out/results/binary-$arm.path")"
      directory="$out/raw/$arm/$shape_key"
      scheduler_log="$directory/scheduler.log"
      printf '[fq-kpack4-fused-store-real] shape=%sx%sx%s arm=%s phase=scheduler\n' \
        "$m" "$n" "$k" "$arm"
      run_phase "$scheduler_log" "$binary" --shape="${m}x${n}x${k}" \
        --iterations=1 --correctness-repeats=1 \
        --symbols-file="$shared/screen-union-symbols.txt" \
        --tm8-max-m=8 --bc-mode=skip || return $?
      python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" scheduler \
        --shape="${m}x${n}x${k}" --manifest "$manifest" --log "$scheduler_log" \
        --screen-symbols "$shared/screen-union-symbols.txt" --policy "$policy" \
        --scalezero-fused "$fused" --delivery-n 32 \
        --symbols-output "$directory/confirm-symbols.txt" \
        --summary-output "$directory/scheduler.json" || return 2
    done
    python3 -B "$root/tools/analyze_fq_q4k_kpack4_fused_store_real_shapes.py" \
      union-symbols --manifest "$manifest" \
      --input "$out/raw/plain/$shape_key/confirm-symbols.txt" \
      --input "$out/raw/store/$shape_key/confirm-symbols.txt" \
      --output "$shared/confirm-union-symbols.txt" || return 2

    if [ $((ordinal % 2)) -eq 1 ]; then order='plain store'; else order='store plain'; fi
    for arm in $order; do
      [ "$arm" = store ] && fused=1 || fused=0
      binary="$(cat "$out/results/binary-$arm.path")"
      directory="$out/raw/$arm/$shape_key"
      confirm_log="$directory/confirm.log"
      printf '[fq-kpack4-fused-store-real] shape=%sx%sx%s arm=%s phase=confirm\n' \
        "$m" "$n" "$k" "$arm"
      run_phase "$confirm_log" "$binary" --shape="${m}x${n}x${k}" \
        --iterations=7 --correctness-repeats=2 \
        --symbols-file="$shared/confirm-union-symbols.txt" \
        --tm8-max-m=8 --bc-mode=skip || return $?
      python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" finalize \
        --shape="${m}x${n}x${k}" --manifest "$manifest" --log "$confirm_log" \
        --symbols "$shared/confirm-union-symbols.txt" --policy "$policy" \
        --scalezero-fused "$fused" --delivery-n 32 \
        --output-json "$directory/summary.json" \
        --output-tsv "$directory/summary.tsv" || return 2
    done
  done < <(python3 -B - "$plan" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
for row in sorted(value["shapes"],key=lambda item:item["shape_key"]):
 print(row["shape_key"],row["m"],row["n"],row["k"],sep="\t")
PY
  )

  for arm in plain store; do
    [ "$arm" = store ] && fused=1 || fused=0
    python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" aggregate \
      --plan "$plan" --policy "$policy" --raw-root "$out/raw/$arm" \
      --scalezero-fused "$fused" --delivery-n 32 \
      --output-json "$out/results/$arm.json" \
      --output-tsv "$out/results/$arm.tsv" || return 2
  done
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_fused_store_real_shapes.py" \
    compare --plan "$plan" --policy "$policy" --manifest "$manifest" \
    --raw-root "$out/raw" --plain "$out/results/plain.json" \
    --store "$out/results/store.json" --threshold "$threshold" \
    --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" | tee "$out/results/summary.log" || return 2
  find "$out/raw" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$out/results/raw-authority.sha256" || return 2
  sha256sum "$plan" "$manifest" "$(cat "$out/results/binary-plain.path")" \
    "$(cat "$out/results/binary-store.path")" "$out/results/summary.json" \
    > "$out/results/authority.sha256" || return 2
  sed -n '1,21p' "$out/results/summary.tsv"
  printf '[fq-kpack4-fused-store-real] PASS sha=%s families=5 shapes=20 variants=plain+store D32 artifacts=%s\n' \
    "$sha" "$out"
  printf '[fq-kpack4-fused-store-real] summary=%s\n' "$out/results/summary.tsv"
}

main "$@"
