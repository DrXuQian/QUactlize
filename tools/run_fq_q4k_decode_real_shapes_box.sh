#!/usr/bin/env bash
# Real-GGUF Q4_K decode sweep for M=1/2/4/8/16.
#
# Phase 1 screens every compiled TC S1 tactic and all four native-batch SIMT
# RPW candidates.  Phase 2 expands only the retained TC symbols across
# S=1/2/4/8.  Phase 3 confirms the board shortlist and SIMT candidates with
# seven samples.  Split-K product ranking adds the registered 80%-HBM reducer
# model with zero launch time; producer-only numbers are never product latency.
set -uo pipefail

atomic_text() {
  local destination="$1" value="$2" current
  current="${destination}.current.$$"
  printf '%s\n' "$value" > "$current" || return 2
  mv -f -- "$current" "$destination" || return 2
}

run_phase() {
  local log="$1" commit="${1}.sha256" actual failed
  shift
  if [ -s "$log" ] && [ -s "$commit" ]; then
    actual="$(sha256sum "$log" | awk '{print $1}')" || return 2
    if [ "$(cat "$commit")" != "$actual" ]; then
      printf '[fq-q4k-decode] FAIL: committed phase log changed: %s\n' "$log" >&2
      return 2
    fi
    if [ "${RESUME:-0}" != 1 ]; then
      printf '[fq-q4k-decode] FAIL: phase log already exists without RESUME=1: %s\n' "$log" >&2
      return 2
    fi
    printf '[fq-q4k-decode] resume phase=%s\n' "$log"
    return 0
  fi
  if [ -e "$log" ] || [ -e "$commit" ]; then
    if [ "${RESUME:-0}" != 1 ]; then
      printf '[fq-q4k-decode] FAIL: uncommitted phase residue without RESUME=1: %s\n' "$log" >&2
      return 2
    fi
    failed="${log}.uncommitted.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    if [ -e "$log" ]; then mv -- "$log" "$failed" || return 2; fi
    if [ -e "$commit" ]; then mv -- "$commit" "${failed}.sha256" || return 2; fi
    printf '[fq-q4k-decode] preserved uncommitted phase residue=%s\n' "$failed"
  fi
  local current="${log}.current.$$" rc
  "$@" > "$current" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    failed="${log}.failed.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mv -- "$current" "$failed" || return 2
    printf '[fq-q4k-decode] FAIL: phase rc=%d preserved=%s\n' "$rc" "$failed" >&2
    tail -80 "$failed" >&2
    return "$rc"
  fi
  mv -- "$current" "$log" || return 2
  actual="$(sha256sum "$log" | awk '{print $1}')" || return 2
  atomic_text "$commit" "$actual" || return 2
}

main() {
  local root workspace_root sha short stamp out inventory policy jobs per_unit prefill_bundle
  local source_state saved_source plan plan_sha artifact generated manifest typed
  local build_dir build_log binary shape_key m n k class refs directory
  local screen_log scheduler_log confirm_tc_log confirm_bc_log
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-decode-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[fq-q4k-decode] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  inventory="${INTERNAL_SWEEP_SPEC:-}"
  if [ -z "$inventory" ] || [ ! -f "$inventory" ]; then
    printf '[fq-q4k-decode] FAIL: INTERNAL_SWEEP_SPEC must name the COMPLETE inventory-v2 JSON\n' >&2
    return 2
  fi
  inventory="$(realpath -e -- "$inventory")" || return 2
  policy="${FQ_Q4K_DECODE_POLICY:-$root/benchmarks/fq_q4k_decode_real_shapes_policy.json}"
  policy="$(realpath -e -- "$policy")" || return 2
  prefill_bundle="${PREFILL_BUNDLE:-}"
  if [ -n "$prefill_bundle" ]; then
    prefill_bundle="$(realpath -e -- "$prefill_bundle")" || return 2
  fi
  jobs="${JOBS:-16}"
  per_unit="${FQ_CONFIGS_PER_UNIT:-4}"
  case "$jobs:$per_unit" in
    *[!0-9:]*|0:*|*:0) printf '[fq-q4k-decode] FAIL: JOBS/FQ_CONFIGS_PER_UNIT must be positive integers\n' >&2; return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[fq-q4k-decode] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes the denominator\n' >&2
    return 2
  fi
  if [ -e "$out" ] && [ "${RESUME:-0}" != 1 ]; then
    printf '[fq-q4k-decode] FAIL: refusing to overwrite %s; set RESUME=1 to continue it\n' "$out" >&2
    return 2
  fi
  mkdir -p "$out/generated" "$out/build" "$out/raw" "$out/results" || return 2

  python3 -B "$root/tools/plan_fq_q4k_decode_real_shapes.py" self-test || return 2
  python3 -B "$root/tools/analyze_fq_q4k_decode_real_shapes.py" self-test || return 2
  python3 -B "$root/ci/check_fq_q4k_decode_batch_and_runner.py" || return 2

  source_state="$({
    git -C "$root" rev-parse HEAD
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum \
      "$inventory" "$policy" \
      "$root/benchmarks/test_fully_quantized_internal_sweep.cu" \
      "$root/benchmarks/fully_quantized_splitk_producer_bench.hpp" \
      "$root/benchmarks/fully_quantized_splitk_producer_unit.inc" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/ppu_packed_metadata_ownership.hpp" \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp" \
      "$root/quactlize/include/dense_splitk_multiformat_ppu.cuh" \
      "$root/quactlize/include/dense_splitk_parallel_ppu.cuh" \
      "$root/quactlize/include/gguf_bc_vecdot.hpp" \
      "$root/quactlize/include/gguf_bc_q4_reader.hpp" \
      "$root/quactlize/include/gguf_packed_unit.hpp" \
      "$root/quactlize/include/ppu_dense_shipping_policy.hpp" \
      "$root/quactlize/include/ppu_format_config.inc" \
      "$root/quactlize/include/ppu_group_schedule.hpp" \
      "$root/quactlize/include/ppu_tactic_space.hpp" \
      "$root/quactlize/csrc/fq_internal_sweep.cmake.in" \
      "$root/tools/plan_fq_q4k_decode_real_shapes.py" \
      "$root/tools/analyze_fq_q4k_decode_real_shapes.py" \
      "$root/tools/archive_scalefirst_q4k_prefill.py" \
      "$root/tools/emit_fully_quantized_splitk_superset.cpp" \
      "$root/tools/fully_quantized_internal_matrix.py" \
      "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
      "$root/tools/run_fq_q4k_decode_real_shapes_box.sh" \
      "$root/ci/check_fq_q4k_decode_batch_and_runner.py" \
      "$root/build.sh"
  } | sha256sum | awk '{print $1}')" || return 2
  saved_source="$out/source-state.sha256"
  if [ -s "$saved_source" ] && [ "$(cat "$saved_source")" != "$source_state" ]; then
    printf '[fq-q4k-decode] FAIL: source/inventory/policy authority changed on resume\n' >&2
    return 2
  fi
  if [ ! -s "$saved_source" ]; then
    atomic_text "$saved_source" "$source_state" || return 2
    git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2
  fi

  plan="$out/plan.json"
  if [ ! -s "$plan" ]; then
    python3 -B "$root/tools/plan_fq_q4k_decode_real_shapes.py" materialize \
      --inventory "$inventory" --policy "$policy" --output "$plan" || return 2
  fi
  python3 -B "$root/tools/plan_fq_q4k_decode_real_shapes.py" list --plan "$plan" > "$out/plan.tsv" || return 2
  plan_sha="$(sha256sum "$plan" | awk '{print $1}')" || return 2
  if [ -s "$out/plan.sha256" ] && [ "$(cat "$out/plan.sha256")" != "$plan_sha" ]; then
    printf '[fq-q4k-decode] FAIL: plan changed on resume\n' >&2
    return 2
  fi
  atomic_text "$out/plan.sha256" "$plan_sha" || return 2

  if [ -n "$prefill_bundle" ]; then
    if [ -s "$out/prefill-archive/archive.json" ]; then
      python3 -B "$root/tools/archive_scalefirst_q4k_prefill.py" \
        --verify "$out/prefill-archive" || return 2
    else
      python3 -B "$root/tools/archive_scalefirst_q4k_prefill.py" \
        --source "$prefill_bundle" --output "$out/prefill-archive" || return 2
    fi
  else
    printf '[fq-q4k-decode] NOTE: PREFILL_BUNDLE unset; decode is complete, prefill archive not requested in this invocation\n'
  fi

  for artifact in 32 64 128 256; do
    generated="$out/generated/a${artifact}"
    manifest="$generated/manifest.json"
    if [ ! -s "$manifest" ]; then
      mkdir -p "$generated" || return 2
      python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
        --qtype 12 --artifact-tk "$artifact" --bchunk 0 \
        --per-unit "$per_unit" --out-dir "$generated" || return 2
    fi
    local generated_sha generated_authority="$out/generated/a${artifact}.sha256"
    generated_sha="$({
      find "$generated" -type f -print0 | sort -z | xargs -0 sha256sum
    } | sha256sum | awk '{print $1}')" || return 2
    if [ -s "$generated_authority" ] && [ "$(cat "$generated_authority")" != "$generated_sha" ]; then
      printf '[fq-q4k-decode] FAIL: generated A=%s authority changed on resume\n' "$artifact" >&2
      return 2
    fi
    atomic_text "$generated_authority" "$generated_sha" || return 2
    typed="$(python3 -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["denominator"]["typed_rows"])' "$manifest")" || return 2
    build_dir="$out/build/a${artifact}"
    binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
    build_log="$out/build/a${artifact}.log"
    if [ ! -x "$binary" ]; then
      printf '[fq-q4k-decode] build A=%s typed=%s (A32 is intentionally BC-only)\n' "$artifact" "$typed"
      (cd "$root" && PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
        TARGET=test_fully_quantized_internal_sweep \
        FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
        FQ_SWEEP_ARTIFACT_TK="$artifact" FQ_SWEEP_BCHUNK=0 \
        FQ_SWEEP_PACKED_FORMAT=0 ./build.sh) > "$build_log" 2>&1
      local rc=$?
      if [ "$rc" -ne 0 ]; then
        printf '[fq-q4k-decode] FAIL: build A=%s rc=%d\n' "$artifact" "$rc" >&2
        tail -120 "$build_log" >&2
        return "$rc"
      fi
      binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
    fi
    if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
      printf '[fq-q4k-decode] FAIL: exact binary missing A=%s path=%s\n' "$artifact" "$binary" >&2
      return 2
    fi
    local binary_sha binary_authority="$out/build/a${artifact}.sha256"
    binary_sha="$(sha256sum "$binary" | awk '{print $1}')" || return 2
    if [ -s "$binary_authority" ] && [ "$(cat "$binary_authority")" != "$binary_sha" ]; then
      printf '[fq-q4k-decode] FAIL: binary A=%s changed on resume\n' "$artifact" >&2
      return 2
    fi
    atomic_text "$binary_authority" "$binary_sha" || return 2

    while IFS=$'\t' read -r _ shape_key m n k class refs; do
      directory="$out/raw/a${artifact}/$shape_key"
      mkdir -p "$directory" || return 2
      screen_log="$directory/screen.log"
      scheduler_log="$directory/scheduler.log"
      confirm_tc_log="$directory/confirm-tc.log"
      confirm_bc_log="$directory/confirm-bc.log"
      printf '[fq-q4k-decode] A=%s shape=%sx%sx%s phase=screen typed=%s\n' \
        "$artifact" "$m" "$n" "$k" "$typed"
      run_phase "$screen_log" "$binary" --shape="${m}x${n}x${k}" \
        --iterations=2 --correctness-repeats=1 --only-split=1 --bc-mode=all || return $?
      python3 -B "$root/tools/analyze_fq_q4k_decode_real_shapes.py" screen \
        --manifest "$manifest" --log "$screen_log" --policy "$policy" \
        --symbols-output "$directory/screen-symbols.txt" \
        --summary-output "$directory/screen.json" || return 2

      if [ "$typed" -gt 0 ]; then
        printf '[fq-q4k-decode] A=%s shape=%sx%sx%s phase=scheduler\n' "$artifact" "$m" "$n" "$k"
        run_phase "$scheduler_log" "$binary" --shape="${m}x${n}x${k}" \
          --iterations=1 --correctness-repeats=1 \
          --symbols-file="$directory/screen-symbols.txt" --bc-mode=skip || return $?
        python3 -B "$root/tools/analyze_fq_q4k_decode_real_shapes.py" scheduler \
          --manifest "$manifest" --log "$scheduler_log" \
          --screen-symbols "$directory/screen-symbols.txt" --policy "$policy" \
          --symbols-output "$directory/confirm-symbols.txt" \
          --summary-output "$directory/scheduler.json" || return 2
        printf '[fq-q4k-decode] A=%s shape=%sx%sx%s phase=confirm-tc\n' "$artifact" "$m" "$n" "$k"
        run_phase "$confirm_tc_log" "$binary" --shape="${m}x${n}x${k}" \
          --iterations=7 --correctness-repeats=2 \
          --symbols-file="$directory/confirm-symbols.txt" --bc-mode=skip || return $?
      fi
      printf '[fq-q4k-decode] A=%s shape=%sx%sx%s phase=confirm-simt\n' "$artifact" "$m" "$n" "$k"
      run_phase "$confirm_bc_log" "$binary" --shape="${m}x${n}x${k}" \
        --iterations=7 --correctness-repeats=2 --bc-mode=only || return $?
    done < <(python3 -B "$root/tools/plan_fq_q4k_decode_real_shapes.py" list --plan "$plan" |
             while IFS=$'\t' read -r listed_artifact rest; do
               if [ "$listed_artifact" = "$artifact" ]; then printf '%s\t%s\n' "$listed_artifact" "$rest"; fi
             done)
  done

  python3 -B "$root/tools/analyze_fq_q4k_decode_real_shapes.py" finalize \
    --plan "$plan" --policy "$policy" --raw-root "$out/raw" \
    --generated-root "$out/generated" --output-dir "$out/results" || return 2
  {
    printf 'schema=quactlize.fq_q4k_decode_real_shapes.run.v1\n'
    printf 'root_sha=%s\nactlize_sha=%s\n' "$sha" "$(git -C "$root/third_party/actlize" rev-parse HEAD)"
    printf 'inventory=%s\ninventory_sha256=%s\n' "$inventory" "$(sha256sum "$inventory" | awk '{print $1}')"
    printf 'policy_sha256=%s\nplan_sha256=%s\nsource_state_sha256=%s\n' \
      "$(sha256sum "$policy" | awk '{print $1}')" "$plan_sha" "$source_state"
    printf 'simt=M<8/native-grid-y/one-launch\n'
    printf 'splitk_reducer=80%%-of-2766GBps/zero-launch\n'
    printf 'summary_sha256=%s\n' "$(sha256sum "$out/results/summary.json" | awk '{print $1}')"
  } > "$out/provenance.txt" || return 2
  printf '[fq-q4k-decode] PASS sha=%s artifacts=%s\n' "$sha" "$out"
  printf '[fq-q4k-decode] summary=%s\n' "$out/results/summary.tsv"
}

main "$@"
