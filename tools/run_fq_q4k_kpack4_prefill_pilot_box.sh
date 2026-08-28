#!/usr/bin/env bash
# First raw-bit and performance pilot for canonical Q4_K K-pack4 prefill.
# The complete TM64/AP0 TK64/TK128/TK256 graph is compiled once, then pruned at runtime:
#   S1 screen -> S1/S2/S4/S8 scheduler -> seven-sample confirmation.
set -uo pipefail

fail() {
  printf '[fq-kpack4-prefill] FAIL: %s\n' "$*" >&2
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
    printf '[fq-kpack4-prefill] resume phase=%s\n' "$log"
    return 0
  fi
  if [ -e "$log" ] || [ -e "$commit" ]; then
    [ "${RESUME:-0}" = 1 ] || {
      fail "phase residue exists without RESUME=1: $log"; return 2; }
    failed="${log}.uncommitted.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    [ ! -e "$log" ] || mv -- "$log" "$failed" || return 2
    [ ! -e "$commit" ] || mv -- "$commit" "${failed}.sha256" || return 2
    printf '[fq-kpack4-prefill] preserved residue=%s\n' "$failed"
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
  actual="$(sha256sum "$log" | awk '{print $1}')" || return 2
  atomic_text "$commit" "$actual"
}

main() {
  local root workspace sha short stamp out inventory policy master jobs per_unit
  local plan policies generated manifest units build_dir build_log target_make
  local binary source_state saved_state screen_log scheduler_log confirm_log rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-kpack4-prefill-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace"/*) ;;
    *) fail 'OUT must be a strict /workspace child'; return 2 ;;
  esac
  inventory="${INTERNAL_SWEEP_SPEC:-}"
  [ -n "$inventory" ] && [ -f "$inventory" ] || {
    fail 'INTERNAL_SWEEP_SPEC must name COMPLETE inventory-v2 JSON'; return 2; }
  inventory="$(realpath -e -- "$inventory")" || return 2
  policy="$root/benchmarks/fq_q4k_kpack4_prefill_pilot_policy.json"
  master="$root/benchmarks/scalefirst_q4k_real_shapes_pruned_policy.json"
  jobs="${JOBS:-16}"
  per_unit="${FQ_CONFIGS_PER_UNIT:-4}"
  case "$jobs:$per_unit" in
    *[!0-9:]*|0:*|*:0) fail 'JOBS/FQ_CONFIGS_PER_UNIT must be positive'; return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ] || \
     [ -n "${FQ_TC_KPACK4_DELIVERY_N:-}" ]; then
    fail 'ambient PPU_DEFS/PPU_EXTRA_DEFS/delivery override changes the pilot'; return 2
  fi
  if [ -e "$out" ] && [ "${RESUME:-0}" != 1 ]; then
    fail "refusing existing OUT without RESUME=1: $out"; return 2
  fi
  mkdir -p "$out/generated" "$out/build" "$out/raw" "$out/results" \
    "$out/policies" || return 2

  python3 -B "$root/tools/plan_scalefirst_q4k_real_shapes.py" self-test || return 2
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_prefill_pilot.py" self-test \
    --policy "$policy" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_generator.py" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_prefill_pilot.py" || return 2

  plan="$out/plan.json"
  policies="$out/policies"
  if [ ! -s "$plan" ]; then
    python3 -B "$root/tools/plan_scalefirst_q4k_real_shapes.py" materialize \
      --inventory "$inventory" --master-policy "$master" \
      --output "$plan" --policies-dir "$policies" || return 2
  fi
  python3 -B - "$plan" <<'PY' || return 2
import json, sys
value = json.load(open(sys.argv[1]))
target = [row for row in value["shapes"]
          if [row["m"], row["n"], row["k"]] == [2048, 1024, 5120]]
assert value["shape_count"] == 15 and value["cell_count"] == 60
assert len(target) == 1 and target[0]["references"]
PY

  generated="$out/generated"
  manifest="$generated/manifest.json"
  if [ ! -s "$manifest" ]; then
    python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
      --qtype 12 --artifact-tk 0 --bchunk 0 --weight-layout q4-kpack4 \
      --kpack4-subsuperblocks --tile-m-filter 64 --per-unit "$per_unit" \
      --out-dir "$generated" || return 2
  fi
  python3 -B - "$manifest" <<'PY' || return 2
import json, sys
value = json.load(open(sys.argv[1]))
assert value["identity"] == {
    "qtype": 12, "format": "Q4_K", "artifact_tile_k": 0, "bchunk": 0,
    "tile_m_filter": 64, "weight_layout": "q4-kpack4",
    "kpack4_subsuperblocks": True}
assert value["denominator"] == {
    "raw_topology_rows": 11520, "provider_expanded_rows": 12000,
    "source_typed_rows": 2754, "typed_rows": 630,
    "selection_reject_rows": 2124, "static_reject_rows": 9246,
    "runtime_tc_cells": 48000, "typed_runtime_tc_cells": 2520}
assert {row["tile_m"] for row in value["typed_rows"]} == {64}
assert {row["a_provider"] for row in value["typed_rows"]} == {"standard-aiu"}
assert {row["tactic_tile_k"] for row in value["typed_rows"]} == {64,128,256}
assert {tk: sum(row["tactic_tile_k"] == tk for row in value["typed_rows"])
        for tk in (64,128,256)} == {64:210,128:210,256:210}
assert value["weight_mapping"]["mapping_id"] == "0x51344b5034540001"
assert value["weight_mapping"]["artifact_tile_k_is_not_an_axis"] is True
PY
  units="$(python3 -B -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["units"]))' "$manifest")" || return 2

  build_dir="$out/build"
  build_log="$out/results/build.log"
  if [ ! -s "$out/results/build.sha256" ]; then
    (cd "$root" && env -u CMAKE_GENERATOR -u CMAKE_TOOLCHAIN_FILE -u CC -u CXX \
      PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
      TARGET=test_fully_quantized_internal_sweep \
      FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
      FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0 \
      FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1 \
      PPU_DEFS= PPU_EXTRA_DEFS= CFLAGS= CXXFLAGS= CPPFLAGS= LDFLAGS= \
      ./build.sh) > "$build_log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      tail -n 180 "$build_log" >&2
      fail "TM64=630 target build rc=$rc artifacts=$out"
      return "$rc"
    fi
    target_make="$(find "$build_dir" -type f \
      -path '*test_fully_quantized_internal_sweep.dir/build.make' -print -quit)"
    binary="$(find "$build_dir" -type f -name test_fully_quantized_internal_sweep \
      -perm -u+x -print -quit)"
    [ -n "$target_make" ] && [ -n "$binary" ] && [ ! -L "$binary" ] || {
      fail 'build identity is incomplete'; return 2; }
    grep -Fqx '[build.sh] FQ_SWEEP_WEIGHT_LAYOUT=1' "$build_log" && \
      grep -F "FullyQuantized internal sweep: q=12 A=0 bc=0 format=0 layout=1 units=$units" \
        "$build_dir/cmake.log" >/dev/null && \
      grep -Eq '^FQ_SWEEP_WEIGHT_LAYOUT(:[^=]*)?=1$' "$build_dir/CMakeCache.txt" && \
      grep -Eq -- '(^|[[:space:]])-DFQ_SWEEP_WEIGHT_LAYOUT=1([[:space:]]|$)' \
        "$target_make" || {
      fail 'generated/layout build ABI differs'; return 2; }
    if grep -Eq -- '-DFQ_TC_KPACK4_DELIVERY_N(=|[[:space:]])' "$target_make" || \
       grep -Eq -- '-DPPU_PACKED_SCALE_FUSED(=|[[:space:]])' "$target_make" || \
       grep -F 'PPU_PACKED_SCALE_FUSED_READ' "$target_make" >/dev/null; then
      fail 'pilot binary carries a delivery/fused-metadata experiment'; return 2
    fi
    printf '%s\n' "$binary" > "$out/results/binary.path" || return 2
    sha256sum "$binary" "$target_make" "$manifest" \
      > "$out/results/build.sha256" || return 2
  else
    [ "${RESUME:-0}" = 1 ] || {
      fail 'build exists without RESUME=1'; return 2; }
    sha256sum -c "$out/results/build.sha256" >/dev/null || {
      fail 'build authority changed'; return 2; }
    binary="$(cat "$out/results/binary.path")"
    [ -x "$binary" ] && [ ! -L "$binary" ] || {
      fail 'resumed binary is missing or linked'; return 2; }
  fi

  source_state="$({
    git -C "$root" rev-parse HEAD
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum "$inventory" "$policy" "$master" "$plan" "$manifest" "$binary" \
      "$root/tools/analyze_fq_q4k_kpack4_prefill_pilot.py" \
      "$root/tools/run_fq_q4k_kpack4_prefill_pilot_box.sh" \
      "$root/ci/check_fq_q4k_kpack4_prefill_pilot.py"
  } | sha256sum | awk '{print $1}')" || return 2
  saved_state="$out/source-state.sha256"
  if [ -s "$saved_state" ] && [ "$(cat "$saved_state")" != "$source_state" ]; then
    fail 'source/inventory/binary authority changed on resume'; return 2
  fi
  atomic_text "$saved_state" "$source_state" || return 2
  git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2

  screen_log="$out/raw/screen.log"
  printf '[fq-kpack4-prefill] shape=2048x1024x5120 phase=screen typed=630 S=1\n'
  run_phase "$screen_log" "$binary" --shape=2048x1024x5120 \
    --iterations=2 --correctness-repeats=1 --only-split=1 \
    --tm8-max-m=8 --bc-mode=all || return $?
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_prefill_pilot.py" screen \
    --manifest "$manifest" --log "$screen_log" --policy "$policy" \
    --symbols-output "$out/results/screen-symbols.txt" \
    --summary-output "$out/results/screen.json" || return 2

  scheduler_log="$out/raw/scheduler.log"
  printf '[fq-kpack4-prefill] shape=2048x1024x5120 phase=scheduler splits=1,2,4,8\n'
  run_phase "$scheduler_log" "$binary" --shape=2048x1024x5120 \
    --iterations=1 --correctness-repeats=1 --only-split=0 \
    --symbols-file="$out/results/screen-symbols.txt" \
    --tm8-max-m=8 --bc-mode=skip || return $?
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_prefill_pilot.py" scheduler \
    --manifest "$manifest" --log "$scheduler_log" --policy "$policy" \
    --screen-symbols "$out/results/screen-symbols.txt" \
    --symbols-output "$out/results/confirm-symbols.txt" \
    --summary-output "$out/results/scheduler.json" || return 2

  confirm_log="$out/raw/confirm.log"
  printf '[fq-kpack4-prefill] shape=2048x1024x5120 phase=confirm iterations=7\n'
  run_phase "$confirm_log" "$binary" --shape=2048x1024x5120 \
    --iterations=7 --correctness-repeats=2 --only-split=0 \
    --symbols-file="$out/results/confirm-symbols.txt" \
    --tm8-max-m=8 --bc-mode=skip || return $?
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_prefill_pilot.py" finalize \
    --manifest "$manifest" --log "$confirm_log" --policy "$policy" \
    --symbols "$out/results/confirm-symbols.txt" \
    --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" || return 2

  python3 -B - "$out/results/summary.json" <<'PY' || return 2
import json, sys
value = json.load(open(sys.argv[1]))
for tk in (64, 128, 256):
    score = value["by_tactic_tile_k"][str(tk)]
    winner = score["winner"]
    runner = score["runner_up"]
    gap = "NONE" if runner is None else \
        f"{(runner['median_us'] / winner['median_us'] - 1) * 100:.6f}"
    print("FQ_KPACK4_PREFILL_TK "
          f"TK={tk} verdict={score['verdict']} candidates={score['candidates']} "
          f"algorithm={winner['algorithm']} S={winner['split']} "
          f"e2e_us={winner['median_us']:.9f} config={winner['config']} "
          f"runner_gap_pct={gap}")
PY

  sha256sum "$plan" "$manifest" "$binary" "$screen_log" "$scheduler_log" \
    "$confirm_log" "$out/results/screen-symbols.txt" \
    "$out/results/confirm-symbols.txt" "$out/results/summary.json" \
    > "$out/results/authority.sha256" || return 2
  sed -n '1,5p' "$out/results/summary.tsv"
  printf '[fq-kpack4-prefill] PASS sha=%s shape=2048x1024x5120 TM64=630/AP0 TK64+128+256 auto64/plain artifacts=%s\n' \
    "$sha" "$out"
  printf '[fq-kpack4-prefill] summary=%s\n' "$out/results/summary.tsv"
}

main "$@"
