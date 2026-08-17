#!/usr/bin/env bash
# Full Q8_0 ScaleFirst internal comparison: every emitted tactic, ordinary DP,
# and every deduplicated capacity/balanced persistent grid authorized by the
# exact final-kernel occupancy.  One binary is built once and reused by every
# plan shape.  An interrupted bundle resumes at the first incomplete shape.
set -uo pipefail

main() {
  local root sha short stamp workspace_root out plan spec gguf build_dir build_log
  local binary table source_state manifest reps iterations jobs rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  out="${OUT:-/workspace/quactlize-scalefirst-internal-q8-$short-$stamp-$$}"
  out="$(realpath -m -- "$out")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) echo "[scalefirst-internal] FAIL: OUT must be a strict /workspace child: $out"; return 2 ;;
  esac

  table="$root/benchmarks/lowbit_dense_i8_configs.inc"
  reps="${REPS:-2}"
  iterations="${ITERATIONS:-5}"
  jobs="${JOBS:-16}"
  case "$reps:$iterations:$jobs" in
    *[!0-9:]*|0:*|*:0:*|*:*:0)
      echo "[scalefirst-internal] FAIL: REPS, ITERATIONS, JOBS must be positive integers"
      return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    echo "[scalefirst-internal] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes the denominator"
    return 2
  fi

  if [ -e "$out" ] && [ "${RESUME:-0}" != 1 ]; then
    echo "[scalefirst-internal] FAIL: $out exists; set RESUME=1 to continue it"
    return 2
  fi
  if [ ! -e "$out" ]; then
    mkdir "$out" || return 2
  fi
  mkdir -p "$out/raw" "$out/results" || return 2

  python3 "$root/tools/analyze_scalefirst_internal_sweep.py" --self-test || return 2
  python3 "$root/ci/check_scalefirst_policy_contract.py" || return 2
  python3 "$root/ci/check_scalefirst_q8_units.py" || return 2
  python3 "$root/ci/check_dense_tactic_table.py" --table "$table" \
    --q8-metadata-negative || return 2

  plan="${PLAN:-$out/plan.json}"
  if [ -n "${PLAN:-}" ]; then
    plan="$(realpath -e -- "$PLAN")" || return 2
  else
    gguf="${GGUF:-}"
    spec="${SPEC:-$root/benchmarks/qwen35_a3b_q8_overnight.json}"
    if [ -z "$gguf" ] || [ ! -f "$gguf" ]; then
      echo "[scalefirst-internal] FAIL: set GGUF=/absolute/model.gguf (or PLAN=/absolute/plan.json)"
      return 2
    fi
    if [ ! -s "$plan" ]; then
      python3 "$root/tools/prefill_sweep.py" plan \
        --spec "$spec" --gguf "$gguf" --output "$plan" || return 2
    fi
  fi
  python3 "$root/tools/prefill_sweep.py" admit --plan "$plan" || return 2
  python3 "$root/tools/analyze_scalefirst_internal_sweep.py" \
    --list-plan "$plan" > "$out/cells.tsv" || return 2
  if awk -F '\t' '$2 != 8 || $7 != "SUPPORTED" {bad=1} END {exit bad ? 0 : 1}' \
      "$out/cells.tsv"; then
    echo "[scalefirst-internal] FAIL: Q8 shard plan contains non-Q8 or unsupported cells"
    cat "$out/cells.tsv"
    return 2
  fi
  local plan_sha
  plan_sha="$(sha256sum "$plan" | awk '{print $1}')" || return 2
  if [ -s "$out/plan.sha256" ]; then
    if [ "$(cat "$out/plan.sha256")" != "$plan_sha" ]; then
      echo "[scalefirst-internal] FAIL: plan changed since this bundle started"
      return 2
    fi
  else
    printf '%s\n' "$plan_sha" > "$out/plan.sha256"
  fi

  source_state="$({
    git -C "$root" rev-parse HEAD
    sha256sum \
      "$root/benchmarks/test_lowbit_dense_bench.cu" \
      "$root/benchmarks/lowbit_dense_unit.inc" \
      "$root/benchmarks/emit_tactic_configs.cpp" \
      "$root/benchmarks/lowbit_dense_i8_configs.inc" \
      "$root/quactlize/include/ppu_tactic_space.hpp" \
      "$root/quactlize/include/scalefirst_persistent_policy.hpp" \
      "$root/quactlize/csrc/CMakeLists.txt.in" \
      "$root/quactlize/csrc/TacticTableUnits.cmake" \
      "$root/tools/run_scalefirst_internal_sweep_box.sh" \
      "$root/tools/analyze_scalefirst_internal_sweep.py" \
      "$root/ci/check_dense_tactic_table.py" \
      "$root/ci/check_scalefirst_policy_contract.py" \
      "$root/ci/check_scalefirst_q8_units.py" \
      "$root/dev/fold_derivation/l209_scalefirst_policy_contract.cpp" \
      "$root/build.sh"
    git -C "$root/third_party/actlize" rev-parse HEAD
  } | sha256sum | awk '{print $1}')" || return 2
  if [ -s "$out/source-state.sha256" ]; then
    if [ "$(cat "$out/source-state.sha256")" != "$source_state" ]; then
      echo "[scalefirst-internal] FAIL: source authority changed since this bundle started"
      return 2
    fi
  else
    printf '%s\n' "$source_state" > "$out/source-state.sha256"
    git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2
  fi

  build_dir="$out/build"
  build_log="$out/build.log"
  binary="$build_dir/ppu_targets/test_scalefirst_q8_persistent_sweep"
  if [ ! -x "$binary" ]; then
    echo "[scalefirst-internal] building 2501-row Q8 shard once: $build_dir"
    (cd "$root" &&
      PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
      TARGET=test_scalefirst_q8_persistent_sweep ./build.sh) > "$build_log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "[scalefirst-internal] FAIL: build rc=$rc; tail follows"
      tail -100 "$build_log"
      return "$rc"
    fi
    binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
  fi
  if [ -z "$binary" ] || [ ! -x "$binary" ]; then
    echo "[scalefirst-internal] FAIL: built executable not found"
    tail -100 "$build_log" 2>/dev/null || true
    return 2
  fi

  manifest="$out/manifest.txt"
  {
    printf 'schema=quactlize.scalefirst_internal_sweep.v1\n'
    printf 'git_sha=%s\n' "$sha"
    printf 'actlize_sha=%s\n' "$(git -C "$root/third_party/actlize" rev-parse HEAD)"
    printf 'source_state_sha256=%s\n' "$source_state"
    printf 'plan=%s\nplan_sha256=%s\n' "$plan" "$plan_sha"
    printf 'table=%s\ntable_sha256=%s\n' "$table" "$(sha256sum "$table" | awk '{print $1}')"
    printf 'binary=%s\nbinary_sha256=%s\n' "$binary" "$(sha256sum "$binary" | awk '{print $1}')"
    printf 'format=Q8_0 qtype=8 ArtifactTileK=32 FoldN_low=1 FoldN_high=1\n'
    printf 'source_rows=2501 prune=none tactic_tk=32,64,128,256 stages=2,3,4,6,8,12\n'
    printf 'algorithm=np+capacity+balanced reps=%s iterations=%s\n' "$reps" "$iterations"
  } > "$manifest"

  local id qtype m n k gs support log rc_file complete_shapes total_shapes
  total_shapes="$(wc -l < "$out/cells.tsv")"
  complete_shapes=0
  while IFS=$'\t' read -r id qtype m n k gs support; do
    log="$out/raw/$id.log"
    rc_file="$out/raw/$id.rc"
    if [ -s "$log" ] && [ "$(cat "$rc_file" 2>/dev/null || true)" = 0 ] &&
       grep -q '^Q8_POLICY_COMPLETE status=COMPLETE ' "$log"; then
      echo "[scalefirst-internal] resume: $id already COMPLETE"
      complete_shapes=$((complete_shapes + 1))
      continue
    fi
    echo "[scalefirst-internal] shape=$id M/N/K=$m/$n/$k ($((complete_shapes + 1))/$total_shapes)"
    BENCH_REPS="$reps" "$binary" \
      "--m=$m" "--n=$n" "--k=$k" --l=1 --g=32 --mode=1 \
      "--iterations=$iterations" --alpha=1 --beta=0 \
      --q8-persistent-policy-sweep > "$log" 2>&1
    rc=$?
    printf '%d\n' "$rc" > "$rc_file"
    if [ "$rc" -ne 0 ]; then
      echo "[scalefirst-internal] $id FAILED rc=$rc (continuing; final summary will be INCOMPLETE)"
      tail -40 "$log"
    else
      complete_shapes=$((complete_shapes + 1))
      grep '^Q8_POLICY_COMPLETE ' "$log" || true
    fi
    python3 "$root/tools/analyze_scalefirst_internal_sweep.py" \
      --plan "$plan" --raw-dir "$out/raw" --binary "$binary" --table "$table" \
      --output "$out/results/summary.json" >/dev/null 2>&1 || true
  done < "$out/cells.tsv"

  python3 "$root/tools/analyze_scalefirst_internal_sweep.py" \
    --plan "$plan" --raw-dir "$out/raw" --binary "$binary" --table "$table" \
    --output "$out/results/summary.json"
  rc=$?
  printf 'summary_sha256=%s\n' "$(sha256sum "$out/results/summary.json" | awk '{print $1}')" >> "$manifest"
  printf '[scalefirst-internal] artifacts=%s\n' "$out"
  return "$rc"
}

main "$@"
