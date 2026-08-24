#!/usr/bin/env bash
# Hash-bound first-prefix bisection for the legacy Q4_K Split-K shared epilogue.
set -uo pipefail

main() {
  local root workspace_root sha short stamp out jobs repeats sdk_root hgobjdump
  local full generated arm defs def build_dir build_log binary summary_log
  local direct_log probe_log build_rc direct_rc probe_rc list_elf

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    echo '[fq-shared-prefix-root] FAIL: ambient defines change exact arms' >&2
    return 2
  fi
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-shared-prefix-root-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) echo "[fq-shared-prefix-root] FAIL: OUT must be below /workspace: $out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    echo "[fq-shared-prefix-root] FAIL: refusing to overwrite $out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  repeats="${PROBE_REPEATS:-512}"
  case "$jobs" in
    *[!0-9]*|0) echo '[fq-shared-prefix-root] FAIL: JOBS must be positive' >&2; return 2 ;;
  esac
  case "$repeats" in
    *[!0-9]*|0) echo '[fq-shared-prefix-root] FAIL: PROBE_REPEATS must be positive' >&2; return 2 ;;
  esac
  mkdir -p "$out/generated/full" "$out/generated/closure" "$out/results" || return 2
  summary_log="$out/results/summary.log"
  : >"$summary_log" || return 2

  # Numeric correctness is the admission gate.  Resource reports are captured
  # from the SDK-owned disassembler to distinguish an operation seam from a
  # register/spill threshold without making disassembly availability optional.
  sdk_root="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
  hgobjdump="${HGOBJDUMP:-${sdk_root:+$sdk_root/bin/hgobjdump}}"
  if [ -z "$hgobjdump" ] || [ ! -x "$hgobjdump" ]; then
    hgobjdump="$(command -v hgobjdump 2>/dev/null || true)"
  fi
  if [ -z "$hgobjdump" ] || [ ! -x "$hgobjdump" ]; then
    echo '[fq-shared-prefix-root] FAIL: SDK hgobjdump is required' >&2
    return 2
  fi
  hgobjdump="$(readlink -f "$hgobjdump")" || return 2
  "$hgobjdump" --version >"$out/results/hgobjdump.version" 2>&1 || return 2
  sha256sum "$hgobjdump" >"$out/results/hgobjdump.sha256" || return 2

  local -a source_paths=(
    dev/fold_derivation/Q4K_FQ_SPLITK_PACKED_OWNER_DEBUG.md
    quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_direct_accumulator_store.hpp
    quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_shared_epilogue_sync_probe.hpp
    quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_shared_prefix_probe.hpp
    quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_splitk_parallel.hpp
    tools/check_fq_split_shared_prefix_root.py
    tools/run_fq_q4k_split_shared_prefix_root_box.sh
    ci/check_fq_split_shared_prefix_root.py
  )
  for source_path in "${source_paths[@]}"; do
    git -C "$root" diff --quiet "$sha" -- "$source_path" || {
      echo "[fq-shared-prefix-root] FAIL: source differs from HEAD: $source_path" >&2
      return 2
    }
  done
  (cd "$root" && sha256sum "${source_paths[@]}") >"$out/results/source.before.sha256" || return 2

  git -C "$root" show \
    "$sha:dev/fold_derivation/l223_fq_splitk_shared_epilogue_layout.expected.txt" \
    >"$out/results/l223-committed-evidence.log" || {
      echo '[fq-shared-prefix-root] FAIL: committed L223 evidence missing' >&2
      return 2
    }
  grep -Fqx \
    '[l223] PASS: exact legacy R2S/S2R map plus value-coordinate and owner negatives' \
    "$out/results/l223-committed-evidence.log" || {
      echo '[fq-shared-prefix-root] FAIL: committed L223 oracle is not admitted' >&2
      return 2
    }

  python3 -B "$root/tools/select_fq_split_timing_closure.py" --self-test || return 2
  python3 -B "$root/tools/check_fq_split_shared_prefix_root.py" --self-test || return 2
  python3 -B "$root/ci/check_fq_split_shared_prefix_root.py" || return 2
  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --tile-m-filter 8 \
    --per-unit 1 --out-dir "$out/generated/full" || return 2
  full="$out/generated/full"
  generated="$out/generated/closure"
  python3 -B "$root/tools/select_fq_split_timing_closure.py" \
    --source-dir "$full" --out-dir "$generated" || return 2

  local -a arms=(
    production-direct
    accumulator-opaque
    clone-opaque
    cta-only
    flat-constant-disjoint
    flat-accumulator-disjoint
    r2s-vector-disjoint
    r2s-scalar-disjoint
    r2s-snapshot-disjoint
    r2s-s2r-vector-disjoint
    r2s-s2r-scalar-disjoint
    full-discard
  )

  for arm in "${arms[@]}"; do
    case "$arm" in
      production-direct)
        defs='' ;;
      accumulator-opaque)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=1' ;;
      clone-opaque)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=2' ;;
      cta-only)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=3' ;;
      flat-constant-disjoint)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=4 PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1' ;;
      flat-accumulator-disjoint)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=5 PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1' ;;
      r2s-vector-disjoint)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=6 PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1' ;;
      r2s-scalar-disjoint)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=7 PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1' ;;
      r2s-snapshot-disjoint)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=8 PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1' ;;
      r2s-s2r-vector-disjoint)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=9 PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1' ;;
      r2s-s2r-scalar-disjoint)
        defs='PPU_SPLITK_SHARED_PREFIX_POLICY=10 PPU_SPLITK_SHARED_PROBE_DISJOINT_STORAGE=1' ;;
      full-discard)
        defs='PPU_SPLITK_SHARED_SYNC_POLICY=3 PPU_SPLITK_SHARED_PROBE_DISCARD_GMEM=1' ;;
      *) return 2 ;;
    esac

    build_dir="$out/build-$arm"
    build_log="$out/results/$arm-build.log"
    mkdir -p "$build_dir" || return 2
    (cd "$root" && PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 JOBS="$jobs" \
      PPU_DEFS="$defs" TARGET=test_fully_quantized_internal_sweep \
      FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
      FQ_SWEEP_ARTIFACT_TK=64 FQ_SWEEP_BCHUNK=0 \
      FQ_SWEEP_PACKED_FORMAT=0 ./build.sh) >"$build_log" 2>&1
    build_rc=$?
    if [ "$build_rc" -ne 0 ]; then
      echo "[fq-shared-prefix-root] FAIL: $arm build rc=$build_rc" >&2
      tail -180 "$build_log" >&2
      echo "[fq-shared-prefix-root] artifacts=$out" >&2
      return "$build_rc"
    fi
    for def in $defs; do
      grep -Fq -- "-D$def" "$build_log" || {
        echo "[fq-shared-prefix-root] FAIL: $arm define missing: $def" >&2
        return 2
      }
    done
    if grep -Fq -- '-DPPU_PACKED_METADATA_OWNER_ONLY' "$build_log"; then
      echo "[fq-shared-prefix-root] FAIL: $arm contains metadata override" >&2
      return 2
    fi
    if [ "$arm" = production-direct ] && \
       grep -Eq -- '-DPPU_SPLITK_(SHARED_PREFIX_POLICY|SHARED_SYNC_POLICY|LEGACY_SHARED_PARTIAL_EPILOGUE)' "$build_log"; then
      echo '[fq-shared-prefix-root] FAIL: production control contains a diagnostic define' >&2
      return 2
    fi

    binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
    if [ ! -x "$binary" ] || [ -L "$binary" ]; then
      binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
    fi
    if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
      echo "[fq-shared-prefix-root] FAIL: $arm binary missing: $binary" >&2
      return 2
    fi

    list_elf="$out/results/$arm-list-elf.txt"
    "$hgobjdump" -lelf "$binary" >"$list_elf" \
      2>"$out/results/$arm-list-elf.err" || {
        echo "[fq-shared-prefix-root] FAIL: $arm hgobjdump -lelf failed" >&2
        return 2
      }
    local resource_count=0 candidate pretty resource registers stack_line
    while IFS= read -r candidate; do
      pretty="$(c++filt "$candidate")"
      if [[ "$pretty" != *GemmUniversalMixedInputSplitKParallel* ]]; then
        continue
      fi
      resource_count=$((resource_count + 1))
      printf '%s\n' "$candidate" >"$out/results/$arm-kernel-${resource_count}.symbol"
      printf '%s\n' "$pretty" >"$out/results/$arm-kernel-${resource_count}.demangled"
      resource="$out/results/$arm-kernel-${resource_count}.resource"
      "$hgobjdump" "-res-usage=$candidate" "$binary" >"$resource" \
        2>"$out/results/$arm-kernel-${resource_count}.resource.err" || return 2
      registers="$(sed -n 's/^Registers:[[:space:]]*//p' "$resource" | head -n1)"
      stack_line="$(sed -n '/^Stack Frame:/p' "$resource" | head -n1)"
      printf 'FQ_SHARED_PREFIX_RESOURCE arm=%s kernel=%d registers=%s stack=%q\n' \
        "$arm" "$resource_count" "${registers:-UNKNOWN}" "${stack_line:-UNKNOWN}" \
        | tee -a "$summary_log"
    done < <(sed -n 's/^.*Func [0-9][0-9]*:[[:space:]]*\([^[:space:]]*\).*$/\1/p' "$list_elf")
    if [ "$resource_count" -eq 0 ]; then
      echo "[fq-shared-prefix-root] FAIL: $arm has no split-K kernel symbol" >&2
      return 2
    fi

    direct_log="$out/results/$arm-direct.log"
    probe_log="$out/results/$arm-probe.log"
    "$binary" --shape=1x1024x5120 --iterations=1 \
      --correctness-repeats="$repeats" --tm8-max-m=8 --bc-mode=skip \
      >"$direct_log" 2>&1
    direct_rc=$?
    "$binary" --shape=1x1024x5120 --iterations=1 \
      --correctness-repeats="$repeats" --tm8-max-m=8 --bc-mode=skip \
      --split-workspace-probe >"$probe_log" 2>&1
    probe_rc=$?
    if { [ "$direct_rc" -ne 0 ] && [ "$direct_rc" -ne 1 ]; } || \
        [ "$probe_rc" -ne 0 ]; then
      echo "[fq-shared-prefix-root] FAIL: $arm rc direct=$direct_rc probe=$probe_rc" >&2
      tail -100 "$direct_log" >&2
      tail -120 "$probe_log" >&2
      echo "[fq-shared-prefix-root] artifacts=$out" >&2
      return 2
    fi
    printf 'FQ_SHARED_PREFIX_BINARY arm=%s defs=%q sha256=%s direct_rc=%d probe_rc=%d repeats=%d kernels=%d\n' \
      "$arm" "$defs" "$(sha256sum "$binary" | awk '{print $1}')" \
      "$direct_rc" "$probe_rc" "$repeats" "$resource_count" \
      | tee -a "$summary_log"
  done

  local -a verdict_args=()
  for arm in "${arms[@]}"; do
    verdict_args+=(
      "--$arm-direct" "$out/results/$arm-direct.log"
      "--$arm-probe" "$out/results/$arm-probe.log"
    )
  done
  python3 -B "$root/tools/check_fq_split_shared_prefix_root.py" \
    "${verdict_args[@]}" \
    | tee "$out/results/verdict.log" \
    | tee -a "$summary_log"
  local verdict_rc=${PIPESTATUS[0]}

  (cd "$root" && sha256sum "${source_paths[@]}") >"$out/results/source.after.sha256" || return 2
  cmp "$out/results/source.before.sha256" "$out/results/source.after.sha256" || {
    echo '[fq-shared-prefix-root] FAIL: source authority changed during run' >&2
    return 2
  }
  sha256sum "$generated/manifest.json" "$out"/results/*.log \
    >"$out/results/authority.sha256" || return 2
  if [ "$verdict_rc" -ne 0 ]; then
    echo "[fq-shared-prefix-root] UNADJUDICATED artifacts=$out" >&2
    return "$verdict_rc"
  fi
  echo "[fq-shared-prefix-root] ROOT_CAUSE_COMPLETE sha=$sha artifacts=$out"
}

main "$@"
