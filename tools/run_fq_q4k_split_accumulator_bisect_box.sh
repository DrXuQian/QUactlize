#!/usr/bin/env bash
# One hash-bound diagnostic bundle: owner-only packed metadata with the
# shipping shared partial epilogue versus direct accumulator delivery.
set -uo pipefail

main() {
  local root workspace_root sha short stamp out jobs repeats
  local full generated arm defs build_dir build_log binary direct_log probe_log
  local l222_evidence
  local build_rc direct_rc probe_rc

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[fq-accumulator-bisect] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes the exact arms\n' >&2
    return 2
  fi
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-accumulator-bisect-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[fq-accumulator-bisect] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[fq-accumulator-bisect] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  repeats="${PROBE_REPEATS:-256}"
  case "$jobs" in
    *[!0-9]*|0) printf '[fq-accumulator-bisect] FAIL: JOBS must be positive\n' >&2; return 2 ;;
  esac
  case "$repeats" in
    *[!0-9]*|0) printf '[fq-accumulator-bisect] FAIL: PROBE_REPEATS must be positive\n' >&2; return 2 ;;
  esac
  mkdir -p "$out/generated/full" "$out/generated/closure" "$out/results" || return 2

  # The box's executable named nvcc delegates device preprocessing to
  # ppu_clang++, which enables PPU FP8 and then cannot find hggc_fp8.h in the
  # NVIDIA/stub fixture.  Do not add a fake header: it would shadow the real
  # SDK.  Consume the exact local CuTe oracle generated and committed at this
  # result SHA; both real PPU arms below are still compiled fresh by hgcc.
  l222_evidence="$out/results/l222-committed-evidence.log"
  git -C "$root" show \
    "$sha:dev/fold_derivation/l222_fq_splitk_direct_accumulator_store.expected.txt" \
    >"$l222_evidence" || {
      printf '[fq-accumulator-bisect] FAIL: result SHA lacks committed L222 evidence\n' >&2
      return 2
    }
  python3 -B "$root/tools/select_fq_split_timing_closure.py" --self-test || return 2
  python3 -B "$root/tools/check_fq_split_accumulator_bisect.py" --self-test || return 2
  python3 -B "$root/ci/check_fq_split_accumulator_bisect.py" || return 2
  python3 -B "$root/ci/check_fq_split_accumulator_bisect.py" \
    --committed-only --evidence "$l222_evidence" || return 2
  printf 'FQ_ACCUMULATOR_BISECT_ORACLE mode=committed-local-oracle source_sha=%s fresh_box_execution=0\n' \
    "$sha"
  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --tile-m-filter 8 \
    --per-unit 1 --out-dir "$out/generated/full" || return 2
  full="$out/generated/full"
  generated="$out/generated/closure"
  python3 -B "$root/tools/select_fq_split_timing_closure.py" \
    --source-dir "$full" --out-dir "$generated" || return 2

  for arm in shared-epilogue direct-accumulator; do
    defs="PPU_PACKED_METADATA_OWNER_ONLY=1"
    if [ "$arm" = direct-accumulator ]; then
      defs="$defs PPU_SPLITK_DIRECT_ACCUMULATOR_STORE=1"
    fi
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
      printf '[fq-accumulator-bisect] FAIL: %s build rc=%d\n' "$arm" "$build_rc" >&2
      tail -160 "$build_log" >&2
      printf '[fq-accumulator-bisect] artifacts=%s\n' "$out" >&2
      return "$build_rc"
    fi
    grep -Fq \
      "PPU_DEFS verified on test_fully_quantized_internal_sweep's compile command: -DPPU_PACKED_METADATA_OWNER_ONLY=1" \
      "$build_log" || {
        printf '[fq-accumulator-bisect] FAIL: %s owner-only define missing\n' "$arm" >&2
        return 2
      }
    if [ "$arm" = direct-accumulator ]; then
      grep -Fq \
        "PPU_DEFS verified on test_fully_quantized_internal_sweep's compile command: -DPPU_SPLITK_DIRECT_ACCUMULATOR_STORE=1" \
        "$build_log" || {
          printf '[fq-accumulator-bisect] FAIL: direct-store define missing\n' >&2
          return 2
        }
    elif grep -Fq -- '-DPPU_SPLITK_DIRECT_ACCUMULATOR_STORE' "$build_log"; then
      printf '[fq-accumulator-bisect] FAIL: shared epilogue arm contains direct-store define\n' >&2
      return 2
    fi

    binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
    if [ ! -x "$binary" ] || [ -L "$binary" ]; then
      binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
    fi
    if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
      printf '[fq-accumulator-bisect] FAIL: %s binary missing: %s\n' "$arm" "$binary" >&2
      return 2
    fi

    direct_log="$out/results/$arm-direct.log"
    probe_log="$out/results/$arm-workspace-probe.log"
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
      printf '[fq-accumulator-bisect] FAIL: %s rc direct=%d probe=%d\n' \
        "$arm" "$direct_rc" "$probe_rc" >&2
      tail -80 "$direct_log" >&2
      tail -100 "$probe_log" >&2
      printf '[fq-accumulator-bisect] artifacts=%s\n' "$out" >&2
      return 2
    fi
    printf 'FQ_ACCUMULATOR_BISECT_ARM arm=%s binary_sha256=%s direct_rc=%d probe_rc=%d repeats=%d\n' \
      "$arm" "$(sha256sum "$binary" | awk '{print $1}')" \
      "$direct_rc" "$probe_rc" "$repeats"
  done

  python3 -B "$root/tools/check_fq_split_accumulator_bisect.py" \
    --epilogue-direct "$out/results/shared-epilogue-direct.log" \
    --epilogue-probe "$out/results/shared-epilogue-workspace-probe.log" \
    --accumulator-direct "$out/results/direct-accumulator-direct.log" \
    --accumulator-probe "$out/results/direct-accumulator-workspace-probe.log" \
    | tee "$out/results/verdict.log" || return 2
  sha256sum "$generated/manifest.json" "$out"/results/*.log \
    >"$out/results/authority.sha256" || return 2
  printf '[fq-accumulator-bisect] DIAGNOSTIC_COMPLETE sha=%s artifacts=%s\n' \
    "$sha" "$out"
}

main "$@"
