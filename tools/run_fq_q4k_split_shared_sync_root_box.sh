#!/usr/bin/env bash
# One hash-bound root-cause bundle for the legacy Q4_K Split-K shared epilogue.
# All four device arms retain identical R2S/S2R tensor partitions and differ
# only in the two synchronization calls.
set -uo pipefail

main() {
  local root workspace_root sha short stamp out jobs repeats
  local full generated l223_evidence arm defs build_dir build_log binary
  local direct_log probe_log build_rc direct_rc probe_rc

  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || return 2
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    echo '[fq-shared-sync-root] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes exact arms' >&2
    return 2
  fi
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-shared-sync-root-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) echo "[fq-shared-sync-root] FAIL: OUT must be a strict /workspace child: $out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    echo "[fq-shared-sync-root] FAIL: refusing to overwrite $out" >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  repeats="${PROBE_REPEATS:-512}"
  case "$jobs" in
    *[!0-9]*|0) echo '[fq-shared-sync-root] FAIL: JOBS must be positive' >&2; return 2 ;;
  esac
  case "$repeats" in
    *[!0-9]*|0) echo '[fq-shared-sync-root] FAIL: PROBE_REPEATS must be positive' >&2; return 2 ;;
  esac
  mkdir -p "$out/generated/full" "$out/generated/closure" "$out/results" || return 2

  l223_evidence="$out/results/l223-committed-evidence.log"
  git -C "$root" show \
    "$sha:dev/fold_derivation/l223_fq_splitk_shared_epilogue_layout.expected.txt" \
    >"$l223_evidence" || {
      echo '[fq-shared-sync-root] FAIL: result SHA lacks committed L223 evidence' >&2
      return 2
    }
  cmp -s "$l223_evidence" \
    "$root/dev/fold_derivation/l223_fq_splitk_shared_epilogue_layout.expected.txt" || {
      echo '[fq-shared-sync-root] FAIL: working-tree L223 evidence differs from result SHA' >&2
      return 2
    }
  grep -Fqx \
    'L223_SHARED_EPILOGUE_LAYOUT writers=512 holes=0 duplicates=0 r2s_conflicts=0 readers=512 reader_holes=0 reader_duplicates=0 s2r_coord_bad=0 s2r_value_bad=0 verdict=PASS' \
    "$l223_evidence" || {
      echo '[fq-shared-sync-root] FAIL: L223 positive mapping evidence missing' >&2
      return 2
    }
  grep -Fqx \
    '[l223] PASS: exact legacy R2S/S2R map plus value-coordinate and owner negatives' \
    "$l223_evidence" || {
      echo '[fq-shared-sync-root] FAIL: L223 negative controls missing' >&2
      return 2
    }

  python3 -B "$root/tools/select_fq_split_timing_closure.py" --self-test || return 2
  python3 -B "$root/tools/check_fq_split_shared_sync_root.py" --self-test || return 2
  python3 -B "$root/ci/check_fq_split_shared_sync_root.py" || return 2
  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 64 --bchunk 0 --tile-m-filter 8 \
    --per-unit 1 --out-dir "$out/generated/full" || return 2
  full="$out/generated/full"
  generated="$out/generated/closure"
  python3 -B "$root/tools/select_fq_split_timing_closure.py" \
    --source-dir "$full" --out-dir "$generated" || return 2

  for arm in vendor-user0 clone-user0 reserved-id1 cta; do
    case "$arm" in
      vendor-user0) defs='PPU_SPLITK_LEGACY_SHARED_PARTIAL_EPILOGUE=1' ;;
      clone-user0) defs='PPU_SPLITK_SHARED_SYNC_POLICY=1' ;;
      reserved-id1) defs='PPU_SPLITK_SHARED_SYNC_POLICY=2' ;;
      cta) defs='PPU_SPLITK_SHARED_SYNC_POLICY=3' ;;
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
      echo "[fq-shared-sync-root] FAIL: $arm build rc=$build_rc" >&2
      tail -160 "$build_log" >&2
      echo "[fq-shared-sync-root] artifacts=$out" >&2
      return "$build_rc"
    fi
    grep -Fq -- "-D$defs" "$build_log" || {
      echo "[fq-shared-sync-root] FAIL: $arm compile define missing: $defs" >&2
      return 2
    }
    if grep -Fq -- '-DPPU_PACKED_METADATA_OWNER_ONLY' "$build_log"; then
      echo "[fq-shared-sync-root] FAIL: $arm contains diagnostic metadata override" >&2
      return 2
    fi

    binary="$build_dir/ppu_targets/test_fully_quantized_internal_sweep"
    if [ ! -x "$binary" ] || [ -L "$binary" ]; then
      binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
    fi
    if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
      echo "[fq-shared-sync-root] FAIL: $arm binary missing: $binary" >&2
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
      echo "[fq-shared-sync-root] FAIL: $arm rc direct=$direct_rc probe=$probe_rc" >&2
      tail -100 "$direct_log" >&2
      tail -120 "$probe_log" >&2
      echo "[fq-shared-sync-root] artifacts=$out" >&2
      return 2
    fi
    printf 'FQ_SHARED_SYNC_BINARY arm=%s defs=%s sha256=%s direct_rc=%d probe_rc=%d repeats=%d\n' \
      "$arm" "$defs" "$(sha256sum "$binary" | awk '{print $1}')" \
      "$direct_rc" "$probe_rc" "$repeats"
  done

  python3 -B "$root/tools/check_fq_split_shared_sync_root.py" \
    --vendor-user0-direct "$out/results/vendor-user0-direct.log" \
    --vendor-user0-probe "$out/results/vendor-user0-probe.log" \
    --clone-user0-direct "$out/results/clone-user0-direct.log" \
    --clone-user0-probe "$out/results/clone-user0-probe.log" \
    --reserved-id1-direct "$out/results/reserved-id1-direct.log" \
    --reserved-id1-probe "$out/results/reserved-id1-probe.log" \
    --cta-direct "$out/results/cta-direct.log" \
    --cta-probe "$out/results/cta-probe.log" \
    | tee "$out/results/verdict.log" || {
      echo "[fq-shared-sync-root] UNADJUDICATED artifacts=$out" >&2
      return 2
    }
  sha256sum "$generated/manifest.json" "$out"/results/*.log \
    >"$out/results/authority.sha256" || return 2
  echo "[fq-shared-sync-root] ROOT_CAUSE_COMPLETE sha=$sha artifacts=$out"
}

main "$@"
