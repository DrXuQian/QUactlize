#!/usr/bin/env bash
# Build and run the complete M1 packed-A tactic x fixed-Split-K sweep.
# All durable artifacts live below /workspace; no unnamed temporary directory is used.
set -uo pipefail

main() {
  local root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  local sha short stamp out workspace_root build_dir build_log run_log manifest binary rc
  local source_state_before source_state_after
  if [ "${SPAN_CURVE:-0}" = 1 ] && [ "${EXACT_WARM_AB:-0}" = 1 ]; then
    echo "[splitk-sweep] FAIL: SPAN_CURVE and EXACT_WARM_AB are separate protocols"
    return 2
  fi
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="${OUT:-/workspace/quactlize-dense-splitk-sweep-$short-$stamp-$$}"
  workspace_root="$(realpath -e /workspace)" || return 2
  out="$(realpath -m -- "$out")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) echo "[splitk-sweep] FAIL: OUT must be a strict /workspace child: $out"; return 2 ;;
  esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    echo "[splitk-sweep] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS would change the search domain"
    return 2
  fi
  if [ -n "$(git -C "$root/third_party/actlize" status --porcelain)" ]; then
    echo "[splitk-sweep] FAIL: third_party/actlize is dirty; the binary would not be reconstructible"
    return 2
  fi
  if [ -e "$out" ]; then
    echo "[splitk-sweep] FAIL: refusing to overwrite existing bundle: $out"
    return 2
  fi
  mkdir "$out" || return 2
  build_dir="$out/build"
  build_log="$out/build.log"
  run_log="$out/sweep.log"
  manifest="$out/manifest.txt"

  source_state_before="$({
    git -C "$root" rev-parse HEAD
    git -C "$root" status --porcelain=v1
    git -C "$root" diff --binary --no-ext-diff HEAD
    while IFS= read -r path; do
      sha256sum "$root/$path"
    done < <(git -C "$root" ls-files --others --exclude-standard | LC_ALL=C sort)
  } | sha256sum | awk '{print $1}')" || return 2
  git -C "$root" diff --binary --no-ext-diff HEAD >"$out/source.patch" || return 2
  git -C "$root" ls-files --others --exclude-standard >"$out/untracked.files" || return 2
  if [ -s "$out/untracked.files" ]; then
    tar -C "$root" -cf "$out/untracked-source.tar" -T "$out/untracked.files" || return 2
  fi

  printf '[splitk-sweep] sha=%s out=%s\n' "$sha" "$out"
  (cd "$root" &&
    PPU_BUILD_DIR="$build_dir" \
    PPU_ARCHS=ppu0010 \
    JOBS="${JOBS:-16}" \
    TARGET=test_lowbit_dense_splitk_sweep \
    ./build.sh) >"$build_log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[splitk-sweep] FAIL: build returned $rc; tail follows"
    tail -80 "$build_log"
    return "$rc"
  fi
  binary="$(grep -m1 '^built: ' "$build_log" | cut -d' ' -f2-)"
  if [ -z "$binary" ] || [ ! -x "$binary" ]; then
    echo "[splitk-sweep] FAIL: build did not report an executable"
    tail -80 "$build_log"
    return 2
  fi
  source_state_after="$({
    git -C "$root" rev-parse HEAD
    git -C "$root" status --porcelain=v1
    git -C "$root" diff --binary --no-ext-diff HEAD
    while IFS= read -r path; do
      sha256sum "$root/$path"
    done < <(git -C "$root" ls-files --others --exclude-standard | LC_ALL=C sort)
  } | sha256sum | awk '{print $1}')" || return 2
  if [ "$source_state_before" != "$source_state_after" ]; then
    echo "[splitk-sweep] FAIL: source tree changed while the binary was built"
    return 2
  fi

  {
    printf 'git_sha=%s\n' "$sha"
    printf 'actlize_sha=%s\n' "$(git -C "$root/third_party/actlize" rev-parse HEAD)"
    printf 'ppu_archs=ppu0010\n'
    printf 'source_state_sha256=%s\n' "$source_state_before"
    printf 'git_status_begin\n'
    git -C "$root" status --short
    printf 'git_status_end\n'
    printf 'binary=%s\n' "$binary"
    printf 'binary_sha256=%s\n' "$(sha256sum "$binary" | awk '{print $1}')"
    printf 'table_sha256=%s\n' "$(sha256sum "$root/benchmarks/lowbit_dense_configs.inc" | awk '{print $1}')"
    printf 'source_patch_sha256=%s\n' "$(sha256sum "$out/source.patch" | awk '{print $1}')"
    if [ -f "$out/untracked-source.tar" ]; then
      printf 'untracked_source_sha256=%s\n' "$(sha256sum "$out/untracked-source.tar" | awk '{print $1}')"
    else
      printf 'untracked_source_sha256=NONE\n'
    fi
    if [ "${EXACT_WARM_AB:-0}" = 1 ]; then
      printf 'timing_protocol=exact_same_address_warm_aggregate_historical_vs_shipping_ordinary_vs_packedA_reshape_vs_internal_S8_producer_two_launch_and_actual_last_fused_e2e\n'
    else
      printf 'cold_protocol=full_B_plus_scale_rotation_over_max_2.16xL2_128MiB\n'
    fi
    printf 'command=%q' "$binary"
    local timed_iterations="${ITERATIONS:-5}"
    [ "${EXACT_WARM_AB:-0}" = 1 ] && timed_iterations="${ITERATIONS:-100}"
    printf ' %q' \
      "--iterations=$timed_iterations" \
      "--warmup-rotations=${WARMUP_ROTATIONS:-1}" \
      "--correctness-repeats=${CORRECTNESS_REPEATS:-1}" \
      "--cold-budget-mib=${COLD_BUDGET_MIB:-512}" \
      "--ce-ghz=${CE_GHZ:-1.70}" \
      "--hbm-gbs=${HBM_GBS:-2766}"
    [ "${SPAN_CURVE:-0}" = 1 ] && printf ' %q' "--span-curve"
    [ "${EXACT_WARM_AB:-0}" = 1 ] && printf ' %q' "--exact-warm-ab"
    [ -n "${L2_BYTES:-}" ] && printf ' %q' "--l2-bytes=$L2_BYTES"
    [ -n "${CU_COUNT:-}" ] && printf ' %q' "--cu=$CU_COUNT"
    printf '\n'
    command -v hgcc >/dev/null 2>&1 && hgcc --version 2>&1 | head -4 || true
  } >"$manifest"

  local timed_iterations="${ITERATIONS:-5}"
  [ "${EXACT_WARM_AB:-0}" = 1 ] && timed_iterations="${ITERATIONS:-100}"
  local -a args=(
    "--iterations=$timed_iterations"
    "--warmup-rotations=${WARMUP_ROTATIONS:-1}"
    "--correctness-repeats=${CORRECTNESS_REPEATS:-1}"
    "--cold-budget-mib=${COLD_BUDGET_MIB:-512}"
    "--ce-ghz=${CE_GHZ:-1.70}"
    "--hbm-gbs=${HBM_GBS:-2766}"
  )
  [ "${SPAN_CURVE:-0}" = 1 ] && args+=("--span-curve")
  [ "${EXACT_WARM_AB:-0}" = 1 ] && args+=("--exact-warm-ab")
  [ -n "${L2_BYTES:-}" ] && args+=("--l2-bytes=$L2_BYTES")
  [ -n "${CU_COUNT:-}" ] && args+=("--cu=$CU_COUNT")
  "$binary" "${args[@]}" 2>&1 | tee "$run_log"
  rc=${PIPESTATUS[0]}
  printf 'run_rc=%d\n' "$rc" >>"$manifest"
  printf 'run_log_sha256=%s\n' "$(sha256sum "$run_log" | awk '{print $1}')" >>"$manifest"
  if [ "$rc" -ne 0 ]; then
    echo "[splitk-sweep] completed with rc=$rc; artifacts preserved at $out"
    return "$rc"
  fi
  if [ "${EXACT_WARM_AB:-0}" = 1 ]; then
    local exact_count exact_schema_bad
    exact_count="$(grep -c '^EXACT_WARM_AB ' "$run_log" || true)"
    exact_schema_bad=0
    if [ "$exact_count" -ne 2 ]; then
      echo "[splitk-sweep] FAIL: exact protocol emitted $exact_count rows, expected 2"
      exact_schema_bad=1
    fi
    while IFS= read -r exact_line; do
      case "$exact_line" in *"fused_last_arriver_selected=1"*) ;; *) exact_schema_bad=1 ;; esac
      case "$exact_line" in *"fused_counters_zero=1"*) ;; *) exact_schema_bad=1 ;; esac
      case "$exact_line" in *"fused_slices=3/3"*) ;; *) exact_schema_bad=1 ;; esac
      case "$exact_line" in *"fused_reuse=8/8"*) ;; *) exact_schema_bad=1 ;; esac
      case "$exact_line" in *"post_timing=RAW-BIT/PASS"*) ;; *) exact_schema_bad=1 ;; esac
    done < <(grep '^EXACT_WARM_AB ' "$run_log" || true)
    if [ "$exact_schema_bad" -ne 0 ]; then
      echo "[splitk-sweep] FAIL: exact fused output schema/correctness contract is incomplete"
      return 2
    fi
    printf 'exact_fused_schema=2/2 PASS\n' >>"$manifest"
  fi
  echo "[splitk-sweep] PASS; artifacts: $out"
}

main "$@"
