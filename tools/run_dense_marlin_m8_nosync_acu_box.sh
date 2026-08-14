#!/usr/bin/env bash
# Diagnostic-only ACU capture for the standalone m8 Marlin cooperative.
#
# ordered keeps the shipping fp16 D-chain and lock; racy-d-chain removes only
# the cross-CTA acquire/release; final-local also removes the D-chain and lets
# the final peer publish its incomplete local partial.  The latter two are
# intentionally numerically invalid and are never sent through verification.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_marlin_m8_ab
ACU="${ACU:-$(command -v acu || true)}"
OUT="${MARLIN_M8_NOSYNC_ACU_OUT:-/workspace/quactlize-dense-marlin-m8-nosync-acu}"

fail() { printf '[marlin-m8-nosync-acu] FAIL: %s\n' "$*" >&2; exit 1; }
[ "$#" -eq 0 ] || fail 'this runner accepts no positional arguments'
[ -n "$ACU" ] && [ -x "$ACU" ] || \
  fail 'ACU is unavailable; set ACU to the site acu executable'
[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ] || \
  fail 'source tree must be clean so every binary/report names one exact SHA'
if [ -e "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  fail "output directory is not empty: $OUT"
fi
mkdir -p "$OUT"

ROOT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
ACTLIZE_SHA="$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)"
git -C "$ROOT" submodule status --recursive >"$OUT/submodule-status.txt"
grep -Eq '^[+\-U]' "$OUT/submodule-status.txt" && \
  fail 'a submodule checkout differs from its recorded gitlink'

ARGS=(--marlin --streamk_exact_fixture --m=1 --n=4096 --k=4096
      --l=1 --g=128 --mode=1 --alpha=1 --beta=0 --iterations=0)
MODES=(ordered racy-d-chain final-local)
BPCS=(1 2 3)
REPORTS=()
BINARIES=()

for mode in "${MODES[@]}"; do
  case "$mode" in
    ordered) hack=0; expected_sync=enabled; expected_correctness=NOT_EVALUATED_PROFILE_ONLY ;;
    racy-d-chain) hack=1; expected_sync=disabled; expected_correctness=INVALID_DATA_RACE ;;
    final-local) hack=2; expected_sync=disabled; expected_correctness=INVALID_MISSING_PEERS ;;
    *) fail "internal unknown mode: $mode" ;;
  esac
  mode_dir="$OUT/$mode"
  build_dir="$mode_dir/build"
  mkdir -p "$build_dir"
  BUILD=(env PPU_BUILD_DIR="$build_dir" PPU_ARCHS=ppu0010 TARGET="$TARGET"
         QUANT=int4 BENCH_GS=128)
  if [ "$hack" -ne 0 ]; then
    BUILD+=(PPU_DEFS="PPU_MARLIN_HANDOFF_HACK=$hack")
  fi
  BUILD+=("$ROOT/build.sh")
  printf '%q ' "${BUILD[@]}" >"$mode_dir/build.command"
  printf '\n' >>"$mode_dir/build.command"
  "${BUILD[@]}" 2>&1 | tee "$mode_dir/build.log"
  if [ "$hack" -ne 0 ]; then
    grep -Fq "PPU_DEFS verified on $TARGET's compile command: -DPPU_MARLIN_HANDOFF_HACK=$hack" \
      "$mode_dir/build.log" || fail "$mode build did not prove its compile-time diagnostic define"
  fi

  mapfile -t bins < <(find "$build_dir" -type f -name "$TARGET" -perm -u+x -print)
  [ "${#bins[@]}" -eq 1 ] || fail "expected one $mode binary, found ${#bins[@]}"
  bin="${bins[0]}"
  BINARIES+=("$bin")
  sha256sum "$bin" >"$mode_dir/binary.sha256"

  if [ "$hack" -eq 0 ]; then
    # Re-establish numerical correctness and lock lifecycle at this exact SHA
    # before any diagnostic report is admitted.
    for bpc in "${BPCS[@]}"; do
      bpc_flag=()
      [ "$bpc" -eq 1 ] || bpc_flag=("--marlin-blocks-per-cu=$bpc")
      cmd=("$bin" "${ARGS[@]}" "${bpc_flag[@]}")
      printf '%q ' "${cmd[@]}" >"$mode_dir/correctness-bpc${bpc}.command"
      printf '\n' >>"$mode_dir/correctness-bpc${bpc}.command"
      "${cmd[@]}" 2>&1 | tee "$mode_dir/correctness-bpc${bpc}.log"
      grep -Fq 'Disposition: Passed' "$mode_dir/correctness-bpc${bpc}.log" || \
        fail "ordered correctness failed for blocks_per_cu=$bpc"
      [ "$(grep -Ec '^  \[dense marlin lock fingerprint\] repeat=[1-8]/8 raw_bitdiff=0 .* stable=1 same-workspace=1 external-lock-reset=0$' "$mode_dir/correctness-bpc${bpc}.log" || true)" -eq 8 ] || \
        fail "ordered lock fingerprint failed for blocks_per_cu=$bpc"
    done
  else
    # A diagnostic binary must reject an ordinary numerical invocation before
    # launch; otherwise a racing/incomplete result could be mistaken for a
    # correctness failure or, worse, a numerical pass.
    set +e
    "$bin" "${ARGS[@]}" >"$mode_dir/ordinary-rejected.log" 2>&1
    reject_rc=$?
    set -e
    [ "$reject_rc" -ne 0 ] || fail "$mode ordinary invocation unexpectedly ran"
    grep -Fq "PPU_MARLIN_HANDOFF_HACK=$hack is a numerically invalid ACU diagnostic" \
      "$mode_dir/ordinary-rejected.log" || fail "$mode rejection did not name the diagnostic mode"
  fi

  for bpc in "${BPCS[@]}"; do
    bpc_flag=()
    [ "$bpc" -eq 1 ] || bpc_flag=("--marlin-blocks-per-cu=$bpc")
    report_base="$mode_dir/marlin-m8-${mode}-bpc${bpc}.report"
    log="$mode_dir/acu-bpc${bpc}.log"
    cmd_file="$mode_dir/acu-bpc${bpc}.command"
    ACU_CMD=("$ACU" -f -o "$report_base" --set full "$bin" "${ARGS[@]}"
             --marlin-profile-subject-only "${bpc_flag[@]}")
    printf '%q ' "${ACU_CMD[@]}" >"$cmd_file"
    printf '\n' >>"$cmd_file"
    "${ACU_CMD[@]}" 2>&1 | tee "$log"
    report_candidates=()
    [ -s "$report_base" ] && report_candidates+=("$report_base")
    [ -s "${report_base}.acurep" ] && report_candidates+=("${report_base}.acurep")
    [ "${#report_candidates[@]}" -eq 1 ] || \
      fail "$mode/BPC$bpc produced ${#report_candidates[@]} unambiguous reports"
    report="${report_candidates[0]}"
    REPORTS+=("$report")
    marker="[dense marlin ACU subject-only] instruction=m8n16k16 blocks_per_cu=$bpc subject_launches=1 device_reference=0 lock_fingerprints=0 handoff=$mode peer_sync=$expected_sync local_cta_sync=enabled numerical_correctness=$expected_correctness"
    grep -Fq "$marker" "$log" || fail "$mode/BPC$bpc subject marker did not close"
    [ "$(grep -Fc '[dense marlin ACU subject-only]' "$log" || true)" -eq 1 ] || \
      fail "$mode/BPC$bpc subject marker count is not one"
    ! grep -Fq 'Disposition:' "$log" || fail "$mode/BPC$bpc ran numerical verification"
    ! grep -Fq '[dense marlin lock fingerprint]' "$log" || fail "$mode/BPC$bpc ran lock fingerprints"
  done
done

{
  printf 'root_sha=%s\nactlize_sha=%s\n' "$ROOT_SHA" "$ACTLIZE_SHA"
  printf 'target=%s\nshape=M1,N4096,K4096,L1,gs128\n' "$TARGET"
  printf 'config=TM8,TN128,TK128,WM8,WN64,WarpK32,S4\n'
  printf 'modes=ordered,racy-d-chain,final-local\nbpcs=1,2,3\n'
  printf 'diagnostic_contract=racy-d-chain retains the fp16 D-chain but has a data race; final-local omits peer contributions; neither is a numerical candidate\n'
  printf 'local_cta_sync=retained\n'
  printf 'reports='
  printf '%s,' "${REPORTS[@]}"
  printf '\n'
} >"$OUT/manifest.txt"

find "$OUT" -type f ! -name bundle.sha256 -print0 | sort -z | \
  xargs -0 sha256sum >"$OUT/bundle.sha256"

printf '[marlin-m8-nosync-acu] CAPTURED: ordered/racy-d-chain/final-local x BPC1/2/3\n'
printf '[marlin-m8-nosync-acu] root-sha=%s\n' "$ROOT_SHA"
printf '[marlin-m8-nosync-acu] diagnostic reports make no numerical correctness claim\n'
