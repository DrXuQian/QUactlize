#!/usr/bin/env bash
# Exact candidate/legacy device closure for the K-pack4 register destination.
set -uo pipefail

build_arm() {
  local root="$1" generated="$2" out="$3" label="$4" defs="$5" jobs="$6"
  local build log rc binary target_make def
  build="$out/build/$label"
  log="$out/results/build-$label.log"
  (cd "$root" && \
    PPU_BUILD_DIR="$build" PPU_ARCHS=ppu0010 JOBS="$jobs" \
    PPU_DEFS="$defs" TARGET=test_fully_quantized_internal_sweep \
    FQ_SWEEP_GENERATED_DIR="$generated" FQ_SWEEP_QTYPE=12 \
    FQ_SWEEP_ARTIFACT_TK=0 FQ_SWEEP_BCHUNK=0 \
    FQ_SWEEP_PACKED_FORMAT=0 FQ_SWEEP_WEIGHT_LAYOUT=1 \
    ./build.sh) >"$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '[fq-kpack4-fragment] FAIL: %s build rc=%d\n' "$label" "$rc" >&2
    tail -n 180 "$log" >&2
    return "$rc"
  fi
  target_make="$(find "$build" -type f \
    -path '*test_fully_quantized_internal_sweep.dir/build.make' \
    -print -quit 2>/dev/null)"
  if ! grep -Fqx '[build.sh] FQ_SWEEP_WEIGHT_LAYOUT=1' "$log" ||
     ! grep -F 'FullyQuantized internal sweep: q=12 A=0 bc=0 format=0 layout=1 units=1' \
       "$build/cmake.log" >/dev/null ||
     [ -z "$target_make" ] ||
     ! grep -Eq -- '(^|[[:space:]])-DFQ_SWEEP_WEIGHT_LAYOUT=1([[:space:]]|$)' \
       "$target_make"; then
    printf '[fq-kpack4-fragment] FAIL: %s layout build ABI is not exact\n' "$label" >&2
    return 2
  fi
  for def in $defs; do
    if ! grep -Fq "PPU_DEFS verified on test_fully_quantized_internal_sweep's compile command: -D${def}" "$log"; then
      printf '[fq-kpack4-fragment] FAIL: %s did not bind -D%s\n' "$label" "$def" >&2
      return 2
    fi
  done
  binary="$build/ppu_targets/test_fully_quantized_internal_sweep"
  if [ ! -x "$binary" ] || [ -L "$binary" ]; then
    binary="$(grep -m1 '^built: ' "$log" | cut -d' ' -f2-)"
  fi
  if [ -z "$binary" ] || [ ! -x "$binary" ] || [ -L "$binary" ]; then
    printf '[fq-kpack4-fragment] FAIL: %s binary missing\n' "$label" >&2
    return 2
  fi
  sha256sum "$binary" >"$out/results/binary-$label.sha256"
  printf '%s\n' "$binary"
}

run_arm() {
  local binary="$1" log="$2" repeats="$3"
  "$binary" --shape=1x1024x5120 --iterations=1 \
    --correctness-repeats="$repeats" --only-split=1 \
    --tm8-max-m=8 --bc-mode=skip | tee "$log"
  local rc=${PIPESTATUS[0]}
  return "$rc"
}

main() {
  local root workspace sha short stamp out jobs repeats full generated
  local candidate legacy candidate_log legacy_log candidate_rc legacy_rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-kpack4-fragment-${short}-${stamp}-$$}")" || return 2
  case "$out" in
    "$workspace"/*) ;;
    *) printf '[fq-kpack4-fragment] FAIL: OUT must be a strict /workspace child\n' >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[fq-kpack4-fragment] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[fq-kpack4-fragment] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes the arms\n' >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  repeats="${CORRECTNESS_REPEATS:-8}"
  case "$jobs:$repeats" in
    *[!0-9:]*|0:*|*:0) printf '[fq-kpack4-fragment] FAIL: JOBS/repeats must be positive\n' >&2; return 2 ;;
  esac
  full="$out/generated/full"
  generated="$out/generated/closure"
  mkdir -p "$full" "$generated" "$out/build" "$out/results" || return 2

  python3 -B "$root/tools/check_fq_q4k_kpack4_fragment_closure.py" self-test || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_generator.py" || return 2
  python3 -B "$root/tools/gen_fully_quantized_splitk_producer_units.py" \
    --qtype 12 --artifact-tk 0 --bchunk 0 --weight-layout q4-kpack4 \
    --tile-m-filter 8 --per-unit 144 --out-dir "$full" || return 2
  python3 -B "$root/tools/check_fq_q4k_kpack4_fragment_closure.py" select \
    --source-dir "$full" --out-dir "$generated" || return 2

  git -C "$root" diff --binary --no-ext-diff HEAD >"$out/source.patch" || return 2
  {
    printf '%s\n' "$sha"
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum \
      "$root/quactlize/include/actlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp" \
      "$root/dev/fold_derivation/l231_q4_kpack4_production_fragment.cu" \
      "$root/dev/fold_derivation/l231_q4_kpack4_production_fragment.expected.txt" \
      "$root/tools/check_fq_q4k_kpack4_fragment_closure.py" \
      "$root/tools/run_fq_q4k_kpack4_fragment_closure_box.sh" \
      "$generated/manifest.json"
  } >"$out/source-authority.sha256" || return 2

  candidate="$(build_arm "$root" "$generated" "$out" candidate "" "$jobs")" || return $?

  candidate_log="$out/results/candidate.log"
  run_arm "$candidate" "$candidate_log" "$repeats"
  candidate_rc=$?
  python3 -B "$root/tools/check_fq_q4k_kpack4_fragment_closure.py" check \
    --candidate-log "$candidate_log" --candidate-rc "$candidate_rc" || return 2
  sha256sum "$candidate" "$candidate_log" \
    >"$out/results/authority.sha256" || return 2
  printf '[fq-kpack4-fragment] PASS sha=%s candidate=6/6 mapping=fixed artifacts=%s\n' \
    "$sha" "$out"
}

main "$@"
