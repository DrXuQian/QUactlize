#!/usr/bin/env bash
# ONE COMMAND, ALL FIVE FORMATS, one merged sample file.
#
# WHY THIS EXISTS. Linking all five formats into a single test_lowbit_moe_bench overflows the small code model:
#
#   relocation truncated to fit: R_X86_64_PC32 against `.tm_clone_table'
#   failed to convert GOTPCREL relocation against '_ZTVN7cutlass16device_exceptionE'; relink with --no-relax
#   /usr/bin/ld: final link failed
#
# 1338 shapes of device fatbin in one executable puts PC-relative relocations past their +-2GB reach. It first
# appeared on 2026-08-08 because that was the first run that actually asked for every format -- every earlier sweep
# passed MOE_FORMATS="i4" and carried a fifth of the binary.
#
# THE SPLIT IS IN THE LINK UNIT, NOT THE SEARCH SPACE. Every format still gets its whole grid; they just do not
# share an executable. That distinction is the whole point: "drop a format" would silently shrink what the sweep can
# find, and this does not.
#
# It is deliberately NOT the fix for the code model. Raising -mcmodel, or moving the fatbin out of the small model,
# may well be right and is being measured separately -- that needs `size -A` on the failing object, not a guess.
# This script is what works today and it keeps working whichever way that lands.
#
#   bash benchmarks/sweep_all_formats.sh                          # prefill band, the C1 shape
#   bash benchmarks/sweep_all_formats.sh --decode                 # M=1 band, the D4 shape
#   MOE_REPS=3 bash benchmarks/sweep_all_formats.sh               # 3 passes, so ties can be resolved
#   FORMATS="Q3_K Q5_K" bash benchmarks/sweep_all_formats.sh      # a subset, when you mean to
#
# Samples from all formats append to ONE jsonl, so `python3 benchmarks/analyse.py <that file> --coverage` sees the
# whole sweep. That works because bench_samples.hpp opens BENCH_JSONL in append mode and every record carries its
# own schema field -- the merge needs no post-processing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORMATS="${FORMATS:-i4 i2 Q3_K Q5_K Q6_K}"
CORES="${MOE_CORES:-192}"
JOBS="${JOBS:-$(nproc)}"
OUT="${BENCH_JSONL:-$ROOT/sweep_all_formats.jsonl}"

TARGET=test_lowbit_moe_bench
ARGS=(256 4096 512 2048 32 4 8)          # C1: 256 experts, 4096 tokens, N=512 K=2048, gs=32, pinned router, top-8
BAND=prefill
if [ "${1:-}" = "--decode" ]; then
  TARGET=test_lowbit_moe_decode_bench
  ARGS=(64 8 2048 2048 32 3)             # D4 band: 64 experts, top-8, N=K=2048, gs=32, decode mode
  BAND=decode
  shift
fi

echo "[sweep-all] band=$BAND target=$TARGET formats: $FORMATS"
echo "[sweep-all] samples -> $OUT   (appended; delete it first for a clean run)"

ok=(); failed=()
for F in $FORMATS; do
  DIR="$ROOT/build_moe_${BAND}_${F}"
  echo
  echo "===================== $F ====================="
  # PER-FORMAT BUILD DIRECTORY, not a shared one. cmake caches MOE_FORMATS, and a shared dir would either reuse the
  # previous format's units or force a full reconfigure -- the first is silently wrong, the second is slow.
  if ! PPU_BUILD_DIR="$DIR" MOE_FORMATS="$F" MOE_CORES="$CORES" JOBS="$JOBS" \
       TARGET="$TARGET" ./build.sh; then
    echo "[sweep-all] BUILD FAILED for $F -- continuing so the other formats still produce data"
    failed+=("$F(build)")
    continue
  fi

  BIN="$(find "$DIR" -name "$TARGET" -type f -perm -u+x | head -1)"
  if [ -z "$BIN" ]; then
    echo "[sweep-all] built but no $TARGET under $DIR -- treating as a failure rather than skipping quietly"
    failed+=("$F(missing-bin)")
    continue
  fi

  # A ROW THAT DID NOT RUN MUST NOT LOOK LIKE A ROW THAT LOST. The bench exits non-zero when a MOE_ONLY filter
  # matches nothing; record that here rather than letting the loop swallow it.
  if BENCH_JSONL="$OUT" "$BIN" "${ARGS[@]}"; then ok+=("$F"); else failed+=("$F(run)"); fi
done

echo
echo "[sweep-all] ran: ${ok[*]:-none}"
if [ ${#failed[@]} -gt 0 ]; then
  echo "[sweep-all] FAILED: ${failed[*]}"
  echo "[sweep-all] the samples above are still valid for the formats that ran -- they are not a whole sweep."
  exit 1
fi
echo "[sweep-all] next: python3 benchmarks/analyse.py $OUT --coverage"
