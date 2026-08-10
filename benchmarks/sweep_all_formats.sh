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
#   FORMATS="q3 q5" bash benchmarks/sweep_all_formats.sh          # a subset, when you mean to
#   SWEEP_DIR=/tmp/sweep bash benchmarks/sweep_all_formats.sh     # somewhere other than ./sweep
#
# THE OUTPUT FILE IS FRESH AND SELF-DESCRIBING, and both halves of that are load-bearing.
#
# FRESH, because bench_samples.hpp opens BENCH_JSONL in APPEND mode. Reusing one path across days accumulates
# several builds in one file; analyse.py then correctly refuses to rank it ("a verdict over two libraries
# describes neither") and an hour of sweep is unreadable. Observed 2026-08-09 on a file holding four MoE builds
# and a dense one, plus records from before `bc` was a row field. So each run gets its own file and this script
# refuses to write into one that exists.
#
# SELF-DESCRIBING, because the name is derived from the ARGUMENTS rather than typed. The same run was being
# written to `/tmp/sweep/grouped_L1.jsonl` -- a name from a different experiment (BOX.md`s L=1 grouped-as-dense
# control) while the run was actually L=256 ragged. Nothing caught it: the records carry their own fixture
# identity so analyse.py grouped them correctly, but every human reading the path was told the wrong thing, and
# BOX.md has an --invariant command that pairs that exact filename against a dense run. A name that cannot
# contradict the run is the only version of this that stays true.
#
# All formats of ONE run share ONE file -- that merge is intended and needs no post-processing, because every
# record carries its own schema field.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# These are the CMake MOE_FORMATS keys, not the JSON schema labels Q3_K/Q5_K/Q6_K.  Passing the latter gets as far
# as configure and then (correctly) matches no format row; using the public keys here is what makes the default
# command actually build all five shards.
# `${VAR-default}`, NOT `${VAR:-default}`. The colon form substitutes when VAR is unset OR empty, so an explicit
# FORMATS="" would silently become all five and the empty-list guard below could never fire. I wrote the guard with
# the colon form still in place, tested it, watched all five formats build instead, and had already claimed in a
# commit message that the guard was verified. Unset means "give me everything"; empty means "I said none", and
# those are different requests.
FORMATS="${FORMATS-q3 q5 q6 i2 i4}"
CORES="${MOE_CORES:-192}"
JOBS="${JOBS:-$(nproc)}"
TARGET=test_lowbit_moe_bench
ARGS=(256 4096 512 2048 32 4 8)          # C1: 256 experts, 4096 tokens, N=512 K=2048, gs=32, pinned router, top-8
BAND=prefill
if [ "${1:-}" = "--decode" ]; then
  TARGET=test_lowbit_moe_decode_bench
  ARGS=(64 8 2048 2048 32 3)             # D4 band: 64 experts, top-8, N=K=2048, gs=32, decode mode
  BAND=decode
  shift
fi

# NAME DERIVED FROM ARGS, never typed. ARGS is [L rows N K gs mode topk], so the path states the run.
OUT_DIR="${SWEEP_DIR:-$ROOT/sweep}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${BAND}_L${ARGS[0]}_r${ARGS[1]}_n${ARGS[2]}_k${ARGS[3]}_gs${ARGS[4]}_${STAMP}"
OUT="${BENCH_JSONL:-$OUT_DIR/$RUN_ID.jsonl}"
mkdir -p "$(dirname "$OUT")"

# REFUSE TO APPEND. bench_samples.hpp opens in append mode, so an existing file would silently gain a second
# build and analyse.py would then refuse to rank the whole thing -- after the sweep, not before it.
if [ -e "$OUT" ]; then
  echo "[sweep-all] $OUT already exists. This script never appends: a file holding two builds cannot be ranked." >&2
  echo "[sweep-all] Move it aside, or pass a different BENCH_JSONL." >&2
  exit 2
fi

echo "[sweep-all] band=$BAND target=$TARGET formats: $FORMATS"
echo "[sweep-all] args: L=${ARGS[0]} rows=${ARGS[1]} N=${ARGS[2]} K=${ARGS[3]} gs=${ARGS[4]} mode=${ARGS[5]} topk=${ARGS[6]:-}"
echo "[sweep-all] samples -> $OUT   (fresh; this run only)"

# AN EMPTY FORMAT LIST IS A MISTAKE, NOT A NO-OP. Without this the loop runs zero times, prints "ran: none" and
# exits 0 -- a sweep that measured nothing reporting success, which is the exact shape this script exists to stop
# on the other side (partial data must not read as a whole sweep).
if [ -z "${FORMATS// }" ]; then
  echo "[sweep-all] FORMATS is empty -- nothing to sweep. Unset it for all five, or name the ones you mean." >&2
  exit 2
fi

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

  BIN="$(find "$DIR" -name "$TARGET" -type f -perm -u+x -print -quit)"
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
