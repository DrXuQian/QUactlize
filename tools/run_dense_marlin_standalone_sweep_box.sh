#!/usr/bin/env bash
# Build and run the first production standalone-Marlin sweep.  This is the
# admitted m8/m16 pair at fixed TN128/TK128/WN64/WarpK32; it is deliberately
# not the retired generic DENSE_MARLIN_SWEEP target.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_lowbit_dense_marlin_standalone_sweep
ROOT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
SHORT_SHA="$(git -C "$ROOT" rev-parse --short=12 HEAD)"
OUT="${MARLIN_STANDALONE_SWEEP_OUT:-/workspace/quactlize-dense-marlin-standalone-sweep-${SHORT_SHA}}"
REPS="${BENCH_REPS:-5}"

fail() {
  printf '[marlin-standalone-sweep] FAIL: %s\n' "$*" >&2
  return 1
}

[ "$#" -eq 0 ] || fail 'this runner accepts no positional arguments'
if [ -e "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  fail "output directory is not empty: $OUT (choose MARLIN_STANDALONE_SWEEP_OUT for a new run)"
fi
mkdir -p "$OUT/build"

git -C "$ROOT" status --porcelain=v1 --untracked-files=all >"$OUT/root-status.txt"
[ ! -s "$OUT/root-status.txt" ] || fail 'source tree is not clean; a sweep must name one exact SHA'
git -C "$ROOT" submodule status --recursive >"$OUT/submodule-status.txt"
if grep -Eq '^[+\-U]' "$OUT/submodule-status.txt"; then
  fail 'a submodule checkout differs from its recorded gitlink'
fi

BUILD=(env PPU_BUILD_DIR="$OUT/build" PPU_ARCHS=ppu0010 TARGET="$TARGET"
       QUANT=int4 BENCH_GS=128 "$ROOT/build.sh")
printf '%q ' "${BUILD[@]}" >"$OUT/build.command"
printf '\n' >>"$OUT/build.command"
"${BUILD[@]}" 2>&1 | tee "$OUT/build.log"

mapfile -t bins < <(find "$OUT/build" -type f -name "$TARGET" -perm -u+x -print)
[ "${#bins[@]}" -eq 1 ] || fail "expected exactly one $TARGET binary, found ${#bins[@]}"
BIN="${bins[0]}"
sha256sum "$BIN" >"$OUT/binary.sha256"

LIST=("$BIN" --list_configs)
printf '%q ' "${LIST[@]}" >"$OUT/list.command"
printf '\n' >>"$OUT/list.command"
"${LIST[@]}" 2>&1 | tee "$OUT/list.log"
grep -Fq 'scheduler=standalone-marlin' "$OUT/list.log" || \
  fail 'binary did not identify the standalone-Marlin table'
grep -Fq 'warp 8x64x32' "$OUT/list.log" || fail 'm8/WarpK32 row is missing'
grep -Fq 'warp 16x64x32' "$OUT/list.log" || fail 'm16/WarpK32 row is missing'

# BPC1 is both the first measurement and the byte-compatible default path.
# Spell it explicitly in the command so the result cannot later be mistaken
# for a different occupancy sweep point.
SAMPLES="$OUT/bpc1.samples.jsonl"
SWEEP=("$BIN" --search_configs --streamk_exact_fixture
       --marlin-blocks-per-cu=1
       --m=1 --n=4096 --k=4096 --l=1 --g=128 --mode=1
       --alpha=1 --beta=0 --iterations=20)
printf 'BENCH_REPS=%q BENCH_JSONL=%q ' "$REPS" "$SAMPLES" >"$OUT/bpc1.command"
printf '%q ' "${SWEEP[@]}" >>"$OUT/bpc1.command"
printf '\n' >>"$OUT/bpc1.command"
BENCH_REPS="$REPS" BENCH_JSONL="$SAMPLES" \
  "${SWEEP[@]}" 2>&1 | tee "$OUT/bpc1.log"

grep -Fq 'blocks_per_cu=1' "$OUT/bpc1.log" || \
  fail 'BPC1 did not reach the scheduler-owned decomposition'
grep -Eq '^==== (WINNER|UNRESOLVED):' "$OUT/bpc1.log" || \
  fail 'the repeated sweep did not produce a ranking verdict'
if grep -Fq 'Disposition: Failed' "$OUT/bpc1.log"; then
  fail 'at least one exact-fixture candidate failed numerical verification'
fi
[ -s "$SAMPLES" ] || fail 'the sweep did not emit its durable JSONL samples'

{
  printf 'root_sha=%s\n' "$ROOT_SHA"
  printf 'target=%s\n' "$TARGET"
  printf 'shape=M1,N4096,K4096,L1,gs128\n'
  printf 'scope=admitted-m8-m16,TN128,TK128,WN64,WarpK32,S4\n'
  printf 'blocks_per_cu=1\nrepetitions=%s\n' "$REPS"
  printf 'binary=%s\n' "$BIN"
} >"$OUT/manifest.txt"
sha256sum "$OUT"/build.command "$OUT"/build.log "$OUT"/binary.sha256 \
  "$OUT"/list.command "$OUT"/list.log "$OUT"/bpc1.command \
  "$OUT"/bpc1.log "$SAMPLES" "$OUT"/manifest.txt >"$OUT/bundle.sha256"

printf '[marlin-standalone-sweep] PASS: BPC1 exact-fixture sweep completed\n'
printf '[marlin-standalone-sweep] sha=%s binary=%s\n' "$ROOT_SHA" "$BIN"
printf '[marlin-standalone-sweep] artifacts=%s\n' "$OUT"
