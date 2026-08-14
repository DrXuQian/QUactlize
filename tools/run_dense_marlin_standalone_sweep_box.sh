#!/usr/bin/env bash
# Build and run the production standalone-Marlin sweep.  The committed tactic
# authority owns every active TM/TN/TK/WN/WarpK/stage axis; this runner refuses
# a binary whose compiled rows differ from that exact authority.  It is
# deliberately not the retired generic DENSE_MARLIN_SWEEP target.
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

# Compare the compiled registry with the committed X-macro row-for-row.  Do
# not restate a row count or a geometry here: doing so would turn a newly
# admitted TN/TK/WN/WarpK value into a silently unmeasured axis.
python3 - "$ROOT/benchmarks/marlin_standalone_configs.inc" \
  "$OUT/list.log" "$OUT/axis-manifest.txt" <<'PY'
import re
import sys
from pathlib import Path

authority = Path(sys.argv[1]).read_text()
listing = Path(sys.argv[2]).read_text()
manifest = Path(sys.argv[3])
row_re = re.compile(
    r"^\s*X\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),CP_ASYNC,B\)",
    re.MULTILINE,
)
list_re = re.compile(
    r"tile\s+(\d+)x(\d+)x(\d+)\s+warp\s+(\d+)x(\d+)x(\d+)\s+stages\s+(\d+)"
)
expected_rows = [tuple(map(int, row)) for row in row_re.findall(authority)]
observed_rows = [tuple(map(int, row)) for row in list_re.findall(listing)]
expected = set(expected_rows)
observed = set(observed_rows)
declared = re.search(r"#define\s+MARLIN_STANDALONE_CFG_ROWS\s+(\d+)", authority)
if (not declared or int(declared.group(1)) != len(expected_rows) or
        len(expected_rows) != len(expected)):
    raise SystemExit("authority count does not equal its unique X-macro rows")
if len(observed_rows) != len(observed):
    raise SystemExit("compiled config listing contains duplicate rows")
if expected != observed:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    raise SystemExit(
        f"compiled rows differ from authority: missing={missing[:8]} extra={extra[:8]}"
    )
names = ("TM", "TN", "TK", "WM", "WN", "WarpK", "Stages")
lines = [f"rows={len(expected)}"]
for i, name in enumerate(names):
    lines.append(f"{name}=" + ",".join(map(str, sorted({row[i] for row in expected}))))
manifest.write_text("\n".join(lines) + "\n")
print("[marlin-standalone-sweep] " + " ".join(lines))
PY

# Run each occupancy point as its own experiment.  In particular, never append
# BPC2 records to the BPC1 JSONL: a downstream analyser must not be able to
# average two scheduler grids under one run identity.
#
# An exact kernel may have a lower occupancy cap than another row in this same
# binary.  run_config() rejects that row before launch.  This runner turns that
# rejection into an explicit NOT RUN while allowing the remaining rows to
# produce a ranking.  If every row is over-cap, the occupancy point itself is
# NOT RUN rather than a failed sweep.
run_sweep() {
  local bpc="$1" label="bpc${1}"
  local samples="$OUT/${label}.samples.jsonl"
  local log="$OUT/${label}.log"
  local verdict="$OUT/${label}.verdict"
  local -a sweep=("$BIN" --search_configs --streamk_exact_fixture
                  "--marlin-blocks-per-cu=$bpc"
                  --m=1 --n=4096 --k=4096 --l=1 --g=128 --mode=1
                  --alpha=1 --beta=0 --iterations=20)

  printf 'BENCH_REPS=%q BENCH_JSONL=%q ' "$REPS" "$samples" >"$OUT/${label}.command"
  printf '%q ' "${sweep[@]}" >>"$OUT/${label}.command"
  printf '\n' >>"$OUT/${label}.command"

  set +e
  BENCH_REPS="$REPS" BENCH_JSONL="$samples" \
    "${sweep[@]}" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e

  [ -s "$samples" ] || fail "BPC${bpc} did not emit its durable JSONL records"
  if grep -Fq 'Disposition: Failed' "$log"; then
    fail "BPC${bpc} has an exact-fixture numerical failure"
  fi

  local over_cap
  over_cap="$(grep -c -- "--marlin-blocks-per-cu=${bpc} is outside the exact kernel occupancy range" "$log" || true)"

  if [ "$rc" -ne 0 ]; then
    if [ "$bpc" -gt 1 ] && [ "$over_cap" -gt 0 ] && grep -Fq 'no config passed' "$log"; then
      printf 'NOT RUN: BPC%d exceeds every admitted kernel occupancy cap (%d rejected attempts)\n' \
        "$bpc" "$over_cap" | tee "$verdict"
      printf '[marlin-standalone-sweep] NOT RUN: BPC%d has no reachable row; continuing bundle closure\n' \
        "$bpc"
      return 0
    fi
    fail "BPC${bpc} sweep returned rc=${rc} for a reason other than an exact occupancy rejection"
  fi

  grep -Fq "blocks_per_cu=${bpc}" "$log" || \
    fail "BPC${bpc} did not reach any scheduler-owned decomposition"
  grep -E '^==== (WINNER|UNRESOLVED):' "$log" >"$verdict" || \
    fail "BPC${bpc} did not produce a ranking verdict"
  [ "$(wc -l <"$verdict")" -eq 1 ] || \
    fail "BPC${bpc} produced more than one ranking verdict"

  if [ "$over_cap" -gt 0 ]; then
    printf '[marlin-standalone-sweep] NOT RUN: BPC%d rejected %d per-kernel attempt(s) above their exact occupancy cap; reachable rows were ranked\n' \
      "$bpc" "$over_cap" | tee -a "$verdict"
  fi
}

# BPC1 is the byte-compatible default value and remains the control. BPC2 is
# the first occupancy cross point requested by the user; higher BPC values are
# deliberately not folded into this config comparison.
run_sweep 1
run_sweep 2

{
  printf 'root_sha=%s\n' "$ROOT_SHA"
  printf 'target=%s\n' "$TARGET"
  printf 'shape=M1,N4096,K4096,L1,gs128\n'
  printf 'scope=all-rows-in-committed-standalone-authority\n'
  sed 's/^/axis_/' "$OUT/axis-manifest.txt"
  printf 'blocks_per_cu=1,2\nrepetitions=%s\n' "$REPS"
  printf 'binary=%s\n' "$BIN"
} >"$OUT/manifest.txt"
sha256sum "$OUT"/build.command "$OUT"/build.log "$OUT"/binary.sha256 \
  "$OUT"/list.command "$OUT"/list.log \
  "$OUT"/axis-manifest.txt \
  "$OUT"/bpc1.command "$OUT"/bpc1.log "$OUT"/bpc1.samples.jsonl "$OUT"/bpc1.verdict \
  "$OUT"/bpc2.command "$OUT"/bpc2.log "$OUT"/bpc2.samples.jsonl "$OUT"/bpc2.verdict \
  "$OUT"/manifest.txt >"$OUT/bundle.sha256"

printf '[marlin-standalone-sweep] PASS: BPC1/BPC2 exact-fixture cross sweep completed (over-cap rows remain explicit NOT RUN)\n'
printf '[marlin-standalone-sweep] sha=%s binary=%s\n' "$ROOT_SHA" "$BIN"
printf '[marlin-standalone-sweep] artifacts=%s\n' "$OUT"
