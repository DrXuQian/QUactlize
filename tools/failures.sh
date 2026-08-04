#!/usr/bin/env bash
# WHAT TO PASTE AFTER A BOX RUN, red or green. No arguments, safe any time.
#
# It prints, for EVERY log run_batch wrote: the summary line, and -- when there are failures -- the failed ids
# and every assertion line.
#
# TWO REASONS IT PRINTS THE GREEN ONES TOO.
#   * "no failures" and "no logs" are different states, and the first version of this script said the same thing
#     for both: it printed a single line and left the reader to wonder whether the run had even reached that
#     stage. Listing what it FOUND makes the distinction visible instead of hinted at.
#   * a green run is not the end of the story here. codex holds nine matrix promotions on reading the per-format
#     pass summaries, so the numbers are the deliverable whether or not anything is red.
set -uo pipefail
OUT="${OUT:-$HOME/ab}"
logs=("$OUT"/pytest_cpu_arm.log "$OUT"/pytest_full.log "$OUT"/fully_quantized_*.log \
      "$OUT"/dense_python_oracle.log "$OUT"/device_vs_cpu_arm.log)
seen=0 red=0
for log in "${logs[@]}"; do
  [ -f "$log" ] || continue
  seen=$((seen+1))
  # HOW OLD, not just what it says. A log from a previous run reads exactly like a fresh one, and that cost a
  # round of "all green" that was green for a checkout two commits back -- five per-format tests instead of six,
  # with the newest oracle simply not present. Comparing against HEAD's commit time is the cheapest thing that
  # can tell the two apart.
  age=""
  if head_t=$(git -C "$(dirname "$0")/.." log -1 --format=%ct 2>/dev/null); then
    log_t=$(stat -c %Y "$log" 2>/dev/null || echo 0)
    [ "$log_t" -lt "$head_t" ] && age="  [STALE: written before the current HEAD commit]"
  fi
  summary=$(grep -E '[0-9]+ (passed|failed|error|skipped)' "$log" | tail -1)
  fails=$(sed -n '/^=* short test summary info/,$p' "$log" | grep -E '^(FAILED|ERROR)' | head -8)
  if [ -n "$fails" ]; then
    red=$((red+1))
    echo "=== $(basename "$log")   [FAILURES]$age"
    echo "    ${summary:-<no summary line>}"
    echo "$fails" | sed 's/^/    /'
    grep -E '^E ' "$log" | sed 's/^E *//' | awk '!seen[$0]++' | head -12 | sed 's/^/      | /'
  else
    echo "=== $(basename "$log")$age"
    echo "    ${summary:-<no summary line -- the run may not have reached this stage>}"
  fi
  echo
done
if [ "$seen" = 0 ]; then
  echo "NO LOGS FOUND under $OUT."
  echo "  That is not the same as 'nothing failed' -- it means run_batch has not written here."
  echo "  Check OUT (currently $OUT), or that ./benchmarks/run_batch.sh pytest actually ran."
else
  echo "$seen log(s) read, $red with failures."
grep -q STALE <<<"$(true)" 2>/dev/null || true
echo "Any line marked STALE predates the current checkout -- re-run before reading anything into it."
fi
