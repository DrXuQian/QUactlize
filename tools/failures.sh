#!/usr/bin/env bash
# WHAT TO PASTE WHEN SOMETHING IS RED. One command, no arguments, safe to run any time.
#
# run_batch writes one log per pass and per format under $OUT (default ~/ab). Reading them by hand means
# remembering which file, then which grep -- and today that cost two ppu001 round trips: once for an
# UnboundLocalError whose text was three lines above where anyone looked, and once for an assertion whose
# message was in the log all along.
#
# It prints, per log that has a failure: the summary line, the failed test ids, and every assertion line.
# Output is capped so the whole thing pastes into a chat.
set -uo pipefail
OUT="${OUT:-$HOME/ab}"
found=0
for log in "$OUT"/pytest_cpu_arm.log "$OUT"/pytest_full.log "$OUT"/fully_quantized_*.log \
           "$OUT"/dense_python_oracle.log "$OUT"/device_vs_cpu_arm.log; do
  [ -f "$log" ] || continue
  # the summary is pytest's last non-empty line
  summary=$(grep -E '[0-9]+ (passed|failed|error|skipped)' "$log" | tail -1)
  fails=$(sed -n '/^=* short test summary info/,$p' "$log" | grep -E '^(FAILED|ERROR)' | head -8)
  [ -n "$fails" ] || continue
  found=1
  echo "=== $(basename "$log")"
  echo "    $summary"
  echo "$fails" | sed 's/^/    /'
  # the assertion text itself: pytest prefixes it with "E ". Dedupe, since parametrised cases repeat it.
  grep -E '^E ' "$log" | sed 's/^E *//' | awk '!seen[$0]++' | head -12 | sed 's/^/      | /'
  echo
done
[ "$found" = 1 ] || echo "no failures in $OUT -- if you expected some, the run may not have reached that stage"
