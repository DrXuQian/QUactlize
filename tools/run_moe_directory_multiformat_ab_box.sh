#!/usr/bin/env bash
# Run the directory-scheduler A/B over every shipping GGUF-K ScaleFirst layout.
#
# This is deliberately NOT a FullyQuantized/packed-metadata benchmark.  The
# weight planes use the real resident xplane ArtifactTileK for each format, and
# metadata stays in the fp16 ScaleFirst ABI.  A packed-unit runner must supply
# native units and ExpectPackedScale=true; calling this path "packed" would
# make two incompatible offline layouts share one result label.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/workspace/quactlize-moe-directory-multiformat-${SHA:0:8}-${STAMP}}"
JOBS="${JOBS:-8}"
REPS="${MOE_REPS:-3}"
mkdir -p "$OUT"

printf '[moe-directory-multiformat] sha=%s out=%s scope=ScaleFirst reps=%s\n' \
  "$SHA" "$OUT" "$REPS"
printf 'format\tshort\tqtype\tquant\tgs\tbits\tplanes\tartifact_tile_k\tweight_layout\tmetadata_layout\tbaseline_us\tpersistent_us\tspeedup\n' \
  >"$OUT/summary.tsv"

# format|short|qtype|qmode|quant|gs|bits|planes|ArtifactTileK|row
# qmode is semantic, not a tuning knob: Q2/Q4/Q5 carry scale+zero while
# signed Q3/Q6 carry scale only.
mapfile -t CASES <<'EOF'
Q2_K|i2|10|0|ScaleZero|16|2+0|1|128|i2 64x64:128 w64x32 s3 bc0->0
Q3_K|q3|11|1|ScaleOnly|16|2+1|2|256|q3 64x64:256 w64x64 s3 bc0->0
Q4_K|i4|12|0|ScaleZero|32|4+0|1|64|i4 64x64:64 w64x32 s3 bc0->0
Q5_K|q5|13|0|ScaleZero|32|4+1|2|256|q5 64x64:256 w64x64 s3 bc0->0
Q6_K|q6|14|1|ScaleOnly|16|4+2|2|128|q6 64x64:128 w64x32 s3 bc0->0
EOF

time_from_result() {
  awk '{ for (i = 1; i <= NF; ++i) if ($i == "us") { print $(i-1); exit } }' "$1"
}

for record in "${CASES[@]}"; do
  IFS='|' read -r name short qtype qmode quant gs bits planes artifact_tk row <<<"$record"
  case_out="$OUT/${name,,}-scalefirst-a${artifact_tk}"
  printf '[moe-directory-multiformat] begin format=%s short=%s qtype=%s quant=%s gs=%s A=%s planes=%s\n' \
    "$name" "$short" "$qtype" "$quant" "$gs" "$artifact_tk" "$planes"
  OUT="$case_out" JOBS="$JOBS" MOE_REPS="$REPS" \
    MOE_FORMAT="$short" MOE_QMODE="$qmode" \
    MOE_ARGS="64 128 2048 2048 $gs 2 8" MOE_ONLY="$row" \
    bash "$ROOT/tools/run_moe_directory_persistent_ab_box.sh" \
    >"$OUT/${name,,}.log" 2>&1 || {
      tail -160 "$OUT/${name,,}.log" >&2
      printf '[moe-directory-multiformat] FAIL: %s A/B failed; artifacts=%s\n' "$name" "$OUT" >&2
      exit 1
    }

  baseline_us="$(time_from_result "$case_out/nonpersistent/result-line.txt")"
  persistent_us="$(time_from_result "$case_out/persistent_directory/result-line.txt")"
  if [ -z "$baseline_us" ] || [ -z "$persistent_us" ]; then
    printf '[moe-directory-multiformat] FAIL: %s result line has no primary event time\n' "$name" >&2
    exit 1
  fi
  speedup="$(awk -v a="$baseline_us" -v b="$persistent_us" 'BEGIN { if (a <= 0 || b <= 0) exit 2; printf "%.6f", a / b }')" || {
    printf '[moe-directory-multiformat] FAIL: %s invalid timing pair %s/%s\n' \
      "$name" "$baseline_us" "$persistent_us" >&2
    exit 1
  }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$short" "$qtype" "$quant" "$gs" "$bits" "$planes" "$artifact_tk" \
    "xplane" "fp16-${quant}" "$baseline_us" "$persistent_us" "$speedup" \
    >>"$OUT/summary.tsv"
  printf '[moe-directory-multiformat] result format=%s baseline=%s_us persistent=%s_us speedup=%sx raw-bit=PASS\n' \
    "$name" "$baseline_us" "$persistent_us" "$speedup"
done

rows="$(( $(wc -l <"$OUT/summary.tsv") - 1 ))"
if [ "$rows" -ne 5 ] || [ "$(cut -f3 "$OUT/summary.tsv" | tail -n +2 | sort -u | wc -l)" -ne 5 ]; then
  printf '[moe-directory-multiformat] FAIL: expected five distinct shipping qtypes, got %s\n' "$rows" >&2
  exit 1
fi
printf '[moe-directory-multiformat] PASS: five ScaleFirst formats; each A/B is full-D raw-bit gated\n'
cat "$OUT/summary.tsv"
printf '[moe-directory-multiformat] artifacts: %s\n' "$OUT"
