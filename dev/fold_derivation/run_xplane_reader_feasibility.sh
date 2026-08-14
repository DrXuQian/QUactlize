#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
work_root=${XPLANE_READER_WORK_ROOT:-/workspace/quactlize-xplane-reader-feasibility}
compiler_tmp="$work_root/compiler-tmp"
mkdir -p "$work_root" "$compiler_tmp"

# Independent anchor: the production xplane writer and ArrangementSlotPermutation
# must agree before the feasibility model is allowed to classify any word window.
TMPDIR="$compiler_tmp" bash "$repo_dir/dev/fold_derivation/run_l137_bc_arrangement_layout.sh" \
  > "$work_root/l137-production-writer-anchor.log"

python3 "$repo_dir/dev/fold_derivation/xplane_reader_feasibility.py" \
  --json "$work_root/feasibility.json" \
  --markdown "$work_root/XPLANE_READER_FEASIBILITY.generated.md"

for plant in wrong-permutation-bit missing-denominator; do
  plant_log="$work_root/plant-$plant.log"
  if python3 "$repo_dir/dev/fold_derivation/xplane_reader_feasibility.py" \
      --plant "$plant" > "$plant_log" 2>&1; then
    echo "[xplane-reader] FAIL: plant $plant escaped" >&2
    exit 1
  fi
  grep -q 'PLANTED_RED' "$plant_log" || {
    echo "[xplane-reader] FAIL: plant $plant failed for an unrelated reason" >&2
    exit 1
  }
done

cmp "$repo_dir/dev/fold_derivation/XPLANE_READER_FEASIBILITY.md" \
    "$work_root/XPLANE_READER_FEASIBILITY.generated.md"

echo "[xplane-reader] PASS: production-writer anchor + exhaustive census + 2/2 planted RED controls"
echo "[xplane-reader] artifacts: $work_root"
