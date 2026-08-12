#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 "$repo/ci/check_dense_marlin_wk4_target.py"
bash "$repo/dev/fold_derivation/run_l140_warpk_tactic_axis.sh"
bash "$repo/dev/fold_derivation/run_l141_warpk_artifact.sh"
bash "$repo/dev/fold_derivation/run_l139_marlin_warpk_reduce.sh"
bash -n "$repo/tools/run_dense_marlin_wk4_box.sh"

echo '[l143] PASS: target seam + exact type + WK4 artifact + CTA-local reduction; no device result claimed'
