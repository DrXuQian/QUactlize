#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$repo/dev/fold_derivation/run_l123_warp_nk_topology.sh"
bash "$repo/dev/fold_derivation/run_l138_wk_shadow_delivery.sh"
bash "$repo/dev/fold_derivation/run_l140_warpk_tactic_axis.sh"
bash "$repo/dev/fold_derivation/run_l139_marlin_warpk_reduce.sh"
bash "$repo/dev/fold_derivation/run_l142_twosource_consumer_compile.sh"
bash "$repo/dev/fold_derivation/run_l143_wk4_production_delivery.sh"
bash "$repo/dev/fold_derivation/run_l141_warpk_artifact.sh"
python3 "$repo/ci/check_dense_marlin_wk4_target.py"
bash -n "$repo/tools/run_dense_marlin_wk4_box.sh"

echo '[l143] PASS: target seam + exact type + shipping artifact/direct-pair consumer + CTA-local reduction; no device result claimed'
