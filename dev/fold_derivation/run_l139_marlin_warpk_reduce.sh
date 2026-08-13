#!/usr/bin/env bash
# Historical filename retained because ci/local_gates.py invokes it.  The old
# TYPE_ONLY arm derived its fragments from the generic M/N-only builder and is
# no longer the shipping Marlin reduction.  L168 binds the standalone 2N x 4K
# trace and rejects a flat-reduction plant against independent classic/Awesome
# source anchors.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$repo/dev/fold_derivation/run_l168_marlin_pipeline_trace.sh"
bash "$repo/dev/fold_derivation/run_l139_standalone_fragment_oracle.sh"
python3 "$repo/ci/check_dense_marlin_wk4_target.py"

echo '[L139] PASS: production TiledMma exhausts 256x32 ownership and raw 4->2->1 scratch cadence; classic/Awesome trace and eleven standalone stack plants close; generic-builder TYPE_ONLY proof retired'
