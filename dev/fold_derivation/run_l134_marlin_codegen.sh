#!/usr/bin/env bash
# Historical filename retained because ci/local_gates.py invokes it.  The old
# source compiled Cfg::MarlinGemm from the generic collective; the standalone
# generated-unit and production scheduler are now the only accepted binding.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "$repo/dev/fold_derivation/run_l169_standalone_marlin_unit.sh"
bash "$repo/dev/fold_derivation/run_l170_standalone_marlin_scheduler.sh"
python3 "$repo/ci/check_dense_marlin_wk4_target.py"

echo '[l134] PASS: standalone generated unit instantiates MarlinCollectivePPU/MarlinSchedulerPPU/MarlinKernelPPU and production descriptor oracle closes; retired generic Cfg PTX is not evidence'
