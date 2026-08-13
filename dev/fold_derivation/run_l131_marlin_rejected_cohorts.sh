#!/usr/bin/env bash
# Historical filename retained because ci/local_gates.py invokes it.  L131's
# old 1/8/16/32-warp witnesses instantiate the retired generic Marlin wrapper;
# they must not be used as evidence for the standalone stack.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"

python3 "$repo/ci/check_dense_marlin_rejection_census.py"

echo '[L131] PASS: standalone tactic census closes 60000 rows with one proved admission; historical generic A2 cohort compilation is PRE_STANDALONE_NOT_EVIDENCE'
