#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "${repo_root}/dev/fold_derivation/l167_classic_marlin_format.py" "$@"
