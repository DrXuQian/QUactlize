#!/usr/bin/env bash
# Stable CI entry; build.sh remains the only public build entry.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/ci/run.py" "$@"
