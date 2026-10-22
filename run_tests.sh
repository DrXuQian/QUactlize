#!/usr/bin/env bash
# CI tests: --suite test --package <built-package> (host + PPU accuracy, no rebuild).
# build.sh --library remains the public package build entry.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/ci/run.py" "$@"
