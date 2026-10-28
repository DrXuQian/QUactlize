#!/usr/bin/env bash
# Public package build. Experiments live on experiments/ppu-tuning.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ "${1:-}" = "--library" ]; then
    shift
fi
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/tools/build_ppu_library.py" "$@"
