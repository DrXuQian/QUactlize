#!/usr/bin/env bash
# Source builds only. Invoke this script; do not source it into a login shell.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/tools/build.py" "$@"
