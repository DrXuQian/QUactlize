#!/usr/bin/env bash
# Device-free exhaustive oracle for the committed dense tail-only Stream-K domain.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORACLE="$ROOT/dev/fold_derivation/l201_streamk_tail_oracle.py"

if [ ! -f "$ORACLE" ]; then
  printf '[l201-runner] FAIL: missing oracle %s\n' "$ORACLE" >&2
  exit 1
fi

PYTHONDONTWRITEBYTECODE=1 exec python3 "$ORACLE"
