#!/usr/bin/env bash
# Compatibility entry point for the former plan-only FullyQuantized matrix.
#
# The matrix now has a real generated device runner.  Keeping a separate
# plan-only command would create a second, weaker publication path, so this
# wrapper deliberately delegates to the one source of runtime truth.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
printf '%s\n' \
  '[fq-internal-matrix] device runner is authoritative; delegating without changing arguments'
exec bash "$root/tools/run_fully_quantized_internal_sweep_box.sh" "$@"
