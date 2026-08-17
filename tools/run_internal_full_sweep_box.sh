#!/usr/bin/env bash
# Run the two independent internal-sweep components, then publish three
# fail-closed leaderboards.  This orchestration layer never builds or selects a
# shipping kernel itself; each component runner owns its exact finite graph.
set -uo pipefail

main() {
  if [ "$#" -ne 0 ]; then
    printf '[internal-full-sweep] FAIL: this runner accepts no positional arguments\n' >&2
    return 2
  fi

  local root workspace_root sha short stamp out
  local scale_runner fq_runner scale_summary_rel fq_summary_rel
  local scale_out fq_out scale_log fq_log scale_rc fq_rc
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace_root="$(realpath -e /workspace)" || {
    printf '[internal-full-sweep] FAIL: /workspace is unavailable\n' >&2
    return 2
  }
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-internal-full-sweep-${short}-${stamp}}")" || return 2
  case "$out" in
    "$workspace_root"/*) ;;
    *) printf '[internal-full-sweep] FAIL: OUT must be a strict /workspace child: %s\n' "$out" >&2; return 2 ;;
  esac
  if [ -e "$out" ]; then
    printf '[internal-full-sweep] FAIL: refusing to overwrite %s\n' "$out" >&2
    return 2
  fi

  scale_runner="${SCALEFIRST_RUNNER:-$root/tools/run_scalefirst_internal_sweep_box.sh}"
  fq_runner="${FULLY_QUANTIZED_RUNNER:-$root/tools/run_fully_quantized_internal_sweep_box.sh}"
  # During development the FQ component used this name.  The fallback remains
  # fail-closed: its current plan-only summary cannot satisfy the merge schema.
  if [ ! -f "$fq_runner" ] && [ -z "${FULLY_QUANTIZED_RUNNER:-}" ]; then
    fq_runner="$root/tools/run_fully_quantized_internal_matrix.sh"
  fi
  for runner in "$scale_runner" "$fq_runner"; do
    if [ ! -f "$runner" ]; then
      printf '[internal-full-sweep] FAIL: component runner not found: %s\n' "$runner" >&2
      return 2
    fi
  done

  scale_summary_rel="${SCALEFIRST_SUMMARY_REL:-results/summary.json}"
  fq_summary_rel="${FULLY_QUANTIZED_SUMMARY_REL:-results/summary.json}"
  case "$scale_summary_rel:$fq_summary_rel" in
    *..*|*//*|/*) printf '[internal-full-sweep] FAIL: summary paths must be simple relative children\n' >&2; return 2 ;;
  esac

  mkdir -p "$out" || return 2
  scale_out="$out/scale-first"
  fq_out="$out/fully-quantized"
  scale_log="$out/scale-first.runner.log"
  fq_log="$out/fully-quantized.runner.log"
  {
    printf 'schema=quactlize.internal_full_sweep.run.v1\n'
    printf 'root_sha=%s\n' "$sha"
    printf 'actlize_sha=%s\n' "$(git -C "$root/third_party/actlize" rev-parse HEAD)"
    printf 'created_utc=%s\n' "$stamp"
    printf 'scale_first_runner=%s\n' "$scale_runner"
    printf 'scale_first_runner_sha256=%s\n' "$(sha256sum "$scale_runner" | awk '{print $1}')"
    printf 'fully_quantized_runner=%s\n' "$fq_runner"
    printf 'fully_quantized_runner_sha256=%s\n' "$(sha256sum "$fq_runner" | awk '{print $1}')"
    printf 'merger_sha256=%s\n' "$(sha256sum "$root/tools/merge_internal_full_sweep.py" | awk '{print $1}')"
    printf 'gguf=%s\n' "${GGUF:-UNSET}"
    printf 'internal_sweep_spec=%s\n' "${INTERNAL_SWEEP_SPEC:-UNSET}"
  } >"$out/orchestration.provenance.txt"

  printf '[internal-full-sweep] running ScaleFirst component: %s\n' "$scale_runner"
  OUT="$scale_out" INTERNAL_SWEEP_COMPONENT=scale_first \
    bash "$scale_runner" 2>&1 | tee "$scale_log"
  scale_rc=${PIPESTATUS[0]}

  printf '[internal-full-sweep] running FullyQuantized component: %s\n' "$fq_runner"
  OUT="$fq_out" INTERNAL_SWEEP_COMPONENT=fully_quantized \
    bash "$fq_runner" 2>&1 | tee "$fq_log"
  fq_rc=${PIPESTATUS[0]}

  {
    printf 'scale_first_rc=%d\n' "$scale_rc"
    printf 'fully_quantized_rc=%d\n' "$fq_rc"
    printf 'scale_first_log_sha256=%s\n' "$(sha256sum "$scale_log" | awk '{print $1}')"
    printf 'fully_quantized_log_sha256=%s\n' "$(sha256sum "$fq_log" | awk '{print $1}')"
  } >>"$out/orchestration.provenance.txt"
  if [ "$scale_rc" -ne 0 ] || [ "$fq_rc" -ne 0 ]; then
    printf '[internal-full-sweep] INCOMPLETE: scale_first_rc=%d fully_quantized_rc=%d; no merged winner published\n' \
      "$scale_rc" "$fq_rc" >&2
    printf '[internal-full-sweep] artifacts: %s\n' "$out" >&2
    return 3
  fi

  local scale_summary="$scale_out/$scale_summary_rel"
  local fq_summary="$fq_out/$fq_summary_rel"
  if [ ! -f "$scale_summary" ] || [ ! -f "$fq_summary" ]; then
    printf '[internal-full-sweep] INCOMPLETE: component summary missing\n' >&2
    printf '  ScaleFirst: %s (%s)\n' "$scale_summary" "$([ -f "$scale_summary" ] && printf present || printf missing)" >&2
    printf '  FullyQuantized: %s (%s)\n' "$fq_summary" "$([ -f "$fq_summary" ] && printf present || printf missing)" >&2
    return 3
  fi

  python3 -B "$root/tools/merge_internal_full_sweep.py" self-test \
    | tee "$out/merger-self-test.log"
  local self_rc=${PIPESTATUS[0]}
  if [ "$self_rc" -ne 0 ]; then
    printf '[internal-full-sweep] FAIL: merger self-test returned rc=%d\n' "$self_rc" >&2
    return "$self_rc"
  fi
  python3 -B "$root/tools/merge_internal_full_sweep.py" merge \
    --scale-first "$scale_summary" --fully-quantized "$fq_summary" \
    --out "$out/results" 2>&1 | tee "$out/merge.log"
  local merge_rc=${PIPESTATUS[0]}
  if [ "$merge_rc" -ne 0 ]; then
    printf '[internal-full-sweep] INCOMPLETE: merge contract rejected the component results\n' >&2
    printf '[internal-full-sweep] artifacts: %s\n' "$out" >&2
    return "$merge_rc"
  fi
  {
    printf 'scale_first_summary_sha256=%s\n' "$(sha256sum "$scale_summary" | awk '{print $1}')"
    printf 'fully_quantized_summary_sha256=%s\n' "$(sha256sum "$fq_summary" | awk '{print $1}')"
    printf 'merged_summary_sha256=%s\n' "$(sha256sum "$out/results/summary.json" | awk '{print $1}')"
  } >>"$out/orchestration.provenance.txt"
  printf '[internal-full-sweep] PASS\n'
  printf '[internal-full-sweep] cells: %s\n' "$out/results/cells.tsv"
  printf '[internal-full-sweep] winners: %s\n' "$out/results/winners.tsv"
  printf '[internal-full-sweep] summary: %s\n' "$out/results/summary.json"
  printf '[internal-full-sweep] artifacts: %s\n' "$out"
}

main "$@"
