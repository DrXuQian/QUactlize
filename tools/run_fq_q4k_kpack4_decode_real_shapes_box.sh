#!/usr/bin/env bash
# Native Q4_K K-pack4 decode sweep over the inventory-owned five N/K families
# and M=1/2/4/8.  One 144-row TM8 AP0/AP1 binary serves every shape.
set -uo pipefail

atomic_text() {
  local destination="$1" value="$2" current
  current="${destination}.current.$$"
  printf '%s\n' "$value" > "$current" || return 2
  mv -f -- "$current" "$destination" || return 2
}

run_phase() {
  local log="$1" commit="${1}.sha256" actual failed current rc
  shift
  if [ -s "$log" ] && [ -s "$commit" ]; then
    actual="$(sha256sum "$log" | awk '{print $1}')" || return 2
    if [ "$(cat "$commit")" != "$actual" ]; then
      printf '[fq-kpack4-real] FAIL: committed phase changed: %s\n' "$log" >&2
      return 2
    fi
    if [ "${RESUME:-0}" != 1 ]; then
      printf '[fq-kpack4-real] FAIL: phase exists without RESUME=1: %s\n' "$log" >&2
      return 2
    fi
    printf '[fq-kpack4-real] resume phase=%s\n' "$log"
    return 0
  fi
  if [ -e "$log" ] || [ -e "$commit" ]; then
    if [ "${RESUME:-0}" != 1 ]; then
      printf '[fq-kpack4-real] FAIL: phase residue without RESUME=1: %s\n' "$log" >&2
      return 2
    fi
    failed="${log}.uncommitted.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    if [ -e "$log" ]; then mv -- "$log" "$failed" || return 2; fi
    if [ -e "$commit" ]; then mv -- "$commit" "${failed}.sha256" || return 2; fi
    printf '[fq-kpack4-real] preserved residue=%s\n' "$failed"
  fi
  current="${log}.current.$$"
  "$@" > "$current" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    failed="${log}.failed.$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mv -- "$current" "$failed" || return 2
    printf '[fq-kpack4-real] FAIL: phase rc=%d preserved=%s\n' "$rc" "$failed" >&2
    tail -100 "$failed" >&2
    return "$rc"
  fi
  mv -- "$current" "$log" || return 2
  actual="$(sha256sum "$log" | awk '{print $1}')" || return 2
  atomic_text "$commit" "$actual"
}

validate_pilot_bundle() {
  local root="$1" bundle="$2"
  python3 -B "$root/tools/check_fq_q4k_kpack4_pilot_bundle.py" validate \
    --root "$root" --bundle "$bundle"
}

main() {
  local root workspace sha short stamp out inventory policy jobs per_unit pilot
  local plan manifest binary source_state saved_state shape_key m n k directory
  local screen_log scheduler_log confirm_log
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 2
  workspace="$(realpath -e /workspace)" || return 2
  sha="$(git -C "$root" rev-parse HEAD)" || return 2
  short="${sha:0:8}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return 2
  out="$(realpath -m -- "${OUT:-/workspace/quactlize-fq-q4k-kpack4-real-${short}-${stamp}-$$}")" || return 2
  case "$out" in "$workspace"/*) ;; *)
    printf '[fq-kpack4-real] FAIL: OUT must be a strict /workspace child\n' >&2
    return 2;; esac
  inventory="${INTERNAL_SWEEP_SPEC:-}"
  if [ -z "$inventory" ] || [ ! -f "$inventory" ]; then
    printf '[fq-kpack4-real] FAIL: INTERNAL_SWEEP_SPEC must name COMPLETE inventory-v2 JSON\n' >&2
    return 2
  fi
  inventory="$(realpath -e -- "$inventory")" || return 2
  policy="$(realpath -e -- "${FQ_Q4K_DECODE_POLICY:-$root/benchmarks/fq_q4k_decode_real_shapes_policy.json}")" || return 2
  if ! cmp -s "$policy" "$root/benchmarks/fq_q4k_decode_real_shapes_policy.json"; then
    printf '[fq-kpack4-real] FAIL: policy must be byte-identical to the pilot policy\n' >&2
    return 2
  fi
  jobs="${JOBS:-16}"
  per_unit="${FQ_CONFIGS_PER_UNIT:-4}"
  case "$jobs:$per_unit" in *[!0-9:]*|0:*|*:0)
    printf '[fq-kpack4-real] FAIL: JOBS/FQ_CONFIGS_PER_UNIT must be positive\n' >&2
    return 2;; esac
  if [ -n "${PPU_DEFS:-}" ] || [ -n "${PPU_EXTRA_DEFS:-}" ]; then
    printf '[fq-kpack4-real] FAIL: ambient PPU_DEFS/PPU_EXTRA_DEFS changes denominator\n' >&2
    return 2
  fi
  if [ -e "$out" ] && [ "${RESUME:-0}" != 1 ]; then
    printf '[fq-kpack4-real] FAIL: refusing existing OUT without RESUME=1: %s\n' "$out" >&2
    return 2
  fi
  mkdir -p "$out/raw" "$out/results" || return 2

  python3 -B "$root/tools/plan_fq_q4k_decode_real_shapes.py" self-test || return 2
  python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" self-test \
    --policy "$policy" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_generator.py" || return 2
  python3 -B "$root/ci/check_fq_q4k_kpack4_real_shapes_runner.py" || return 2
  python3 -B "$root/tools/check_fq_q4k_kpack4_pilot_bundle.py" self-test \
    --root "$root" || return 2

  plan="$out/plan.json"
  if [ ! -s "$plan" ]; then
    python3 -B "$root/tools/plan_fq_q4k_decode_real_shapes.py" materialize \
      --inventory "$inventory" --policy "$policy" --output "$plan" || return 2
  fi
  python3 -B - "$plan" <<'PY' || return 2
import json,sys
value=json.load(open(sys.argv[1]))
assert value["family_count"]==5 and value["shape_count"]==20
assert sorted({tuple(row[k] for k in ("m","n","k")) for row in value["shapes"]})
PY

  pilot="${PILOT_BUNDLE:-}"
  if [ -z "$pilot" ]; then
    pilot="$out/pilot-source"
    printf '[fq-kpack4-real] no PILOT_BUNDLE; building one reusable 144-row pilot\n'
    OUT="$pilot" JOBS="$jobs" FQ_CONFIGS_PER_UNIT="$per_unit" \
      bash "$root/tools/run_fq_q4k_kpack4_pilot_box.sh" || return $?
  fi
  pilot="$(realpath -e -- "$pilot")" || return 2
  case "$pilot" in "$workspace"/*) ;; *)
    printf '[fq-kpack4-real] FAIL: PILOT_BUNDLE must be a strict /workspace child\n' >&2
    return 2;; esac
  manifest="$pilot/generated/manifest.json"
  binary="$pilot/build/ppu_targets/test_fully_quantized_internal_sweep"
  validate_pilot_bundle "$root" "$pilot" || return 2

  source_state="$({
    git -C "$root" rev-parse HEAD
    git -C "$root/third_party/actlize" rev-parse HEAD
    sha256sum "$inventory" "$policy" "$plan" "$manifest" "$binary" \
      "$root/tools/analyze_fq_q4k_kpack4_pilot.py" \
      "$root/tools/run_fq_q4k_kpack4_decode_real_shapes_box.sh"
  } | sha256sum | awk '{print $1}')" || return 2
  saved_state="$out/source-state.sha256"
  if [ -s "$saved_state" ] && [ "$(cat "$saved_state")" != "$source_state" ]; then
    printf '[fq-kpack4-real] FAIL: source/inventory/pilot authority changed on resume\n' >&2
    return 2
  fi
  atomic_text "$saved_state" "$source_state" || return 2
  git -C "$root" diff --binary --no-ext-diff HEAD > "$out/source.patch" || return 2

  while IFS=$'\t' read -r shape_key m n k; do
    directory="$out/raw/$shape_key"
    mkdir -p "$directory" || return 2
    screen_log="$directory/screen.log"
    scheduler_log="$directory/scheduler.log"
    confirm_log="$directory/confirm.log"
    printf '[fq-kpack4-real] shape=%sx%sx%s phase=screen typed=144\n' "$m" "$n" "$k"
    run_phase "$screen_log" "$binary" --shape="${m}x${n}x${k}" \
      --iterations=2 --correctness-repeats=1 --only-split=1 \
      --tm8-max-m=8 --bc-mode=all || return $?
    python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" screen \
      --shape="${m}x${n}x${k}" --manifest "$manifest" --log "$screen_log" \
      --policy "$policy" --symbols-output "$directory/screen-symbols.txt" \
      --summary-output "$directory/screen.json" || return 2

    printf '[fq-kpack4-real] shape=%sx%sx%s phase=scheduler\n' "$m" "$n" "$k"
    run_phase "$scheduler_log" "$binary" --shape="${m}x${n}x${k}" \
      --iterations=1 --correctness-repeats=1 \
      --symbols-file="$directory/screen-symbols.txt" \
      --tm8-max-m=8 --bc-mode=skip || return $?
    python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" scheduler \
      --shape="${m}x${n}x${k}" --manifest "$manifest" --log "$scheduler_log" \
      --screen-symbols "$directory/screen-symbols.txt" --policy "$policy" \
      --symbols-output "$directory/confirm-symbols.txt" \
      --summary-output "$directory/scheduler.json" || return 2

    printf '[fq-kpack4-real] shape=%sx%sx%s phase=confirm\n' "$m" "$n" "$k"
    run_phase "$confirm_log" "$binary" --shape="${m}x${n}x${k}" \
      --iterations=7 --correctness-repeats=2 \
      --symbols-file="$directory/confirm-symbols.txt" \
      --tm8-max-m=8 --bc-mode=skip || return $?
    python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" finalize \
      --shape="${m}x${n}x${k}" --manifest "$manifest" --log "$confirm_log" \
      --symbols "$directory/confirm-symbols.txt" --policy "$policy" \
      --output-json "$directory/summary.json" \
      --output-tsv "$directory/summary.tsv" || return 2
  done < <(python3 -B - "$plan" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
for row in sorted(value["shapes"],key=lambda item:item["shape_key"]):
 print(row["shape_key"],row["m"],row["n"],row["k"],sep="\t")
PY
  )

  python3 -B "$root/tools/analyze_fq_q4k_kpack4_pilot.py" aggregate \
    --plan "$plan" --policy "$policy" --raw-root "$out/raw" \
    --output-json "$out/results/summary.json" \
    --output-tsv "$out/results/summary.tsv" || return 2
  find "$out/raw" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$out/results/raw-authority.sha256" || return 2
  sha256sum "$plan" "$manifest" "$binary" "$out/results/summary.json" \
    > "$out/results/authority.sha256" || return 2
  sed -n '1,21p' "$out/results/summary.tsv"
  printf '[fq-kpack4-real] PASS sha=%s families=5 shapes=20 artifacts=%s\n' \
    "$sha" "$out"
  printf '[fq-kpack4-real] summary=%s\n' "$out/results/summary.tsv"
}

main "$@"
