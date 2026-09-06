#!/usr/bin/env bash
# Bounded real-shape tuning. Does not read timings from the old full campaign.
# Build only the per-shape candidate union; cache modules for later searches.
# Run this script with bash, never source it into the interactive shell.
set -euo pipefail

main() {
  local root out sdk mode budget jobs qtypes devices cache
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  out="${OUT:?Set OUT to a result directory on the data disk}"
  sdk="${PPU_SDK:?Set PPU_SDK to the SDK root containing bin/hgcc}"
  mode="${MODE:-screen}"
  budget="${CANDIDATE_BUDGET:-32}"
  jobs="${JOBS:-192}"
  qtypes="${QTYPES:-10,11,12,13,14}"
  devices="${DEVICES:-0,1,2,3,4,5,6,7}"
  case "$mode" in screen|pilot|build|retry) ;; *)
    printf '[kpack-budgeted] MODE must be screen, pilot, build or retry\n' >&2; return 2;; esac
  case "$out" in /*) ;; *) printf '[kpack-budgeted] OUT must be absolute\n' >&2; return 2;; esac
  case "$out" in /|/root|/workspace|/root/autodl-tmp)
    printf '[kpack-budgeted] OUT must be a dedicated child directory\n' >&2; return 2;; esac
  if [ -L "$out" ]; then printf '[kpack-budgeted] OUT may not be a symlink\n' >&2; return 2; fi
  if [ ! -x "$sdk/bin/hgcc" ] || [ ! -x "$sdk/bin/hgobjdump" ]; then
    printf '[kpack-budgeted] SDK lacks bin/hgcc or bin/hgobjdump: %s\n' "$sdk" >&2; return 2
  fi
  mkdir -p "$out"
  out="$(realpath -e -- "$out")"
  sdk="$(realpath -e -- "$sdk")"
  cache="${BUILD_CACHE:-$out/build}"
  case "$cache" in /*) ;; *) printf '[kpack-budgeted] BUILD_CACHE must be absolute\n' >&2; return 2;; esac
  cache="$(realpath -m -- "$cache")"
  case "$cache" in /|/root|/workspace|"$out")
    printf '[kpack-budgeted] BUILD_CACHE must be a dedicated build directory, not OUT itself\n' >&2; return 2;; esac
  exec 9>"$out/campaign.lock"
  if ! flock -n 9; then
    printf '[kpack-budgeted] another launcher owns this OUT\n' >&2; return 2
  fi
  export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib64:$sdk/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="$sdk/bin:$PATH"
  local -a scope=() anchors=() retry=()
  if [ "$mode" = pilot ]; then scope=(--pilot); fi
  if [ -n "${ANCHORS:-}" ]; then anchors=(--anchors "$ANCHORS"); fi
  if [ "$mode" = retry ]; then retry=(--retry-failures); fi
  printf '[kpack-budgeted] SOURCE sha=%s\n' "$(git -C "$root" rev-parse HEAD)"
  # First admit the new DSO transport on real Q4 dense/grouped workloads.
  # Its modules are a subset of the full selection and share the same cache;
  # its timings never substitute for the fresh full screen below.
  if [ "$mode" = screen ]; then
    printf '[kpack-budgeted] MODULE_GATE qtype=12 workloads=4 routes=8\n'
    python3 -B "$root/tools/kpack_tuning_plan.py" --output "$out/module-gate-plan.json" \
      --budget "$budget" --qtypes 12 --pilot
    python3 -B "$root/tools/build_kpack_tuner.py" --plan "$out/module-gate-plan.json" \
      --output "$cache" --sdk "$sdk" --jobs "$jobs" \
      --parents-per-module "${PARENTS_PER_MODULE:-1}"
    cp -- "$cache/bundle.json" "$out/module-gate-bundle.json"
    python3 -B "$root/tools/run_kpack_tuner.py" --plan "$out/module-gate-plan.json" \
      --bundle "$out/module-gate-bundle.json" --output "$out/module-gate" \
      --sdk "$sdk" --devices "$devices" --iterations 2
  fi
  printf '[kpack-budgeted] PLAN mode=%s qtypes=%s budget_parents=%s old_cartesian=DISABLED\n' "$mode" "$qtypes" "$budget"
  python3 -B "$root/tools/kpack_tuning_plan.py" --output "$out/plan.json" \
    --budget "$budget" --qtypes "$qtypes" "${scope[@]}" "${anchors[@]}"
  printf '[kpack-budgeted] BUILD jobs=%s logical_cpus=%s parents_per_module=%s cache=%s\n' \
    "$jobs" "$(getconf _NPROCESSORS_ONLN)" "${PARENTS_PER_MODULE:-1}" "$cache"
  python3 -B "$root/tools/build_kpack_tuner.py" --plan "$out/plan.json" \
    --output "$cache" --sdk "$sdk" --jobs "$jobs" \
    --parents-per-module "${PARENTS_PER_MODULE:-1}"
  cp -- "$cache/bundle.json" "$out/bundle.json"
  if [ "$mode" = build ]; then return 0; fi
  printf '[kpack-budgeted] RUN devices=%s correctness=1 samples=%s same_fixture_across_modules=1\n' \
    "$devices" "${ITERATIONS:-2}"
  python3 -B "$root/tools/run_kpack_tuner.py" --plan "$out/plan.json" \
    --bundle "$out/bundle.json" --output "$out/screen" --sdk "$sdk" \
    --devices "$devices" --iterations "${ITERATIONS:-2}" "${retry[@]}"
  printf '[kpack-budgeted] DONE summary=%s heuristic_input=%s\n' \
    "$out/screen/summary.tsv" "$out/screen/heuristic-input.json"
}
main "$@"
