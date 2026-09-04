#!/usr/bin/env bash
# Exact epilogue-only closure for the grouped Q4 K-pack TM8 failure.
# It builds one binary and compares the same M=9,N=64 output:
#   control:   production TN32/WN32, two one-warp CTAs on N
#   legacy:    test-local TN64/WN16 with the historical width-eight copy
#   candidate: production TN64/WN16 with fragment-capped output ownership
# The device kernel constructs no A, B, metadata, mainloop or scheduler state.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=test_ppu_m8n16_collective
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/quactlize-m8-epilogue-topology.XXXXXX")}"
BUILD_ROOT="${PPU_BUILD_DIR:-$OUT/build}"
BUILD_LOG="$OUT/build.log"
RUN_LOG="$OUT/run.log"
mkdir -p "$OUT"

one_binary() {
  local -a bins=()
  mapfile -t bins < <(find "$BUILD_ROOT" -type f -name "$TARGET" -perm -u+x 2>/dev/null)
  if [ "${#bins[@]}" -ne 1 ]; then
    printf '[m8-epilogue-topology] FAIL binary_count=%d build=%s\n' \
      "${#bins[@]}" "$BUILD_ROOT" >&2
    return 1
  fi
  printf '%s\n' "${bins[0]}"
}

main() {
  python3 "$ROOT/ci/check_m8n16_epilogue_topology_contract.py" || return 2
  printf '[m8-epilogue-topology] source=%s actlize=%s artifacts=%s\n' \
    "$(git -C "$ROOT" rev-parse HEAD)" \
    "$(git -C "$ROOT/third_party/actlize" rev-parse HEAD)" "$OUT"

  if ! env PPU_BUILD_DIR="$BUILD_ROOT" PPU_ARCHS=ppu0010 \
      PPU_DEFS=PPU_PACKED_SCALE=1 TARGET="$TARGET" \
      "$ROOT/build.sh" 2>&1 | tee "$BUILD_LOG"; then
    printf '[m8-epilogue-topology] INFRASTRUCTURE_FAIL phase=build log=%s\n' \
      "$BUILD_LOG" >&2
    tail -80 "$BUILD_LOG" >&2
    return 2
  fi

  local bin
  bin="$(one_binary)" || return 2
  local run_rc=0
  "$bin" --epilogue-topology-only >"$RUN_LOG" 2>&1 || run_rc=$?
  cat "$RUN_LOG"

  local control m8_control subject control_owner legacy_owner subject_owner ab
  control="$(grep '^FQ_M8_EPILOGUE_TOPOLOGY arm=tn32-wn32-control ' "$RUN_LOG" || true)"
  m8_control="$(grep '^FQ_M8_EPILOGUE_TOPOLOGY arm=tn64-wn16-candidate-m8-control ' "$RUN_LOG" || true)"
  subject="$(grep '^FQ_M8_EPILOGUE_TOPOLOGY arm=tn64-wn16-candidate ' "$RUN_LOG" || true)"
  control_owner="$(grep '^FQ_M8_EPILOGUE_FIRST_TILE_OWNERSHIP arm=tn32-wn32-control ' "$RUN_LOG" || true)"
  legacy_owner="$(grep '^FQ_M8_EPILOGUE_FIRST_TILE_OWNERSHIP arm=tn64-wn16-legacy ' "$RUN_LOG" || true)"
  subject_owner="$(grep '^FQ_M8_EPILOGUE_FIRST_TILE_OWNERSHIP arm=tn64-wn16-candidate ' "$RUN_LOG" || true)"
  ab="$(grep '^FQ_M8_EPILOGUE_TOPOLOGY_AB ' "$RUN_LOG" || true)"
  if [ -z "$control" ] || [ -z "$m8_control" ] || [ -z "$subject" ] || \
     [ -z "$control_owner" ] || [ -z "$legacy_owner" ] || \
     [ -z "$subject_owner" ] || [ -z "$ab" ]; then
    printf '[m8-epilogue-topology] INFRASTRUCTURE_FAIL phase=markers rc=%d log=%s\n' \
      "$run_rc" "$RUN_LOG" >&2
    return 2
  fi

  local verdict
  if [[ "$control" != *' packed_metadata=1 '* ]] || \
     [[ "$m8_control" != *' packed_metadata=1 '* ]] || \
     [[ "$subject" != *' packed_metadata=1 '* ]] || \
     [[ "$control" != *' cta_threads=32 fragment=8 output_alignment=8 epi_thread_map=8x4 '* ]] || \
     [[ "$m8_control" != *' cta_threads=128 fragment=4 output_alignment=4 epi_thread_map=8x16 '* ]] || \
     [[ "$subject" != *' cta_threads=128 fragment=4 output_alignment=4 epi_thread_map=8x16 '* ]]; then
    verdict=CONTROL_DIRTY
  elif [[ "$control" != *' positive_bad=0/576 '* ]] || \
     [[ "$control" != *' negative_oracle_bad=0/576 '* ]] || \
     [[ "$control" != *' observed_red=288 expected_red=288 '* ]] || \
     [[ "$control" != *' cohort_red=[0,0,144,144] EXPECTED_RED' ]] || \
     [[ "$m8_control" != *' positive_bad=0/512 '* ]] || \
     [[ "$m8_control" != *' negative_oracle_bad=0/512 '* ]] || \
     [[ "$m8_control" != *' observed_red=384 expected_red=384 '* ]] || \
     [[ "$m8_control" != *' cohort_red=[0,128,128,128] EXPECTED_RED' ]] || \
     [[ "$control_owner" != *' ownership_bad=0/576 '* ]] || \
     [[ "$control_owner" != *' first8_bad=0/512 guard_bad=0 finite_bad=0 '* ]] || \
     [[ "$control_owner" != *' row8_written=0/64 '* ]] || \
     [[ "$control_owner" != *' cohort_written=[0,0,0,0] EXACT_OWNER' ]]; then
    verdict=CONTROL_DIRTY
  elif [[ "$legacy_owner" != *' ownership_bad=64/576 '* ]] || \
       [[ "$legacy_owner" != *' first8_bad=0/512 guard_bad=0 finite_bad=0 '* ]] || \
       [[ "$legacy_owner" != *' row8_written=64/64 '* ]] || \
       [[ "$legacy_owner" != *' cohort_written=[16,16,16,16] ILLEGAL_ROW8_WRITE' ]]; then
    verdict=LEGACY_RED_MISSING
  elif [[ "$subject_owner" != *' ownership_bad=0/576 '* ]] || \
       [[ "$subject_owner" != *' first8_bad=0/512 guard_bad=0 finite_bad=0 '* ]] || \
       [[ "$subject_owner" != *' row8_written=0/64 '* ]] || \
       [[ "$subject_owner" != *' cohort_written=[0,0,0,0] EXACT_OWNER' ]] || \
       [[ "$subject" != *' positive_bad=0/576 '* ]] || \
       [[ "$subject" != *' negative_oracle_bad=0/576 '* ]] || \
       [[ "$subject" != *' observed_red=432 expected_red=432 '* ]] || \
       [[ "$subject" != *' cohort_red=[0,144,144,144] EXPECTED_RED' ]] || \
       [[ "$ab" != *' cross_bad=0/576 verdict=CANDIDATE_MATCHES_CONTROL' ]] || \
       [ "$run_rc" -ne 0 ]; then
    verdict=CANDIDATE_DIRTY
  else
    verdict=LEGACY_RED_CANDIDATE_GREEN
  fi
  printf 'FQ_M8_EPILOGUE_TOPOLOGY_VERDICT verdict=%s run_rc=%d artifacts=%s\n' \
    "$verdict" "$run_rc" "$OUT"
  [ "$verdict" = LEGACY_RED_CANDIDATE_GREEN ]
}

main "$@"
