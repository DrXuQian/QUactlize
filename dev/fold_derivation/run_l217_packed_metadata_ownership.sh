#!/usr/bin/env bash
set -uo pipefail

main() {
  local repo out compiler positive_rc negative_rc tail_negative_rc
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
  out="${QUACTLIZE_L217_OUT:-/workspace/quactlize-l217-packed-metadata-ownership}/run-$$"
  mkdir -p "$out" || return 2
  compiler="${NVCC:-$(command -v nvcc 2>/dev/null || true)}"
  if [ -z "$compiler" ]; then
    printf '[l217-runner] FAIL: nvcc unavailable\n' >&2
    return 2
  fi
  local common=(-std=c++17 -O2 -x cu -arch=sm_80 --expt-relaxed-constexpr -w
                -I "$repo/dev/fold_derivation/stub_inc"
                -I "$repo/third_party/actlize/include"
                -I "$repo/quactlize/include")

  "$compiler" "${common[@]}" \
    "$repo/dev/fold_derivation/l217_packed_metadata_ownership.cu" \
    -o "$out/derived" >"$out/derived-build.log" 2>&1 || {
      tail -n 120 "$out/derived-build.log" >&2
      return 2
    }
  "$out/derived" | tee "$out/derived.log"
  positive_rc=${PIPESTATUS[0]}

  "$compiler" "${common[@]}" -DL217_LEGACY_ONE_COLUMN=1 \
    "$repo/dev/fold_derivation/l217_packed_metadata_ownership.cu" \
    -o "$out/legacy" >"$out/legacy-build.log" 2>&1 || {
      tail -n 120 "$out/legacy-build.log" >&2
      return 2
    }
  "$out/legacy" >"$out/legacy.log" 2>&1
  negative_rc=$?

  "$compiler" "${common[@]}" -DL217_SKIP_TAIL_ZERO=1 \
    "$repo/dev/fold_derivation/l217_packed_metadata_ownership.cu" \
    -o "$out/missing-tail-zero" >"$out/missing-tail-zero-build.log" 2>&1 || {
      tail -n 120 "$out/missing-tail-zero-build.log" >&2
      return 2
    }
  "$out/missing-tail-zero" >"$out/missing-tail-zero.log" 2>&1
  tail_negative_rc=$?

  if [ "$positive_rc" -ne 0 ] || [ "$negative_rc" -eq 0 ] || [ "$tail_negative_rc" -eq 0 ]; then
    printf '[l217-runner] FAIL: positive=%d legacy=%d missing-tail-zero=%d\n' \
      "$positive_rc" "$negative_rc" "$tail_negative_rc" >&2
    return 1
  fi
  grep -Fqx \
    'L217_CASE tile_n=64 threads=32 owners=64 cpt=1 copy_missing=32 first_copy_missing=32 decode_missing=0 unowned_reads=32 map_bad=0 duplicate_publishers=0 predicate_bad=256 first_predicate_bad=512' \
    "$out/legacy.log" || {
      printf '[l217-runner] FAIL: legacy arm lost the exact device failure signature\n' >&2
      cat "$out/legacy.log" >&2
      return 1
    }
  grep -Fqx \
    'L217_SUMMARY variant=derived-ownership cases=6 copy_missing=0 decode_missing=0 unowned_reads=0 map_bad=0 duplicate_publishers=0 predicate_bad=0 verdict=PASS' \
    "$out/derived.log" || {
      printf '[l217-runner] FAIL: derived ownership denominator changed\n' >&2
      return 1
    }
  grep -Fqx \
    'L217_TOTAL_SUMMARY cases=6 missing=0 duplicates=0 decoded=2880 zeroed=960 verdict=PASS' \
    "$out/derived.log" || {
      printf '[l217-runner] FAIL: decode-owner total-overwrite denominator changed\n' >&2
      return 1
    }
  grep -Eq '^L217_TOTAL_SUMMARY cases=6 missing=[1-9][0-9]* duplicates=0 .* verdict=FAIL$' \
    "$out/missing-tail-zero.log" || {
      printf '[l217-runner] FAIL: missing tail-zero plant stayed green or changed signature\n' >&2
      cat "$out/missing-tail-zero.log" >&2
      return 1
    }
  grep -Fqx \
    'L217_CASE tile_n=64 threads=128 owners=64 cpt=1 copy_missing=0 first_copy_missing=-1 decode_missing=0 unowned_reads=0 map_bad=0 duplicate_publishers=64 predicate_bad=0 first_predicate_bad=-1' \
    "$out/legacy.log" || {
      printf '[l217-runner] FAIL: legacy arm lost the exact two-publisher CTA128 signature\n' >&2
      cat "$out/legacy.log" >&2
      return 1
    }
  printf '[l217-runner] PASS: exact 32-column hole, CTA128 duplicate publishers and missing tail-zero RED; six derived ownership/total-overwrite cells exact; artifacts=%s\n' "$out"
}

main "$@"
