#!/usr/bin/env bash
# Run the REAL GEMV unit generator (a slice of CMakeLists.txt, via cmake -P) against the COMMITTED authority.
#
# WHY THE MANIFEST IS THE AUTHORITY.  The tactic Cartesian product and legality live in C++; the checked view
# consumed by CMake is benchmarks/gemv_tactic_units.cmake.  Re-deriving old hand-written tiers here would make
# this gate compare production with a second, stale generator.  Instead this script includes that exact manifest,
# runs production's foreach/file(WRITE) body, and compares every generated filename with every authority row.
#
# The normal invocation proves all four arms:
#   * full authority: exactly 540 units / 10 groups;
#   * GEMV_GROUPS=i4-native: exactly that non-empty subset and no other unit;
#   * a malformed authority row is rejected by production's seven-field parser;
#   * deleting one authority unit is rejected by the authority-count seam.
#
# BAD=malformed or BAD=missing runs one planted negative directly.  It succeeds only when production rejects the
# plant; this is useful when a caller wants to demonstrate either control independently.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CML="${QUACTLIZE_CMAKE:-$ROOT/quactlize/csrc/CMakeLists.txt.in}"
AUTH="$ROOT/benchmarks/gemv_tactic_units.cmake"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

[ -f "$CML" ] || { echo "  [FAIL] gen_gemv_units_check: missing CMake source $CML"; exit 1; }
[ -f "$AUTH" ] || { echo "  [FAIL] gen_gemv_units_check: missing committed authority $AUTH"; exit 1; }

# Slice exactly the production group table and generator, stopping before the executable that consumes it.  The
# include immediately above `_GEMV_GROUPS` is supplied explicitly in run.cmake so CMAKE_CURRENT_LIST_DIR remains
# irrelevant under cmake -P; positive arms include the committed manifest itself, while negative arms include one
# isolated planted copy.
awk '/^set\(_GEMV_GROUPS/{p=1} /^quactlize_ppu_executable\(/{if(p)exit} p' "$CML" > "$OUT/slice.cmake"
[ -s "$OUT/slice.cmake" ] || {
  echo "  [FAIL] gen_gemv_units_check: empty generator slice -- production anchors moved"
  exit 1
}
grep -q '^[[:space:]]*foreach(_cfg IN LISTS _GEMV_AUTHORITY_UNITS)' "$OUT/slice.cmake" || {
  echo "  [FAIL] gen_gemv_units_check: slice no longer consumes _GEMV_AUTHORITY_UNITS"
  exit 1
}

declared=$(sed -n 's/^set(_GEMV_AUTHORITY_UNIT_COUNT \([0-9][0-9]*\))$/\1/p' "$AUTH")
rows=$(grep -c '^  "[^"[:space:]][^"]*"$' "$AUTH")
groups=$(awk '/^set\(_GEMV_GROUPS/{f=1;next} f&&/^\)/{exit} f&&/\|/{n++} END{print n+0}' "$OUT/slice.cmake")
[ "$declared" = 540 ] || {
  echo "  [FAIL] committed GEMV authority declares $declared units, expected the pinned full space 540"
  exit 1
}
[ "$rows" = 540 ] || {
  echo "  [FAIL] committed GEMV authority contains $rows rows, expected 540"
  exit 1
}
[ "$groups" = 10 ] || {
  echo "  [FAIL] production _GEMV_GROUPS contains $groups groups, expected 10"
  exit 1
}

# Write the expected generated filenames from the AUTHORITY ROWS, not from a second list of axes.  Artifact TileK
# is not part of the filename because it is fixed by the selected layout group; the authority guarantees the
# remaining (StepK, Threads, CtaN, Chunk) tuple is unique inside that group.
authority_filenames() {
  local manifest="$1" filter="$2" out="$3"
  awk -F'|' -v filter="$filter" '
    /^  "/ {
      fmt=$1; sub(/^  "/,"",fmt)
      lay=$2; artifact=$3; sk=$4; th=$5; cn=$6; ch=$7; sub(/"[[:space:]]*$/,"",ch)
      if      (fmt=="int4") short="i4"
      else if (fmt=="int2") short="i2"
      else if (fmt=="int1") short="i1"
      else if (fmt=="q3")   short="q3"
      else if (fmt=="q6")   short="q6"
      else { print "unknown authority format " fmt > "/dev/stderr"; exit 7 }
      group=short "-" lay
      if (filter=="" || group==filter)
        print "gemv_unit_" short "_" lay "_s" sk "_t" th "_n" cn "_c" ch ".cu"
    }
  ' "$manifest" | LC_ALL=C sort > "$out"
}

mutate_authority() {
  local mode="$1" src="$2" dst="$3"
  case "$mode" in
    none)
      cp "$src" "$dst"
      cmp -s "$src" "$dst" || {
        echo "  [FAIL] positive authority copy differs from committed manifest"
        return 1
      }
      ;;
    malformed)
      # Add an eighth field to exactly one authority row.  The generator's own seven-field parser must name it.
      awk 'BEGIN{done=0} !done && /^  "/ {sub(/"[[:space:]]*$/, "|MALFORMED\""); done=1} {print}
           END{if(!done) exit 8}' "$src" > "$dst" || return 1
      ;;
    missing)
      # Delete exactly one unit while leaving _GEMV_AUTHORITY_UNIT_COUNT=540.  A count-only gate that derives its
      # expectation from the shortened list would accept this; production must compare it with the pinned count.
      awk 'BEGIN{done=0} !done && /^  "/ {done=1; next} {print} END{if(!done) exit 8}' "$src" > "$dst" || return 1
      ;;
    *)
      echo "  [FAIL] unknown authority mutation '$mode'"
      return 1
      ;;
  esac
}

generate_case() {
  local name="$1" filter="$2" mutation="$3" expect_failure="$4"
  local dir="$OUT/$name" manifest log="$OUT/$name/log"
  mkdir -p "$dir"
  if [ "$mutation" = none ]; then
    manifest="$AUTH"
  else
    manifest="$OUT/$name/authority.cmake"
    mutate_authority "$mutation" "$AUTH" "$manifest" || return 1
  fi
  {
    echo "set(CMAKE_CURRENT_BINARY_DIR \"$dir\")"
    echo "set(MOE_CORES 1024)"
    echo "set(GEMV_GROUPS \"$filter\")"
    echo "include(\"$manifest\")"
    cat "$OUT/slice.cmake"
  } > "$dir/run.cmake"

  if cmake -P "$dir/run.cmake" > "$log" 2>&1; then
    if [ "$expect_failure" = 1 ]; then
      echo "  [FAIL] gen_gemv_units_check ($name): planted $mutation authority was ACCEPTED"
      sed 's/^/           /' "$log"
      return 1
    fi
  else
    if [ "$expect_failure" != 1 ]; then
      echo "  [FAIL] gen_gemv_units_check ($name): production generator errored"
      sed 's/^/           /' "$log"
      return 1
    fi
    case "$mutation" in
      malformed)
        grep -q "has 8 fields, expected 7" "$log" || {
          echo "  [FAIL] malformed authority failed for the wrong reason"
          sed 's/^/           /' "$log"
          return 1
        }
        ;;
      missing)
        grep -q "GEMV generator emitted 539 units, authority requires 540" "$log" || {
          echo "  [FAIL] missing-unit authority failed for the wrong reason"
          sed 's/^/           /' "$log"
          return 1
        }
        ;;
    esac
    echo "  [ok]   gen_gemv_units_check ($name): production rejected planted $mutation authority"
    return 0
  fi

  local gen="$dir/gemv_units" inc="$dir/gemv_units/gemv_perf_units.inc"
  [ -f "$inc" ] || {
    echo "  [FAIL] gen_gemv_units_check ($name): no gemv_perf_units.inc produced"
    return 1
  }
  authority_filenames "$manifest" "$filter" "$dir/expected.files" || return 1
  find "$gen" -maxdepth 1 -type f -name 'gemv_unit_*.cu' -printf '%f\n' | LC_ALL=C sort > "$dir/actual.files"

  local expected got inc_n inc_g decls calls wanted_groups fail=0
  expected=$(wc -l < "$dir/expected.files" | tr -d ' ')
  got=$(wc -l < "$dir/actual.files" | tr -d ' ')
  inc_n=$(sed -n 's/^#define GEMV_UNIT_COUNT \([0-9][0-9]*\)$/\1/p' "$inc")
  inc_g=$(sed -n 's/^#define GEMV_GROUP_COUNT \([0-9][0-9]*\)$/\1/p' "$inc")
  decls=$(grep -c '^void gemv_unit_' "$inc")
  calls=$(grep -cE '^    gemv_unit_.*\(sh, _bf, b\[[0-9]+\]\);' "$inc")
  wanted_groups=$([ -n "$filter" ] && echo 1 || echo 10)

  [ "$got" = "$expected" ] || { echo "  [FAIL] $name: units on disk $got != authority subset $expected"; fail=1; }
  [ "$inc_n" = "$expected" ] || { echo "  [FAIL] $name: GEMV_UNIT_COUNT $inc_n != $expected"; fail=1; }
  [ "$inc_g" = "$wanted_groups" ] || {
    echo "  [FAIL] $name: GEMV_GROUP_COUNT $inc_g != $wanted_groups"; fail=1;
  }
  [ "$decls" = "$expected" ] || { echo "  [FAIL] $name: $decls declarations != $expected"; fail=1; }
  [ "$calls" = "$expected" ] || { echo "  [FAIL] $name: $calls calls != $expected"; fail=1; }
  if ! cmp -s "$dir/expected.files" "$dir/actual.files"; then
    echo "  [FAIL] $name: generated filename set differs from committed authority subset"
    diff -u "$dir/expected.files" "$dir/actual.files" | head -80 | sed 's/^/           /'
    fail=1
  fi

  if [ -n "$filter" ]; then
    [ "$expected" -gt 0 ] || { echo "  [FAIL] $name: GEMV_GROUPS=$filter selected zero units"; fail=1; }
    if grep -qv '^gemv_unit_i4_native_' "$dir/actual.files"; then
      echo "  [FAIL] $name: GEMV_GROUPS=i4-native leaked another group"
      grep -v '^gemv_unit_i4_native_' "$dir/actual.files" | sed 's/^/           /'
      fail=1
    fi
    grep -q 'gemv_group_names\[\].*{"i4-native"}' "$inc" || {
      echo "  [FAIL] $name: generated group-name table is not exactly i4-native"
      fail=1
    }
  fi
  [ "$fail" = 0 ] || return 1
  echo "  [ok]   gen_gemv_units_check ($name): $got authority units, $inc_g group(s), exact declarations/calls/files"
}

case "${BAD:-}" in
  "")
    generate_case full "" none 0 || exit 1
    generate_case i4-native i4-native none 0 || exit 1
    generate_case malformed "" malformed 1 || exit 1
    generate_case missing "" missing 1 || exit 1
    echo "  [ok]   gen_gemv_units_check: full=540/10, i4-native non-empty/exclusive, both authority plants red"
    ;;
  1|malformed)
    generate_case malformed "" malformed 1 || exit 1
    ;;
  missing)
    generate_case missing "" missing 1 || exit 1
    ;;
  *)
    echo "  [FAIL] gen_gemv_units_check: BAD must be malformed or missing"
    exit 1
    ;;
esac
