#!/usr/bin/env bash
# Gate for the MoE sweep unit GENERATOR. It SLICES ../CMakeLists.txt at run time rather than holding a copy, because the
# copy went stale within an hour: the committed .cmake still had the old dispatcher shape and reported OK while the real
# generator had moved on to per-format Best slots. A gate that can disagree with the thing it gates is not a gate.
#
# WHY IT MUST BE CMAKE. The bug it was written for could only be caught by CMake: `;` IS the list separator, so a WarpN
# list written "32;64" inside a row does not stay inside it -- foreach(IN LISTS) splits the row and the stray fragment has
# one field. A python mirror of the same table cannot see this (python's split(';') has no such semantics); the mirror
# validated the enumeration and passed while the real configure died with six "list index out of range" errors per row.
# The generator also printed "128 generated units" while broken, because 8 malformed iterations x 16 equals the 5 correct
# formats x 16 -- so the count that exists to catch a shrunken enumeration agreed with the right answer for the wrong
# reason.
#
#   ./gen_moe_units_check.sh                 assert the generator is sane (expectations DERIVED from the emitted tables)
#   BAD=1 ./gen_moe_units_check.sh           negative control: put a ';' back in a row, must FATAL
#   BAD=2 ./gen_moe_units_check.sh           negative control: drop one shape per format, must FATAL
#
# The negative controls get their OWN output directory. The script starts with REMOVE_RECURSE, so running a broken variant
# into the shared dir wipes the good tree and dies partway, and the wreckage reads exactly like a real failure -- I
# diagnosed "q6 is missing WarpN=64" from the corpse of my own negative control once.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Told by build.sh; the fallback is the repo layout, for running this by hand. It used to be "one directory up",
# which held CMakeLists.txt before the tree split into quactlize/include, tests/ and benchmarks/. Nothing noticed,
# because nothing CALLED this script -- it was wired into neither build.sh nor ci/local_gates.py. It is now.
CML="${QUACTLIZE_CMAKE:-$HERE/../../quactlize/csrc/CMakeLists.txt.in}"
CSRC="$(cd "$(dirname "$CML")" && pwd)"
OUT="$HERE/.moe_units_check${BAD:+_bad$BAD}"
GEN="$OUT/gen.cmake"
mkdir -p "$OUT"

# THE SLICE NOW HAS DEPENDENCIES, and they are sliced/included too rather than transcribed. The generator stopped being
# self-contained on 2026-08-06, when the unit shapes moved from four axis lists to the emitted tactic tables: it now calls
# qz_resolve_sources (defined ABOVE the slice, in the same file) and qz_parse_tactic_xmacro (a separate module the real
# CMakeLists includes). A fragment missing them dies with "Unknown CMake command", which is loud but useless -- and the
# tempting repair, pasting the two functions in here, recreates exactly the stale-copy failure the first paragraph is
# about. So: slice the one, include the other, and check both landed.
# qz_resolve_sources: from its own `function(` line through the matching `endfunction()`.
awk '/^function\(qz_resolve_sources/{p=1} p{print} p&&/^endfunction\(\)/{exit}' "$CML" > "$OUT/resolve.cmake"
grep -q '^endfunction()' "$OUT/resolve.cmake" || {
  echo "  [FAIL] gen_moe_units_check: could not slice qz_resolve_sources out of $CML -- the anchors moved"; exit 1; }
# QZ_SRC_DIRS: the five directories the flat overlay used to be. Sliced from the REAL CMakeLists.txt for the same reason
# as everything else here. In the overlay the variable is unset and every name resolves flat; locally the tables live in
# benchmarks/, so the gate needs the resolving form -- which is also the form the resolution bug would live in.
awk '/^get_filename_component\(QZ_ROOT/{p=1} p{print} p&&/^set\(QZ_SRC_DIRS/{q=1} q&&/\)[[:space:]]*$/{exit}' \
  "$CSRC/CMakeLists.txt" > "$OUT/srcdirs.cmake"
grep -q '^set(QZ_SRC_DIRS' "$OUT/srcdirs.cmake" || {
  echo "  [FAIL] gen_moe_units_check: could not slice QZ_SRC_DIRS out of $CSRC/CMakeLists.txt -- the anchors moved"; exit 1; }
[ -f "$CSRC/TacticTableUnits.cmake" ] || {
  echo "  [FAIL] gen_moe_units_check: $CSRC/TacticTableUnits.cmake is missing"; exit 1; }

# THE POLICY STATE MUST BE THE REAL ONE. cmake -P starts every policy at OLD unless a minimum is declared, and the slice
# is a slice of a file that runs under the project's. That is not hypothetical: the first run of this repaired gate died
# with "parsed 201 tactic rows, but the table declares 202" -- CMP0007 OLD makes list() drop empty elements, so every
# blank line in the table shifted the physical-line indices qz_parse_tactic_xmacro uses. Taken from the top-level
# CMakeLists rather than written here, because a version transcribed into a gate is one more thing that can disagree.
QZ_ROOT="$(cd "$CSRC/../.." && pwd)"
CMIN="$(grep -m1 -oE '^cmake_minimum_required\(VERSION [0-9.]+' "$QZ_ROOT/CMakeLists.txt" | grep -oE '[0-9.]+$' || true)"
[ -n "$CMIN" ] || { echo "  [FAIL] gen_moe_units_check: no cmake_minimum_required in $QZ_ROOT/CMakeLists.txt"; exit 1; }

{
  echo "cmake_minimum_required(VERSION $CMIN)"
  echo "set(CMAKE_CURRENT_BINARY_DIR \"$OUT\")"
  # The generator resolves table paths relative to this, so it must be csrc/ and not wherever cmake -P was started.
  echo "set(CMAKE_CURRENT_SOURCE_DIR \"$CSRC\")"
  cat "$OUT/srcdirs.cmake"
  cat "$OUT/resolve.cmake"
  echo "include(\"$CSRC/TacticTableUnits.cmake\")"
  echo 'file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/moe_units")'
  # the slice: from the format table to just before the executable that consumes the generated sources
  awk '/^set\(_MOE_FORMATS/{p=1} /^quactlize_ppu_executable\(/{if(p)exit} p' "$CML" \
    | { case "${BAD:-}" in
          1) sed 's/4|2|32,64/4|2|32;64/' ;;
          # Drop one shape per format AFTER the generator has read the table: the table is untouched, so the derived
          # expectation below still says the shape should exist and the set comparison is what has to notice.
          2) sed 's/^  list(REMOVE_DUPLICATES _MOE_SHAPES)$/&\n  list(REMOVE_AT _MOE_SHAPES 0)/' ;;
          *) cat ;;
        esac; }
  cat <<'ASSERT'
# THE EXPECTATION IS DERIVED FROM THE TABLES, BY A SECOND PARSE. Until 2026-08-06 it was the product of the axis lists,
# which is no longer what the generator enumerates -- it projects the emitted rows onto (TileM,TileN,WarpM,WarpN) and
# de-duplicates, because one sweep unit is one shape and loops every stage count at runtime. Restating 415 here would be
# the hand-maintained expectation this gate has already been burned by twice.
#
# NOT via qz_parse_tactic_xmacro. That helper is what the generator uses; calling it here would make the expectation a
# restatement of the answer. string(REGEX MATCHALL) over whole X(...) rows is a genuinely different parse -- no line
# splitting, no continuation handling -- and it cross-checks against the count each table declares about itself, so a
# regex that silently matches too few rows cannot pass.
set(_exp_names "")
set(_exp_rows 0)
foreach(_row IN LISTS _MOE_FORMATS)
  string(REPLACE "|" ";" _f "${_row}")
  list(GET _f 0 _nm)
  list(GET _f 7 _emit)
  # Resolved WITHOUT qz_resolve_sources, for the same reason: which file the generator opened is part of what is checked.
  set(_hit "")
  foreach(_d IN LISTS QZ_SRC_DIRS)
    if(EXISTS "${_d}/lowbit_grouped_${_emit}_configs.inc")
      list(APPEND _hit "${_d}/lowbit_grouped_${_emit}_configs.inc")
    endif()
  endforeach()
  list(LENGTH _hit _nhit)
  if(NOT _nhit EQUAL 1)
    message(FATAL_ERROR "table for ${_nm}: lowbit_grouped_${_emit}_configs.inc has ${_nhit} candidates in QZ_SRC_DIRS, expected 1")
  endif()
  list(GET _hit 0 _t)
  file(READ "${_t}" _txt)
  string(REGEX MATCHALL "X\\([0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,B\\)" _hits "${_txt}")
  list(LENGTH _hits _nrows)
  string(TOUPPER "${_emit}" _uc)
  if(NOT _txt MATCHES "#define[ \t]+LOWBIT_GROUPED_${_uc}_CFG_ROWS[ \t]+([0-9]+)")
    message(FATAL_ERROR "${_t}: no LOWBIT_GROUPED_${_uc}_CFG_ROWS to check the parse against")
  endif()
  if(NOT _nrows EQUAL CMAKE_MATCH_1)
    message(FATAL_ERROR "${_t}: this gate matched ${_nrows} rows, the table declares ${CMAKE_MATCH_1}")
  endif()
  math(EXPR _exp_rows "${_exp_rows} + ${_nrows}")
  foreach(_h IN LISTS _hits)
    string(REGEX REPLACE "^X\\(|,B\\)$" "" _h "${_h}")
    string(REPLACE "," ";" _hf "${_h}")
    list(GET _hf 0 _tm)
    list(GET _hf 1 _tn)
    list(GET _hf 3 _wm)
    list(GET _hf 4 _wn)
    list(APPEND _exp_names "moe_unit_${_nm}_tn${_tn}_wn${_wn}_tm${_tm}_wm${_wm}")
  endforeach()
endforeach()
list(REMOVE_DUPLICATES _exp_names)
list(LENGTH _exp_names _exp)
list(LENGTH _MOE_FORMATS _nfmt)

file(GLOB _gen "${_MOE_GEN_DIR}/*.cu")
set(_got_names "")
foreach(_g IN LISTS _gen)
  get_filename_component(_b "${_g}" NAME_WE)
  list(APPEND _got_names "${_b}")
endforeach()
list(LENGTH _got_names _ngen)
message(STATUS "generated .cu on disk: ${_ngen} (expected ${_exp} distinct shapes over ${_exp_rows} table rows; generator said ${_MOE_UNIT_N})")

# SET EQUALITY, NOT COUNT EQUALITY -- and this replaces two checks rather than adding a third. It subsumes the old
# per-pattern count (`moe_unit_q6_tn64_wn64_*` must appear TileM x WarpM times, written to catch a dropped WarpN) and the
# old stray-name glob (`moe_unit_64_*`, written to catch the ';' bug), both of which were counts and both of which the
# opening paragraph's failure -- the right count from the wrong iterations -- goes straight through. Naming the units
# that differ is also the only form that says WHICH shape went missing.
set(_missing "${_exp_names}")
if(_got_names)
  list(REMOVE_ITEM _missing ${_got_names})
endif()
set(_extra "${_got_names}")
if(_exp_names)
  list(REMOVE_ITEM _extra ${_exp_names})
endif()
if(_missing OR _extra)
  list(LENGTH _missing _nmiss)
  list(LENGTH _extra _nextra)
  set(_show "")
  foreach(_x IN LISTS _missing)
    set(_show "${_show}\n    MISSING  ${_x}")
  endforeach()
  foreach(_x IN LISTS _extra)
    set(_show "${_show}\n    EXTRA    ${_x}")
  endforeach()
  message(FATAL_ERROR "generated units do not match the shapes in the tables: ${_nmiss} missing, ${_nextra} extra${_show}")
endif()
# The generator's own list length, against the files that reached the disk. Independent of everything above: it catches a
# file(WRITE) that failed, which leaves the name in _MOE_UNIT_SRCS and nothing on disk.
if(NOT _ngen EQUAL _MOE_UNIT_N)
  message(FATAL_ERROR "generator listed ${_MOE_UNIT_N} sources but ${_ngen} are on disk")
endif()
# the dispatcher must declare and call every unit, and name one Best slot per FORMAT
file(READ "${_MOE_GEN_DIR}/moe_bench_units.inc" _d)
string(REGEX MATCHALL "\nvoid moe_unit_" _dc "${_d}")
list(LENGTH _dc _ndc)
string(REGEX MATCHALL "\n  moe_unit_" _cc "${_d}")
list(LENGTH _cc _ncc)
if(NOT _ndc EQUAL _exp OR NOT _ncc EQUAL _exp)
  message(FATAL_ERROR "dispatcher has ${_ndc} declarations and ${_ncc} calls, expected ${_exp} of each")
endif()
foreach(_need "MOE_UNIT_COUNT ${_exp}" "MOE_FMT_COUNT ${_nfmt}" "moe_fmt_names")
  string(FIND "${_d}" "${_need}" _f)
  if(_f EQUAL -1)
    message(FATAL_ERROR "dispatcher is missing '${_need}' -- main will not compile against it")
  endif()
endforeach()
message(STATUS "OK: ${_exp} units == the distinct shapes in ${_nfmt} emitted tables (${_exp_rows} rows), dispatcher ${_exp}/${_exp} with ${_nfmt} format slots")
ASSERT
} > "$GEN"

log="$OUT/log"
if ! cmake -P "$GEN" > "$log" 2>&1; then
  if [ -n "${BAD:-}" ]; then
    echo "  [ok]   gen_moe_units_check (negative control BAD=$BAD): the generator was REJECTED"
    grep -m1 -E "fields, expected|do not match the shapes" "$log" | sed 's/^/           /'
    exit 0
  fi
  echo "  [FAIL] gen_moe_units_check: the generator errored"; sed 's/^/           /' "$log"; exit 1
fi
if [ -n "${BAD:-}" ]; then
  echo "  [FAIL] gen_moe_units_check (negative control BAD=$BAD): a broken generator was ACCEPTED"
  sed 's/^/           /' "$log"; exit 1
fi
sed 's/^/           /' "$log"
