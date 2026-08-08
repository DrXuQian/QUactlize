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
#   BAD=3 ./gen_moe_units_check.sh           negative control: drop one and duplicate another, same raw count, must FATAL
#   BAD=4 ./gen_moe_units_check.sh           negative control: mix a bc0 wrapper into a bc1 compile policy, must FATAL
#   BAD=5 MOE_TM_LIST=16 ...                  negative control: bypass an advertised tile filter, must FATAL
#   BAD=6 MOE_STAGES=12 ...                   negative control: drop the device stage flag, must FATAL
#   BAD=7                                      negative control: corrupt every wrapper's ArtifactTileK, must FATAL
#
# Every invocation gets its OWN temporary output directory. build.sh is intentionally safe to run concurrently now;
# a fixed .moe_units_check directory let one variant remove another variant's generated sources midway through this gate.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Told by build.sh; the fallback is the repo layout, for running this by hand. It used to be "one directory up",
# which held CMakeLists.txt before the tree split into quactlize/include, tests/ and benchmarks/. Nothing noticed,
# because nothing CALLED this script -- it was wired into neither build.sh nor ci/local_gates.py. It is now.
CML="${QUACTLIZE_CMAKE:-$HERE/../../quactlize/csrc/CMakeLists.txt.in}"
CSRC="$(cd "$(dirname "$CML")" && pwd)"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/quactlize-moe-units-check.XXXXXX")"
cleanup() { rm -rf -- "$OUT"; }
trap cleanup EXIT
GEN="$OUT/gen.cmake"
MOE_CHECK_CORES="${MOE_CHECK_CORES:-192}"
MOE_CHECK_FORMATS="${MOE_FORMATS:-}"
MOE_CHECK_TM_LIST="${MOE_TM_LIST:-}"
MOE_CHECK_TN_LIST="${MOE_TN_LIST:-}"
MOE_CHECK_WM_LIST="${MOE_WM_LIST:-}"
MOE_CHECK_STAGES="${MOE_STAGES:-}"
_safe_cache_list_re='^[A-Za-z0-9_;]*$'
for _value in "$MOE_CHECK_FORMATS" "$MOE_CHECK_TM_LIST" "$MOE_CHECK_TN_LIST" "$MOE_CHECK_WM_LIST" "$MOE_CHECK_STAGES"; do
  [[ "$_value" =~ $_safe_cache_list_re ]] || {
    echo "  [FAIL] gen_moe_units_check: unsafe cache-list spelling '$_value'"; exit 1; }
done

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
# The slice starts at _MOE_FORMATS, but two inputs it consumes are initialised earlier. Slice those real blocks too:
# PPU_EXTRA_DEFS carries the legacy MOE_STAGES_N interface, and MOE_CORES is validated before its first GEMV use.
awk '/^set\(_PPU_EXTRA_DEV\)/{p=1} p{print} p&&/^endif\(\)/{exit}' "$CML" > "$OUT/extra_defs.cmake"
grep -q '^set(_PPU_EXTRA_HOST)' "$OUT/extra_defs.cmake" || {
  echo "  [FAIL] gen_moe_units_check: could not slice PPU_EXTRA_DEFS handling out of $CML"; exit 1; }
awk '/^set\(MOE_CORES .*CACHE STRING/{p=1} p{print} p&&/^endif\(\)/{exit}' "$CML" > "$OUT/moe_cores.cmake"
grep -q 'MOE_CORES must be a positive integer' "$OUT/moe_cores.cmake" || {
  echo "  [FAIL] gen_moe_units_check: could not slice early MOE_CORES validation out of $CML"; exit 1; }

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
  echo "set(MOE_CORES \"$MOE_CHECK_CORES\" CACHE STRING \"\")"
  cat "$OUT/moe_cores.cmake"
  echo "set(MOE_FORMATS \"$MOE_CHECK_FORMATS\" CACHE STRING \"\")"
  echo "set(MOE_TM_LIST \"$MOE_CHECK_TM_LIST\" CACHE STRING \"\")"
  echo "set(MOE_TN_LIST \"$MOE_CHECK_TN_LIST\" CACHE STRING \"\")"
  echo "set(MOE_WM_LIST \"$MOE_CHECK_WM_LIST\" CACHE STRING \"\")"
  echo "set(MOE_STAGES \"$MOE_CHECK_STAGES\" CACHE STRING \"\")"
  echo "set(_CHECK_MOE_FORMATS \"$MOE_CHECK_FORMATS\")"
  echo "set(_CHECK_MOE_TM_LIST \"$MOE_CHECK_TM_LIST\")"
  echo "set(_CHECK_MOE_TN_LIST \"$MOE_CHECK_TN_LIST\")"
  echo "set(_CHECK_MOE_WM_LIST \"$MOE_CHECK_WM_LIST\")"
  echo "set(_CHECK_MOE_STAGES \"$MOE_CHECK_STAGES\")"
  echo 'set(PPU_EXTRA_DEFS "$ENV{PPU_DEFS}")'
  echo 'set(_CHECK_PPU_EXTRA_DEFS "$ENV{PPU_DEFS}")'
  cat "$OUT/extra_defs.cmake"
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
          # Preserve the raw count while changing the multiset. A count-only gate and a set-only gate each miss half
          # of this plant; raw==unique==expected is the required invariant.
          3) sed 's/^  list(REMOVE_DUPLICATES _MOE_SHAPES)$/&\n  list(GET _MOE_SHAPES 1 _planted_dup)\n  list(REMOVE_AT _MOE_SHAPES 0)\n  list(APPEND _MOE_SHAPES "${_planted_dup}")/' ;;
          # The collective reads PPU_B_CHUNK once per TU. Make every per-shape request say 1 while the bc0 TU policy
          # remains 0; the content gate must reject it before a compile can silently instantiate the wrong policy.
          4) sed 's/"#define UNIT_B_CHUNK ${_bc}\\n"/"#define UNIT_B_CHUNK 1\\n"/' ;;
          # Accept the requested TM/TN/WM values but make their row predicate unconditional. The independent
          # expected-name projection below must observe the extra wrappers.
          5) sed 's/list(FIND _MOE_${_field}_LIST "${_r_${_field_lc}}" _axis_idx)/set(_axis_idx 0) # planted bypass/' ;;
          # Keep the stage row projection but lose one half of the target compile contract. Shape counts alone
          # cannot see this; the exact DEV/HOST flag assertion below must.
          6) sed 's/list(APPEND _PPU_EXTRA_DEV "${_stage_dev}")/# planted missing device flag/' ;;
          # A wrapper that falls back to T (or any other value) is a live kernel with the wrong resident layout.
          # Keep the name's table-derived A intact and corrupt the compile macro so the content check must catch it.
          7) sed 's/"#define UNIT_ARTIFACT_TILEK ${_artifact_tile_k}\\n"/"#define UNIT_ARTIFACT_TILEK 999\\n"/' ;;
          *) cat ;;
        esac; }
  cat <<'ASSERT'
# Compare an actual NAME multiset to an expected unique set. Checking raw length before de-duplicating is what rejects
# "drop one + duplicate another", which a plain set comparison cannot see.
function(qz_assert_exact_names LABEL EXPECTED_VAR ACTUAL_VAR)
  set(_expected "${${EXPECTED_VAR}}")
  set(_actual "${${ACTUAL_VAR}}")
  list(LENGTH _expected _ne)
  list(LENGTH _actual _na)
  set(_unique "${_actual}")
  list(REMOVE_DUPLICATES _unique)
  list(LENGTH _unique _nu)
  set(_missing "${_expected}")
  if(_unique)
    list(REMOVE_ITEM _missing ${_unique})
  endif()
  set(_extra "${_unique}")
  if(_expected)
    list(REMOVE_ITEM _extra ${_expected})
  endif()
  if(NOT _na EQUAL _ne OR NOT _nu EQUAL _ne OR _missing OR _extra)
    list(LENGTH _missing _nmiss)
    list(LENGTH _extra _nextra)
    set(_first_missing "<none>")
    set(_first_extra "<none>")
    if(_missing)
      list(GET _missing 0 _first_missing)
    endif()
    if(_extra)
      list(GET _extra 0 _first_extra)
    endif()
    message(FATAL_ERROR "${LABEL}: expected ${_ne} unique shapes, got ${_na} rows / ${_nu} unique; ${_nmiss} missing (${_first_missing}), ${_nextra} extra (${_first_extra})")
  endif()
endfunction()

# Derive one band's shape set from its tracked tables by a parser independent of qz_parse_tactic_xmacro, then inspect
# the generated batch CONTENT. A .cu basename is now a physical TU, not a shape identity.
function(qz_assert_moe_band LABEL TABLE_SUFFIX GEN_DIR GEN_TU_N GEN_SRCS EXPECT_MMAX EXPECT_DECODE EXPECT_BAND EXPECT_BENCH)
  set(_exp_names "")
  set(_exp_rows 0)
  set(_exp_formats "")
  foreach(_row IN LISTS _MOE_ALL_FORMATS)
    string(REPLACE "|" ";" _f "${_row}")
    list(GET _f 0 _nm)
    list(GET _f 7 _emit)
    if(NOT "${_CHECK_MOE_FORMATS}" STREQUAL "")
      list(FIND _CHECK_MOE_FORMATS "${_nm}" _fmt_idx)
      if(_fmt_idx EQUAL -1)
        continue()
      endif()
    endif()
    list(APPEND _exp_formats "${_nm}")
    set(_hit "")
    foreach(_d IN LISTS QZ_SRC_DIRS)
      if(EXISTS "${_d}/lowbit_grouped_${_emit}${TABLE_SUFFIX}_configs.inc")
        list(APPEND _hit "${_d}/lowbit_grouped_${_emit}${TABLE_SUFFIX}_configs.inc")
      endif()
    endforeach()
    list(LENGTH _hit _nhit)
    if(NOT _nhit EQUAL 1)
      message(FATAL_ERROR "${LABEL} table for ${_nm}: lowbit_grouped_${_emit}${TABLE_SUFFIX}_configs.inc has ${_nhit} candidates, expected 1")
    endif()
    list(GET _hit 0 _t)
    file(READ "${_t}" _txt)
    string(REGEX MATCHALL "X\\([0-9,]+,B\\)" _hits "${_txt}")
    list(LENGTH _hits _nrows)
    string(TOUPPER "${_emit}" _uc)
    file(STRINGS "${_t}" _artifact_defs
         REGEX "^[ \t]*#define[ \t]+LOWBIT_GROUPED_${_uc}_CFG_ARTIFACT_TILEK([ \t]|$)")
    list(LENGTH _artifact_defs _nartifact_defs)
    if(NOT _nartifact_defs EQUAL 1)
      message(FATAL_ERROR "${_t}: independently found ${_nartifact_defs} ArtifactTileK definitions, expected one")
    endif()
    list(GET _artifact_defs 0 _artifact_def)
    if(NOT _artifact_def MATCHES
       "^[ \t]*#define[ \t]+LOWBIT_GROUPED_${_uc}_CFG_ARTIFACT_TILEK[ \t]+([1-9][0-9]*)[ \t]*$")
      message(FATAL_ERROR "${_t}: malformed independent ArtifactTileK definition '${_artifact_def}'")
    endif()
    set(_artifact_tile_k "${CMAKE_MATCH_1}")
    if(NOT _txt MATCHES "#define[ \t]+LOWBIT_GROUPED_${_uc}_CFG_ROWS[ \t]+([0-9]+)")
      message(FATAL_ERROR "${_t}: no declared row count")
    endif()
    if(NOT _nrows EQUAL CMAKE_MATCH_1)
      message(FATAL_ERROR "${_t}: independent parser matched ${_nrows} rows, table declares ${CMAKE_MATCH_1}")
    endif()
    math(EXPR _exp_rows "${_exp_rows} + ${_nrows}")
    foreach(_h IN LISTS _hits)
      string(REGEX REPLACE "^X\\(|,B\\)$" "" _h "${_h}")
      string(REPLACE "," ";" _hf "${_h}")
      list(GET _hf 0 _tm)
      list(GET _hf 1 _tn)
      list(GET _hf 2 _tk)
      list(GET _hf 3 _wm)
      list(GET _hf 4 _wn)
      list(GET _hf 5 _stage)
      list(GET _hf 6 _bc)
      set(_selected TRUE)
      foreach(_axis TM TN WM)
        if(NOT "${_CHECK_MOE_${_axis}_LIST}" STREQUAL "")
          string(TOLOWER "${_axis}" _axis_lc)
          list(FIND _CHECK_MOE_${_axis}_LIST "${_${_axis_lc}}" _axis_idx)
          if(_axis_idx EQUAL -1)
            set(_selected FALSE)
          endif()
        endif()
      endforeach()
      if(NOT "${_CHECK_MOE_STAGES}" STREQUAL "")
        list(FIND _CHECK_MOE_STAGES "${_stage}" _stage_idx)
        if(_stage_idx EQUAL -1)
          set(_selected FALSE)
        endif()
      endif()
      if(NOT _selected)
        continue()
      endif()
      list(APPEND _exp_names
           "moe_unit_${_nm}_tn${_tn}_wn${_wn}_tm${_tm}_wm${_wm}_tk${_tk}_a${_artifact_tile_k}_bc${_bc}")
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES _exp_names)
  list(LENGTH _exp_names _exp)
  list(LENGTH _exp_formats _nfmt)
  set(_exp_bc0 0)
  set(_exp_bc1 0)
  foreach(_name IN LISTS _exp_names)
    if(_name MATCHES "_bc0$")
      math(EXPR _exp_bc0 "${_exp_bc0} + 1")
    elseif(_name MATCHES "_bc1$")
      math(EXPR _exp_bc1 "${_exp_bc1} + 1")
    else()
      message(FATAL_ERROR "${LABEL}: expected name has no boolean BC suffix: ${_name}")
    endif()
  endforeach()

  # Independently recompute the smallest legal batch. For cores below the number of non-empty BC buckets, the minimum
  # remains one TU per bucket and therefore takes more than one wave.
  set(_batch 1)
  set(_max_bucket ${_exp_bc0})
  if(_exp_bc1 GREATER _max_bucket)
    set(_max_bucket ${_exp_bc1})
  endif()
  while(TRUE)
    math(EXPR _exp_tus "(${_exp_bc0} + ${_batch} - 1) / ${_batch} + (${_exp_bc1} + ${_batch} - 1) / ${_batch}")
    if(_exp_tus LESS_EQUAL MOE_CORES OR _batch EQUAL _max_bucket)
      break()
    endif()
    math(EXPR _batch "${_batch} + 1")
  endwhile()

  # Crossing a format boundary is required only when independent per-format rounding would cost a TU. With a
  # restricted axis, every format can end exactly on this batch size; then a correct global batcher has no boundary
  # to cross. The old unconditional assertion called that case a failure even though its TU count was already the
  # global ceil. Derive both counts from the independently parsed names and ask for a crossing only when they differ.
  set(_per_format_tus 0)
  foreach(_fmt IN LISTS _exp_formats)
    set(_fmt_bc0 0)
    set(_fmt_bc1 0)
    foreach(_name IN LISTS _exp_names)
      if(_name MATCHES "^moe_unit_${_fmt}_.*_bc0$")
        math(EXPR _fmt_bc0 "${_fmt_bc0} + 1")
      elseif(_name MATCHES "^moe_unit_${_fmt}_.*_bc1$")
        math(EXPR _fmt_bc1 "${_fmt_bc1} + 1")
      endif()
    endforeach()
    math(EXPR _per_format_tus
         "${_per_format_tus} + (${_fmt_bc0} + ${_batch} - 1) / ${_batch} + (${_fmt_bc1} + ${_batch} - 1) / ${_batch}")
  endforeach()

  file(GLOB _gen "${GEN_DIR}/moe_batch_bc*.cu")
  list(SORT _gen)
  list(LENGTH _gen _ngen)
  set(_listed_sources "${GEN_SRCS}")
  qz_assert_exact_names("${LABEL} returned source list" _gen _listed_sources)
  set(_got_names "")
  set(_got_bc0 0)
  set(_got_bc1 0)
  set(_got_tus_bc0 0)
  set(_got_tus_bc1 0)
  set(_partial_bc0 0)
  set(_partial_bc1 0)
  set(_cross_format FALSE)
  foreach(_g IN LISTS _gen)
    file(READ "${_g}" _src)
    string(REGEX MATCHALL "#define[ \t]+PPU_B_CHUNK[ \t]+[01]" _policy_defs "${_src}")
    list(LENGTH _policy_defs _npolicy)
    if(NOT _npolicy EQUAL 1)
      message(FATAL_ERROR "${LABEL}: ${_g} has ${_npolicy} TU-level PPU_B_CHUNK definitions, expected one")
    endif()
    string(REGEX REPLACE ".*[ \t]([01])$" "\\1" _tu_bc "${_policy_defs}")
    string(REGEX MATCHALL "#define[ \t]+UNIT_FN[ \t]+moe_unit_[A-Za-z0-9_]+" _fn_defs "${_src}")
    string(REGEX MATCHALL "#define[ \t]+UNIT_B_CHUNK[ \t]+[01]" _unit_bc_defs "${_src}")
    string(REGEX MATCHALL "#define[ \t]+UNIT_ARTIFACT_TILEK[ \t]+[1-9][0-9]*" _unit_artifact_defs "${_src}")
    string(REGEX MATCHALL "#include[ \t]+\"moe_bench_unit.inc\"" _unit_includes "${_src}")
    string(REGEX MATCHALL "#include[ \t]+\"moe_bench_band.inc\"" _band_includes "${_src}")
    list(LENGTH _fn_defs _nfns)
    list(LENGTH _unit_bc_defs _nubc)
    list(LENGTH _unit_artifact_defs _nua)
    list(LENGTH _unit_includes _ninc)
    list(LENGTH _band_includes _nband)
    if(_nfns LESS 1 OR _nfns GREATER _batch OR NOT _nubc EQUAL _nfns OR NOT _nua EQUAL _nfns OR
       NOT _ninc EQUAL _nfns OR NOT _nband EQUAL 1)
      message(FATAL_ERROR "${LABEL}: ${_g} has shapes=${_nfns}, UNIT_B_CHUNK=${_nubc}, UNIT_ARTIFACT_TILEK=${_nua}, unit includes=${_ninc}, band includes=${_nband}; expected 1..${_batch} and one field/include per shape")
    endif()
    if(_nfns GREATER 0)
      math(EXPR _last_unit "${_nfns} - 1")
      foreach(_unit_idx RANGE 0 ${_last_unit})
        list(GET _fn_defs ${_unit_idx} _fn_line)
        list(GET _unit_artifact_defs ${_unit_idx} _artifact_line)
        string(REGEX REPLACE ".*[ \t]" "" _unit_fn "${_fn_line}")
        string(REGEX REPLACE ".*[ \t]" "" _unit_artifact "${_artifact_line}")
        if(NOT _unit_fn MATCHES "_a${_unit_artifact}_bc[01]$")
          message(FATAL_ERROR
            "${LABEL}: ${_g} gives ${_unit_fn} UNIT_ARTIFACT_TILEK=${_unit_artifact}, which disagrees with its table-derived identity")
        endif()
      endforeach()
    endif()
    set(_unit_formats "")
    string(REGEX MATCHALL "#define[ \t]+UNIT_NAME[ \t]+[A-Za-z0-9_]+" _name_defs "${_src}")
    foreach(_line IN LISTS _name_defs)
      string(REGEX REPLACE ".*[ \t]" "" _fmt "${_line}")
      list(APPEND _unit_formats "${_fmt}")
    endforeach()
    list(REMOVE_DUPLICATES _unit_formats)
    list(LENGTH _unit_formats _nunit_formats)
    if(_nunit_formats GREATER 1)
      set(_cross_format TRUE)
    endif()
    foreach(_line IN LISTS _fn_defs)
      string(REGEX REPLACE ".*[ \t]" "" _name "${_line}")
      list(APPEND _got_names "${_name}")
    endforeach()
    foreach(_line IN LISTS _unit_bc_defs)
      string(REGEX REPLACE ".*[ \t]" "" _shape_bc "${_line}")
      if(NOT _shape_bc STREQUAL _tu_bc)
        message(FATAL_ERROR "${LABEL}: ${_g} mixes UNIT_B_CHUNK=${_shape_bc} into PPU_B_CHUNK=${_tu_bc}")
      endif()
    endforeach()
    if(_tu_bc EQUAL 0)
      math(EXPR _got_bc0 "${_got_bc0} + ${_nfns}")
      math(EXPR _got_tus_bc0 "${_got_tus_bc0} + 1")
      if(_nfns LESS _batch)
        math(EXPR _partial_bc0 "${_partial_bc0} + 1")
      endif()
    else()
      math(EXPR _got_bc1 "${_got_bc1} + ${_nfns}")
      math(EXPR _got_tus_bc1 "${_got_tus_bc1} + 1")
      if(_nfns LESS _batch)
        math(EXPR _partial_bc1 "${_partial_bc1} + 1")
      endif()
    endif()
  endforeach()

  qz_assert_exact_names("${LABEL} generated wrappers" _exp_names _got_names)
  math(EXPR _exp_tus_bc0 "(${_exp_bc0} + ${_batch} - 1) / ${_batch}")
  math(EXPR _exp_tus_bc1 "(${_exp_bc1} + ${_batch} - 1) / ${_batch}")
  math(EXPR _rem_bc0 "${_exp_bc0} % ${_batch}")
  math(EXPR _rem_bc1 "${_exp_bc1} % ${_batch}")
  set(_exp_partial_bc0 0)
  set(_exp_partial_bc1 0)
  if(_rem_bc0 GREATER 0)
    set(_exp_partial_bc0 1)
  endif()
  if(_rem_bc1 GREATER 0)
    set(_exp_partial_bc1 1)
  endif()
  if(NOT _ngen EQUAL _exp_tus OR NOT GEN_TU_N EQUAL _exp_tus OR
     NOT _got_bc0 EQUAL _exp_bc0 OR NOT _got_bc1 EQUAL _exp_bc1 OR
     NOT _got_tus_bc0 EQUAL _exp_tus_bc0 OR NOT _got_tus_bc1 EQUAL _exp_tus_bc1 OR
     NOT _partial_bc0 EQUAL _exp_partial_bc0 OR NOT _partial_bc1 EQUAL _exp_partial_bc1)
    message(FATAL_ERROR "${LABEL}: expected shapes bc=${_exp_bc0}/${_exp_bc1}, TUs=${_exp_tus_bc0}/${_exp_tus_bc1}; got shapes ${_got_bc0}/${_got_bc1}, disk TUs ${_got_tus_bc0}/${_got_tus_bc1}, generator said ${GEN_TU_N}")
  endif()
  if(_nfmt GREATER 1 AND _batch GREATER 1 AND _per_format_tus GREATER _exp_tus AND NOT _cross_format)
    message(FATAL_ERROR "${LABEL}: no batch crosses a format boundary; per-format rounding silently costs extra TUs")
  endif()

  file(READ "${GEN_DIR}/moe_bench_units.inc" _d)
  string(REGEX MATCHALL "\nvoid[ \t]+moe_unit_[A-Za-z0-9_]+" _decl_lines "${_d}")
  set(_decl_names "")
  foreach(_line IN LISTS _decl_lines)
    string(REGEX REPLACE ".*[ \t]" "" _name "${_line}")
    list(APPEND _decl_names "${_name}")
  endforeach()
  string(REGEX MATCHALL "\n[ ][ ]moe_unit_[A-Za-z0-9_]+\\(bd" _call_lines "${_d}")
  set(_call_names "")
  foreach(_line IN LISTS _call_lines)
    string(REGEX REPLACE "^\n[ ]+" "" _name "${_line}")
    string(REGEX REPLACE "\\(bd$" "" _name "${_name}")
    list(APPEND _call_names "${_name}")
  endforeach()
  qz_assert_exact_names("${LABEL} dispatcher declarations" _exp_names _decl_names)
  qz_assert_exact_names("${LABEL} dispatcher calls" _exp_names _call_names)
  foreach(_need "MOE_UNIT_COUNT ${_exp}" "MOE_TU_COUNT ${_exp_tus}" "MOE_FMT_COUNT ${_nfmt}" "moe_fmt_names")
    string(FIND "${_d}" "${_need}" _found)
    if(_found EQUAL -1)
      message(FATAL_ERROR "${LABEL}: dispatcher is missing '${_need}'")
    endif()
  endforeach()

  file(READ "${GEN_DIR}/moe_bench_band.inc" _band)
  foreach(_need "MOE_TABLE_DECODE ${EXPECT_DECODE}" "MOE_TABLE_M_MAX ${EXPECT_MMAX}" "MOE_TABLE_BAND_STR \"${EXPECT_BAND}\"" "MOE_TABLE_BENCH_STR \"${EXPECT_BENCH}\"")
    string(FIND "${_band}" "${_need}" _found)
    if(_found EQUAL -1)
      message(FATAL_ERROR "${LABEL}: band metadata is missing '${_need}'")
    endif()
  endforeach()
  message(STATUS "OK ${LABEL}: ${_exp} shapes (bc0=${_exp_bc0}, bc1=${_exp_bc1}) from ${_exp_rows} rows -> ${_ngen} TUs at batch=${_batch}; dispatcher exact")
endfunction()

# MOE_STAGES is not part of a wrapper name, so the exact generated-name comparison above cannot prove it had an
# effect. Inspect the two flag channels the real target wrapper consumes. A missing device flag would compile all
# stages under hgcc; a missing host flag would make the host and device halves disagree about the same header.
set(_expected_stage_dev "")
set(_expected_stage_host "")
foreach(_stage IN LISTS _CHECK_MOE_STAGES)
  string(CONCAT _expected_dev_flag "-D" "MOE_STAGES_${_stage}")
  list(APPEND _expected_stage_dev "${_expected_dev_flag}")
  list(APPEND _expected_stage_host "MOE_STAGES_${_stage}")
endforeach()
string(REPLACE " " ";" _legacy_stage_defs "${_CHECK_PPU_EXTRA_DEFS}")
foreach(_legacy IN LISTS _legacy_stage_defs)
  if(_legacy MATCHES "^MOE_STAGES_(2|3|4|6|8|12)(=.*)?$")
    string(CONCAT _expected_legacy_dev "-D" "${_legacy}")
    list(APPEND _expected_stage_dev "${_expected_legacy_dev}")
    list(APPEND _expected_stage_host "${_legacy}")
  endif()
endforeach()
set(_actual_stage_dev "")
foreach(_flag IN LISTS _PPU_EXTRA_DEV)
  if(_flag MATCHES "^-DMOE_STAGES_")
    list(APPEND _actual_stage_dev "${_flag}")
  endif()
endforeach()
set(_actual_stage_host "")
foreach(_flag IN LISTS _PPU_EXTRA_HOST)
  if(_flag MATCHES "^MOE_STAGES_")
    list(APPEND _actual_stage_host "${_flag}")
  endif()
endforeach()
foreach(_pair "_expected_stage_dev|_actual_stage_dev|device" "_expected_stage_host|_actual_stage_host|host")
  string(REPLACE "|" ";" _fields "${_pair}")
  list(GET _fields 0 _expected_var)
  list(GET _fields 1 _actual_var)
  list(GET _fields 2 _channel)
  set(_expected "${${_expected_var}}")
  set(_actual "${${_actual_var}}")
  list(SORT _expected)
  list(SORT _actual)
  if(NOT "${_actual}" STREQUAL "${_expected}")
    message(FATAL_ERROR "MOE_STAGES ${_channel} flags: expected '${_expected}', got '${_actual}'")
  endif()
endforeach()

qz_assert_moe_band(full "" "${_MOE_GEN_DIR}" "${_MOE_UNIT_N}" "${_MOE_UNIT_SRCS}" 0 0 full lowbit_moe)
qz_assert_moe_band(decode _decode "${_MOE_DECODE_GEN_DIR}" "${_MOE_DECODE_UNIT_N}" "${_MOE_DECODE_UNIT_SRCS}" 3 1 decode lowbit_moe_decode)
ASSERT
} > "$GEN"

log="$OUT/log"
if ! cmake -P "$GEN" > "$log" 2>&1; then
  if [ -n "${BAD:-}" ]; then
    case "$BAD" in
      1) _reason_re='_MOE_FORMATS row.*fields, expected' ;;
      2|3) _reason_re='generated wrappers' ;;
      5) _reason_re='one field/include per shape' ;;
      4) _reason_re='mixes UNIT_B_CHUNK' ;;
      6) _reason_re='MOE_STAGES device flags' ;;
      7) _reason_re='UNIT_ARTIFACT_TILEK=.*disagrees' ;;
      *) _reason_re='a^' ;;
    esac
    if ! _reason="$(grep -m1 -E "$_reason_re" "$log")"; then
      echo "  [FAIL] gen_moe_units_check (negative control BAD=$BAD): rejected for the wrong reason"
      sed 's/^/           /' "$log"
      exit 1
    fi
    echo "  [ok]   gen_moe_units_check (negative control BAD=$BAD): the generator was REJECTED"
    printf '           %s\n' "$_reason"
    exit 0
  fi
  echo "  [FAIL] gen_moe_units_check: the generator errored"; sed 's/^/           /' "$log"; exit 1
fi
if [ -n "${BAD:-}" ]; then
  echo "  [FAIL] gen_moe_units_check (negative control BAD=$BAD): a broken generator was ACCEPTED"
  sed 's/^/           /' "$log"; exit 1
fi
sed 's/^/           /' "$log"

# Compile REAL generated multi-include sources against a tiny host stub. The ordinary local syntax gate uses a zero-unit
# dispatcher and therefore cannot see helper-name collisions or a UNIT_* macro leaking from one shape into the next.
fixture="$OUT/reentrant"
mkdir -p "$fixture/quactlize_extensions/cutlass/gemm/collective"
cp "$HERE/../../benchmarks/moe_bench_unit.inc" "$fixture/moe_bench_unit.inc"
touch "$fixture/quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
touch "$fixture/quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"
cat >"$fixture/lowbit_moe_bench.hpp" <<'STUB'
#pragma once
struct Band {};
struct Best {};
inline void moe_chunk_vote(int) {}
#define MOE1(...) do {} while (0)
#define MOE2(...) do {} while (0)
STUB
cross=""
bc1=""
for src in "$OUT"/moe_units/moe_batch_bc0_*.cu; do
  [ "$(awk '/^#define UNIT_NAME/{print $3}' "$src" | sort -u | wc -l)" -gt 1 ] && { cross="$src"; break; }
done
for src in "$OUT"/moe_units/moe_batch_bc1_*.cu; do [ -f "$src" ] && { bc1="$src"; break; }; done
compile_srcs=()
if [ -z "$MOE_CHECK_FORMATS$MOE_CHECK_TM_LIST$MOE_CHECK_TN_LIST$MOE_CHECK_WM_LIST$MOE_CHECK_STAGES" ]; then
  [ -n "$cross" ] && [ -n "$bc1" ] || { echo "  [FAIL] default sweep has no cross-format bc0 batch or bc1 batch to compile"; exit 1; }
  compile_srcs=("$cross" "$bc1")
else
  for src in "$OUT"/moe_units/moe_batch_bc0_*.cu "$OUT"/moe_units/moe_batch_bc1_*.cu; do
    [ -f "$src" ] || continue
    compile_srcs+=("$src")
    [ "${#compile_srcs[@]}" -eq 2 ] && break
  done
  [ "${#compile_srcs[@]}" -gt 0 ] || { echo "  [FAIL] restricted sweep emitted no TU to compile"; exit 1; }
fi
for src in "${compile_srcs[@]}"; do
  c++ -std=c++17 -x c++ -I"$fixture" -c "$src" -o "$fixture/$(basename "$src").o" || {
    echo "  [FAIL] generated multi-shape TU does not compile with the re-entrant unit fixture: $src"; exit 1; }
done
if [ -n "$cross" ] && [ -n "$bc1" ]; then
  echo "           -- re-entrant generated TU compile: cross-format bc0 + bc1 OK"
else
  echo "           -- restricted generated TU compile: ${#compile_srcs[@]} selected batch(es) OK"
fi
