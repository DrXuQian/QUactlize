#!/usr/bin/env bash
# Build the actlize (PPU cutlass3) W4A16 gs=128 comparison bench.
#
# WHY THIS IS NOT IN THE MARLIN Makefile. actlize does not build with the bare `nvcc` (no -arch) that the
# marlin kernels use. Its toolchain is the PPU SDK: hgcc as the device compiler (-arch=ppu_10), g++ as host,
# linking libhggc_wrapper / libhggcrt1 / libhggc, with -DSWITCH_TO_HGGCRT -DCUTLASS_USE_PACKED_TUPLE=1. All of
# that is set up by third_party/actlize/cmake/PPUToolchain.cmake, so we drive actlize's own cmake.
#
# WHY THE OVERLAY. actlize's examples are an explicit foreach() list in examples/CMakeLists.txt that
# add_subdirectory's each one; there is no out-of-tree example hook. The least-invasive way to build our .cu
# through the *proven* example machinery is to drop it in as a new example dir and append it to that list.
# We do it as untracked files + a restorable one-line edit, so the submodule's tracked content is unchanged
# (the script restores examples/CMakeLists.txt at the end).
#
# Prereq: PPU_SDK=<path with bin/hgcc> (or PPU_HOME). ppu001 == ppu0010 == ACOMPUTE 10000.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # the repository root
ACTLIZE="$(cd "$HERE/third_party/actlize" && pwd)"
# THE SOURCE IS NO LONGER ONE FLAT DIRECTORY. The tree separates kernels, tests and benchmarks, and the overlay
# FLATTENS all of them into one example directory -- which is only sound because no basename repeats across them
# (45 files, 45 distinct names; the completeness check below is what keeps that true as files are added).
# gemv_lowbit stays a real subdirectory because sources say #include "gemv_lowbit/gemv_launcher.hpp": the directory
# name is load-bearing, so renaming it would have been churn for nothing.
_src_dirs=(quactlize/include tests benchmarks)
_subdir_src="quactlize/include/gemv_lowbit"
EX_NAME="99_kernels_w4a16_compare"
EX_DIR="$ACTLIZE/examples/$EX_NAME"
EX_LIST="$ACTLIZE/examples/CMakeLists.txt"
ARCH="${PPU_ARCHS:-ppu0010}"
# Default to this box's SDK location; override with PPU_SDK=<path> (or PPU_HOME) if it moves.
PPU_SDK_ROOT="${PPU_SDK:-${PPU_HOME:-/sim/eec/shared/junfu.qx/PPU_SDK}}"

# THE OVERLAY MANIFEST, PRODUCED ONCE. Everything that copies files, and everything that checks what would be
# copied, reads this function -- there is no second enumeration of the globs.
#
# It emits one line per file, either "<abs path>" for a file that lands at the top of the example directory, or
# "<dest>|<abs path>" where <dest> ends in "/" for a subdirectory or names a rename. That shape exists so the
# checker can materialise the overlay exactly, without knowing the rules.
#
# WHY IT IS A FUNCTION. The first version had --print-overlay enumerate the globs a second time, next to the copy
# path that enumerates them for real. Review pointed out that the gate then proved the PRINT implementation and said
# nothing about the COPY implementation -- the exact duplication the gate was introduced to remove, reintroduced by
# the gate. The extension whitelist also appeared three times, and _overlay_dirs was declared twice.
#
# *.inc is in the whitelist because the MoE sweep's generated units all #include moe_bench_unit.inc. Leaving it out
# did not fail here: it failed 100+ lines into hgcc as "fatal error: moe_bench_unit.inc: No such file or directory",
# once per generated unit.
_OVERLAY_EXTS=(cu cpp cuh hpp h inc)

# $1 = directory (absolute), $2 = optional "dest/" prefix, $3.. = extensions to use (defaults to _OVERLAY_EXTS)
_emit_dir() {
  local dir="$1" dest="$2"; shift 2
  local exts=("$@"); [ ${#exts[@]} -eq 0 ] && exts=("${_OVERLAY_EXTS[@]}")
  [ -d "$dir" ] || return 0
  local e f
  shopt -s nullglob
  for e in "${exts[@]}"; do
    for f in "$dir"/*."$e"; do printf '%s%s\n' "${dest:+$dest|}" "$f"; done
  done
  shopt -u nullglob
}

overlay_manifest() {
  local _sd
  for _sd in "${_src_dirs[@]}"; do _emit_dir "$HERE/$_sd" ""; done
  # dev/'s TOP LEVEL, when it exists. These are DEVICE probes -- swzl_ldmatrix_probe reads the hardware swizzle, the
  # ablations and sweeps run on the accelerator -- so they belong on the box, and before the reorganisation they sat
  # alongside everything else and were overlaid as a matter of course. The move to dev/ dropped them from the overlay
  # while CMakeLists.txt kept registering them, and a missing source is a CONFIGURE-time error, so cmake failed for
  # EVERY target: eleven build variants all reporting "the macro did not reach the device compile".
  #
  # NO .cpp here, and that asymmetry is deliberate rather than an oversight: dev/ holds .cu probes only. It is called
  # out because a checker that assumed the shared list would pass a dev/*.cpp the overlay would never copy.
  _emit_dir "$HERE/dev" "" cu cuh hpp h inc
  # A library split across a subdirectory has to come whole; most subdirectories (fold_derivation/, low_bit/) are
  # host-only harnesses that must NOT reach the box, which is why this is one named path and not a recursive walk.
  _emit_dir "$HERE/$_subdir_src" "gemv_lowbit/"
  printf 'CMakeLists.txt|%s\n' "$HERE/quactlize/csrc/CMakeLists.txt.in"
}

# --print-overlay: the EXACT list of files the overlay would copy, one per line, then exit. This exists so that
# nothing else has to MODEL this script. dev/fold_derivation/overlay_targets_check.py used to reconstruct the globs
# in python and got them subtly wrong -- it applied the common extension list to dev/, which this script's dev glob
# does not, and it parsed `_src_dirs=(...)` with python .split(), so quoting the entries would have broken it. Both
# were found by review rather than by use, which is the point: a second implementation of a list is a second list.
#
# Deliberately BEFORE the SDK check. The cheap local checks below are the ones a developer without a PPU can run,
# and gating them on hgcc means they never run anywhere they would be useful.
if [ "${1:-}" = "--print-overlay" ]; then
  overlay_manifest
  exit 0
fi

# CHEAP LOCAL CHECKS FIRST -- and BEFORE THE SDK GATE, which is the only thing that makes "first" true. They used to
# sit after it, so on any machine without hgcc the script exited before running a single one of them: exactly the
# machines where a local check is the only check available. Not one of them needs the SDK.
#
# because the expensive failures here are configure-time and box-only. Three target
# registrations once named a helper that does not exist; nothing caught it until cmake ran on the box, which
# costs a full pull-and-build to discover a typo.
#
# THE CHECKS ARE TOLD WHERE THINGS ARE rather than deriving it. All three assumed the pre-reorganisation layout, in
# which this script's directory held CMakeLists.txt, the headers and gemv_lowbit/ side by side. After the split into
# quactlize/include, tests/ and benchmarks/ all three failed -- loudly, each with its own self-check ("the deny list
# matches nothing", "the anchors moved"), which is the only reason the breakage was visible at all rather than three
# checks quietly passing over an empty file set. Exporting the paths from the one place that already knows them
# means they cannot drift from the overlay again.
export QUACTLIZE_ROOT="$HERE"
# dev/ IS IN THIS LIST WHEN IT EXISTS, because its top-level sources are overlaid onto the box (see the overlay step
# below). The portability check's whole job is to cover what ships; listing a narrower set than the overlay copies is
# the same drift this export was introduced to remove, one directory later.
export QUACTLIZE_SRC_DIRS="${_src_dirs[*]}$([ -d "$HERE/dev" ] && echo " dev")"
export QUACTLIZE_GEMV_DIR="$HERE/$_subdir_src"
export QUACTLIZE_CMAKE="$HERE/quactlize/csrc/CMakeLists.txt.in"
if [ -x "$HERE/dev/fold_derivation/cmake_calls_check.sh" ]; then
  "$HERE/dev/fold_derivation/cmake_calls_check.sh" || exit 1
fi
# NVIDIA-only spellings in box-built sources. The syntax check cannot see these: local nvcc has cuda_fp16.h
# whether or not -D__HGGCCC__ is passed, so a wrong-platform include parses clean and only hgcc disagrees.
if [ -x "$HERE/dev/fold_derivation/ppu_portability_check.py" ]; then
  "$HERE/dev/fold_derivation/ppu_portability_check.py" || exit 1
fi
# The unit generators, run against the axis lists rather than a written-down count. A malformed row once
# produced the RIGHT unit count out of the wrong iterations, so the configure log looked correct.
if [ -x "$HERE/dev/fold_derivation/gen_gemv_units_check.sh" ]; then
  "$HERE/dev/fold_derivation/gen_gemv_units_check.sh" || exit 1
fi
# THE ONE THAT WOULD HAVE SAVED TWO BOX ROUND-TRIPS. CMakeLists.txt names sources and this script decides which
# directories are copied; the two lists live in different files, in different languages, and nothing compared them
# until cmake ran on the accelerator. A source CMake names but the overlay lacks is a CONFIGURE-time error, so it
# fails for EVERY target at once and the message names only whichever file it tripped over first.
if [ -x "$HERE/dev/fold_derivation/overlay_targets_check.py" ]; then
  "$HERE/dev/fold_derivation/overlay_targets_check.py" || exit 1
fi
# The MoE sweep's generator, the sibling of gen_gemv_units_check above. It was called from nowhere and its path to
# CMakeLists.txt had been stale since the reorganisation -- two facts that hid each other, since an uncalled check
# cannot report a broken path.
if [ -x "$HERE/dev/fold_derivation/gen_moe_units_check.sh" ]; then
  "$HERE/dev/fold_derivation/gen_moe_units_check.sh" || exit 1
fi

# The checks above live under dev/fold_derivation and are absent from a main-branch checkout, where the `-x` test
# simply skips them. That is intended -- but it means a release build silently runs FEWER checks than a dev build, so
# say which happened rather than leaving it to be inferred from nothing being printed.
[ -d "$HERE/dev/fold_derivation" ] || echo "  NOTE: dev/fold_derivation absent (main checkout) -- generator and portability checks skipped"

if [ ! -x "$PPU_SDK_ROOT/bin/hgcc" ]; then
  echo "ERROR: hgcc not found at $PPU_SDK_ROOT/bin/hgcc. Set PPU_SDK=<path> and re-run." >&2
  exit 1
fi
export PATH="$PPU_SDK_ROOT/bin:$PATH"

cleanup() {
  # Restore ONLY the example-list registration + remove the overlay. The actlize W4A16 changes now live as real
  # commits on the DrXuQian/actlize fork (submodule branch ppu-w4a16-dev), NOT as build-time patches, so we must
  # NOT `git checkout --` the include/ files here (that would wipe uncommitted collective WIP during iteration).
  # RESTORE THE EXACT BYTES WE SAVED, not whatever git holds. `git checkout --` discards any UNCOMMITTED edit
  # someone had in that file, which is not this script's to throw away -- and on the box that file belongs to a
  # submodule someone may be working in.
  if [ -f "${_EX_LIST_BACKUP:-}" ]; then cp "$_EX_LIST_BACKUP" "$EX_LIST"; rm -f "$_EX_LIST_BACKUP"; fi
  rm -rf "$EX_DIR"
}
trap cleanup EXIT
echo "[build.sh] CUTLASS_PPU_ARCHS=$ARCH"

# NOTE: the former MoE/gs32 *.patch files are now baked into the submodule (fork ppu-w4a16-dev). No patch step.


# --- overlay our example into the actlize example tree ---
mkdir -p "$EX_DIR"
# nullglob so patterns that match nothing (e.g. no *.cpp right now) vanish instead of aborting under set -e.
# *.inc is in the list because the MoE sweep's generated units all #include moe_bench_unit.inc, and this glob is an
# EXTENSION WHITELIST: leaving it out did not fail here, it failed 100+ lines into hgcc as
# `fatal error: moe_bench_unit.inc: No such file or directory` repeated once per generated unit.
# ONE PRODUCER, ONE CONSUMER. Every path below comes from overlay_manifest above; nothing here re-enumerates.
_n_overlaid=0 _n_dev=0
while IFS= read -r _line; do
  case "$_line" in
    *"|"*) _dest="${_line%%|*}"; _src="${_line#*|}"
           case "$_dest" in
             */) mkdir -p "$EX_DIR/$_dest"; cp "$_src" "$EX_DIR/$_dest" ;;
             *)  cp "$_src" "$EX_DIR/$_dest" ;;
           esac ;;
    *)     cp "$_line" "$EX_DIR/"
           case "$_line" in "$HERE/dev/"*) _n_dev=$((_n_dev+1)) ;; esac ;;
  esac
  _n_overlaid=$((_n_overlaid+1))
done < <(overlay_manifest)
[ "$_n_dev" -gt 0 ] && echo "  overlaying $_n_dev dev/ probe source(s)"
echo "  overlay: $_n_overlaid file(s)"

# ...and then ASSERT THE WHITELIST IS COMPLETE. The criterion is GIT TRACKING, and the two earlier attempts show why it
# has to be something this specific:
#
#   * scanning the #includes of the overlaid files could not catch the bug it was written for -- the file that includes
#     moe_bench_unit.inc is a GENERATED unit, created later by cmake in the build tree, so it is not in the scan set at
#     overlay time. It passed on the real tree while the real build was broken.
#   * "every regular file here must be copied unless its extension is ignored" then failed the build on this box for
#     *.acurep -- acu reports, an untracked artifact of previous runs that does not exist in a fresh checkout, so the
#     ignore list could not have been written correctly from a clean tree.
#
# Tracked-ness is the right criterion because the box builds from COMMITTED state: what the build can possibly need is
# exactly what is committed. Artifacts (acu reports, logs, binaries, dumped weights) are untracked and skip themselves, and
# a newly added .def or .tpp is tracked the moment it is `git add`ed, which is also the moment it could reach the box.
_ignored='sh|md|py|log|bin|json|patch|pyc|txt'
_missing=""
if _tracked=$(git -C "$HERE" ls-files "${_src_dirs[@]}" 2>/dev/null) && [ -n "$_tracked" ]; then
  while IFS= read -r _p; do
    # Paths come back relative to the repository root now. Everything flattens to its basename in $EX_DIR except
    # the gemv_lowbit subdirectory, which keeps its name because the #includes say so.
    case "$_p" in
      "$_subdir_src"/*) _b="gemv_lowbit/$(basename "$_p")" ;;
      */data/*)         continue ;;                       # fixtures are read at runtime, not compiled
      *)                _b="$(basename "$_p")" ;;
    esac
    echo "$_b" | grep -qE "\.($_ignored)\$" && continue
    [ -f "$EX_DIR/$_b" ] || _missing="$_missing $_b"
  done <<< "$_tracked"
else
  echo "  NOTE: not a git checkout, skipping the overlay completeness check"
fi
if [ -n "$_missing" ]; then
  echo "  ERROR: the overlay is an extension whitelist and it dropped tracked source:$_missing"
  echo "         add the extension to _OVERLAY_EXTS above, or to _ignored if it is genuinely not needed to build."
  exit 1
fi

# AND PROVE THE OVERLAY IS THE CHECKOUT, not a survivor of an earlier run. cleanup() rm -rf's $EX_DIR on exit, so a
# stale copy should be impossible -- but a build once failed with a macro expansion from the PRE-fix header while the
# checkout had the fixed one, and from outside the box there was no way to tell "built an older commit" from "overlay
# was stale". cmp is cheap and turns that into a one-line answer.
#
# IT WALKS THE MANIFEST, not the overlay directory. The previous version took each overlaid file's BASENAME and
# searched _src_dirs for a match -- a third re-derivation of the source-to-destination mapping the manifest already
# holds exactly. It searched the wrong set: dev/ was not in _src_dirs, so the eight device probes were never
# compared at all, and the gemv_lowbit subdirectory matched only by coincidence of path shape. A check that silently
# covers less than it claims is the failure this one exists to prevent.
_stale=""
while IFS= read -r _line; do
  case "$_line" in
    *"|"*) _dest="${_line%%|*}"; _src="${_line#*|}"
           case "$_dest" in
             */) _dst="$EX_DIR/$_dest$(basename "$_src")" ;;
             *)  _dst="$EX_DIR/$_dest" ;;
           esac ;;
    *)     _src="$_line"; _dst="$EX_DIR/$(basename "$_line")" ;;
  esac
  if [ ! -f "$_dst" ]; then
    _stale="$_stale
    MISSING  $_dst  (from $_src)"
  elif ! cmp -s "$_dst" "$_src"; then
    _stale="$_stale
    DIFFERS  $_dst  vs  $_src"
  fi
done < <(overlay_manifest)
if [ -n "$_stale" ]; then
  echo "  ERROR: the overlay does not match the checkout -- the build would not compile your tree:$_stale"
  exit 1
fi


# REGISTER THE EXAMPLE IN actlize's foreach LIST. Without this the overlay is copied, cmake runs, and our
# CMakeLists is never add_subdirectory'd -- so no target is created and no message from it appears. The symptom is
# "No rule to make target test_moe_splitk_bench" together with "cmake did not report PPU_EXTRA_DEFS", which reads
# like a macro-plumbing problem and is not one.
#
# This step existed with its own verification and I DELETED IT while collapsing the overlay into one manifest
# function -- the splice that replaced the copy path took the registration with it. cleanup() restores the file on
# exit, so the edit is not persistent.
#
# dev/fold_derivation/overlay_targets_check.py cannot catch this: it runs cmake on the overlay DIRECTLY, which
# proves the CMakeLists configures and says nothing about whether anything reaches it. The grep below is the only
# check of that, which is why it is a hard failure and not a warning.
_EX_LIST_BACKUP="$(mktemp)"; cp "$EX_LIST" "$_EX_LIST_BACKUP"
# EXACT LIST ENTRY, not a substring anywhere in the file. The name appearing in a comment, or outside the foreach
# block, would satisfy a bare grep and register nothing.
if ! grep -qE "^[[:space:]]*$EX_NAME[[:space:]]*$" "$EX_LIST"; then
  # insert just before the closing paren of the foreach(EXAMPLE ... ) block that ends with 16_ppu_mixed_dtype_gemm
  sed -i "s|^\( *16_ppu_mixed_dtype_gemm\)\$|\1\n  $EX_NAME|" "$EX_LIST"
fi
grep -qE "^[[:space:]]*$EX_NAME[[:space:]]*$" "$EX_LIST" || { echo "ERROR: failed to register example as a list entry in $EX_LIST" >&2; exit 1; }

# --- tile/warp/stages tuning: forward from the environment (defaults match the stock example) ---
TILE_M="${TILE_M:-32}"; TILE_N="${TILE_N:-32}"; WARP_M="${WARP_M:-16}"; WARP_N="${WARP_N:-16}"; STAGES="${STAGES:-3}"
QUANT="${QUANT:-int4}"   # int4 (default) or uint2 -> bench_cutlass_w4a16's QuantType (W4A16 vs W2A16 perf)
TSK="${TSK:-}"           # TileShapeK override (empty = per-quant default: int2->128, int4->64). Set to force, e.g. TSK=128
echo "[build.sh] TILE=${TILE_M}x${TILE_N} WARP=${WARP_M}x${WARP_N} STAGES=${STAGES} QUANT=${QUANT} TSK=${TSK}"

# --- configure & build just our target ---
# OVERRIDABLE, because build.sh rm -rf's this and something that RUNS build.sh to check it must be able to point it
# somewhere disposable. Without the override, ci/box_build_dryrun.sh destroyed the real build tree every time the
# local tier ran -- on the box, that is someone's working build.
BUILD="${PPU_BUILD_DIR:-$ACTLIZE/build_w4a16_compare}"
# EXPLICIT SOURCE DIRECTORY. `cmake ..` only meant "the actlize root" while $BUILD was inside it; once the build
# directory became overridable, `..` resolved to wherever that happened to be. Naming the source is both correct and
# independent of where the build lands.
rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
# FORWARD THE SWEEP AXIS KNOBS. They were added to CMakeLists.txt and then not wired through here, so narrowing a sweep was
# impossible from build.sh -- the knob existed and could not be reached, which is worse than no knob because it reads as
# available. Any MOE_* variable in the environment is passed through as a cache var.
_MOE_VARS=()
for _v in MOE_TM_LIST MOE_TN_LIST MOE_WM_LIST MOE_FORMATS MOE_CORES; do
  if [ -n "${!_v:-}" ]; then _MOE_VARS+=("-D$_v=${!_v}"); echo "[build.sh] $_v=${!_v}"; fi
done
cmake "$ACTLIZE" -DPPU_SDK_ROOT="$PPU_SDK_ROOT" -DCUTLASS_PPU_ARCHS="$ARCH" \
  -DTILE_M="$TILE_M" -DTILE_N="$TILE_N" -DWARP_M="$WARP_M" -DWARP_N="$WARP_N" -DSTAGES="$STAGES" -DBENCH_QUANT="$QUANT" -DTSK="$TSK" \
  -DPPU_EXTRA_DEFS="${PPU_DEFS:-}" "${_MOE_VARS[@]+"${_MOE_VARS[@]}"}" \
  >cmake.log 2>&1 || { tail -40 cmake.log; exit 1; }
# cmake's stdout is redirected above, so its message(STATUS ...) never reaches the terminal. Surface the extra
# defines here instead -- telling someone to "check that the line appeared" when it cannot appear is worse than
# not printing it at all.
if [ -n "${PPU_DEFS:-}" ]; then
  echo "PPU_DEFS applied: $PPU_DEFS"
  grep -F "PPU_EXTRA_DEFS ->" cmake.log || echo "  WARNING: cmake did not report PPU_EXTRA_DEFS -- the defines did NOT reach the build"
fi
TARGET="${TARGET:-bench_cutlass_w4a16}"

# #13: NO SHIPPING TARGET MAY REACH THE LEGACY PACKERS. fold_derivation/legacy_pipeline.hpp still holds
# nfold_regroup_gmem and nfold_place_bits_int1_tk64, on purpose -- they are the INDEPENDENT reference the l58/l61/l64
# gates diff xplane::place_derived against, and deleting them would turn those gates into "the derived walk equals the
# derived walk". But nfold_regroup_gmem moves whole uint32 words, so one word carries one logical column, so it is
# correct only at warp-N extent 32 -- and w64x32 is worth +7 to +9 points, i.e. exactly where the tuning is going. A
# harness that picked it up would miscompute silently.
#
# The gates are host-only files under fold_derivation/ that no CMake target builds, so today nothing shipping can reach
# them. This makes that structural instead of merely true: the header can only be reached by naming its path, so grep
# for the path in the sources CMake actually compiles.
#
# TWO THINGS THIS GREP GOT WRONG ON THE FIRST WRITING, both from stating what grep emits instead of reading it:
#   * `grep -rln <dir>` prints `fold_derivation/l11...` with NO leading slash, so `grep -v "/fold_derivation/"` matched
#     nothing and every gate came back as a violation. --exclude-dir is the option that actually does this.
#   * matching the bare filename also matches the ~30 lines of COMMENTARY that explain why the packers are quarantined,
#     so the two files documenting the rule were reported as breaking it. Anchor on the #include.
_leg=$(grep -rln '^[[:space:]]*#[[:space:]]*include.*legacy_pipeline\.hpp' \
         --include=*.cu --include=*.cuh --include=*.hpp \
         --exclude-dir=fold_derivation "$(dirname "$0")" 2>/dev/null || true)
if [ -n "$_leg" ]; then
  echo "  ERROR: a CMake-built source includes fold_derivation/legacy_pipeline.hpp:"
  printf '           %s\n' $_leg
  echo "         nfold_regroup_gmem is correct only at WN=32. Use xplane::place_derived (see #13); the legacy"
  echo "         packers exist ONLY as the gates' independent reference."
  exit 1
fi

# JOBS overrides -j. Needed since the MoE bench became 9 sources: each hgcc instantiates ~42 kernels and wants GBs, and
# running all of them at once OOM-killed the front end locally -- an OOM kill looks like a compile failure with an empty
# log, which is the least informative way for a build to fail. `JOBS=4 ./build.sh` when memory is tight.
make -j"${JOBS:-$(nproc)}" "$TARGET" 2>&1 | tee make.log

# CMAKE RECEIVING THE DEFINES IS NOT THE SAME AS THIS TARGET GETTING THEM, and the difference is invisible in a perf
# number. The defines used to be attached to three targets by hand, so
#   PPU_DEFS=PPU_B_CHUNK=1 TARGET=test_q3_bconcat_bench ./build.sh
# configured cleanly, printed "PPU_DEFS applied", printed no warning, and produced a binary WITHOUT the macro. The run
# that followed compared a binary against itself and read as "the change does nothing".
#
# LOOK IN build.make, NOT make.log. The first version of this check grepped make.log and cried wolf on every build: the
# device compiles are add_custom_command with a COMMENT, so make.log holds "[100%] [hgcc] foo.cu" and never a compile
# line, meaning the grep could not succeed even when the flag WAS present. cmake's generated build.make does carry the
# full command, so that is what gets checked -- and it is checked for THIS target's directory, which is the claim that
# matters ("cmake received it" is a different and weaker claim, already covered above).
if [ -n "${PPU_DEFS:-}" ]; then
  _bm=$(find . -path "*${TARGET}.dir/build.make" | head -1)
  if [ -z "$_bm" ]; then
    echo "  WARNING: no build.make found for $TARGET -- cannot verify the defines reached it."
  else
    for _d in $PPU_DEFS; do
      # A WHOLE ARGUMENT, not a substring: -DSK_QUANT=2 must not be satisfied by -DSK_QUANT=20.
      if grep -qE -- "(^|[[:space:]])-D$(printf '%s' "$_d" | sed 's/[][\.^$*+?(){}|]/\\&/g')([[:space:]]|$)" "$_bm"; then
        echo "PPU_DEFS verified on $TARGET's compile command: -D$_d"
      else
        echo "  WARNING: -D$_d is NOT on $TARGET's compile command -- THIS BUILD DOES NOT HAVE IT."
        echo "           Any A/B against it is a binary compared with itself. (checked $_bm)"
      fi
    done
  fi
fi

BIN="$(find "$BUILD" -name "$TARGET" -type f -perm -u+x | head -1)"
echo
echo "built: $BIN"
# Per-target run hint. The old line advertised --m=/--mode= for EVERY target, but only the compare example
# parses that shape: on a positional-arg target each flag becomes atoi("--m=2048") == 0, so copying the hint
# silently selects L=0 or Mb=0 and the test passes vacuously. One such copy already cost a box round.
case "$TARGET" in
  test_moe_grouped_verify)
    echo "run:   $BIN [L] [Mb] [ragged?] [gs]      # POSITIONAL, no --flags"
    echo "       $BIN 8 1                          # Mmax==1, required by PPU_A_CUBE_H=1"
    echo "       MOEG_SMEM=1 MOEG_DUMP=/tmp/d.bin $BIN 8 1     # then MOEG_CHECK=/tmp/d.bin on the other build" ;;
  test_moe_splitk_bench|test_lowbit_moe_bench)
    echo "run:   $BIN <L> <rows> <N> <K> <gs> <mode>   # POSITIONAL; mode 3 => rows is top-k"
    echo "       $BIN 64 8 2048 2048 32 3"
    echo "       SPLITK_CFG='16x128:256 w16x16 s2' SPLITK_S=1 $BIN 64 8 2048 2048 32 3   # select one acu row" ;;
  test_gemv_perf)
    echo "run:   $BIN [shape]                      # GEMV_FMT=/GEMV_CFG= select rows" ;;
  *)
    echo "run:   $BIN --m=2048 --n=4096 --k=4096 --g=128 --mode=1 --iterations=100" ;;
esac
