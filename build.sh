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
_src_dirs=(quactlize/include quactlize/csrc/device tests benchmarks)
_subdir_src="quactlize/include/gemv_lowbit"
ARCH="${PPU_ARCHS:-ppu0010}"
# Default to this box's SDK location; override with PPU_SDK=<path> (or PPU_HOME) if it moves.

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
_OVERLAY_EXTS=(cu cpp cuh hpp h inc cmake)

# $1 = directory (absolute), $2 = optional "dest/" prefix, $3.. = extensions to use (defaults to _OVERLAY_EXTS)
_emit_dir() {
  local dir="$1" dest="$2"; shift 2
  local exts=("$@"); [ ${#exts[@]} -eq 0 ] && exts=("${_OVERLAY_EXTS[@]}")
  [ -d "$dir" ] || return 0
  local e f
  shopt -s nullglob
  for e in "${exts[@]}"; do
    for f in "$dir"/*."$e"; do
      # *_cuda_probe.* is a local host-CUDA harness by convention. It includes cuda_runtime.h and is compiled by
      # pytest with nvcc; letting it into the PPU overlay sends NVIDIA runtime calls to hgcc. Keep this in lockstep
      # with dev/fold_derivation/ppu_portability_check.py's identically named exclusion.
      case "$(basename "$f")" in *_cuda_probe.*) continue ;; esac
      printf '%s%s\n' "${dest:+$dest|}" "$f"
    done
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
  _emit_dir "$HERE/quactlize/csrc" "" cmake
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

# NO PERSONAL PATH AS A DEFAULT. This repo is published as github.com/DrXuQian/quactlize; a stranger's home
# directory baked in as the fallback is both useless to anyone else and a leak. The site-specific location goes
# in the environment (or PPU_SDK_SITE_DEFAULT for a shared machine's profile), and an unset SDK says so.
PPU_SDK_ROOT="${PPU_SDK:-${PPU_HOME:-${PPU_SDK_SITE_DEFAULT:-}}}"
# FAIL BEFORE HGCC, PART TWO: THE SUBMODULE MUST BE THE ONE THIS TREE RECORDS.
#
# `git pull` updates the gitlink and does NOT check out the submodule. So a machine that pulls a commit which
# moved actlize forward keeps compiling the OLD actlize headers against the NEW parent sources, and the failure
# surfaces as `no type named 'PipelineDriver' in CollectiveMma<...>` several template frames deep -- which reads
# as a kernel defect, not as a checkout that never happened. That is exactly what it cost on 2026-08-05: the box
# had actlize at 46a2b851 while the tree recorded f42db663, and the resulting errors were forwarded as a dense
# regression.
#
# One `git diff --submodule` would have said so, and nobody runs it before a build. So the build runs it.
if command -v git >/dev/null 2>&1 && [ -e "$HERE/.git" ]; then
  _want="$(git -C "$HERE" ls-tree HEAD third_party/actlize 2>/dev/null | awk '{print $3}')"
  _have="$(git -C "$HERE/third_party/actlize" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$_want" ] && [ "$_want" != "$_have" ]; then
    echo "[build.sh] third_party/actlize is not the commit this tree records." >&2
    echo "             recorded  ${_want}" >&2
    echo "             checked out ${_have:-<none: submodule not initialised>}" >&2
    echo "           Compiling anyway mixes new parent sources with old submodule headers, which fails deep in" >&2
    echo "           template instantiation and looks like a kernel bug. Fix the checkout:" >&2
    echo "             git -C $HERE submodule update --init --recursive" >&2
    exit 1
  fi
fi

TARGET="${TARGET:-test_lowbit_dense_bench}"

# THE FIVE-FORMAT FULL BENCH IS NOT A VIABLE LINK UNIT.  Its generated hgcc objects are valid, but putting all of
# their device payloads into one x86-64 executable stretches the final host ELF past the signed 32-bit relocations
# used by crtbeginS.o and the host stubs.  The failure otherwise arrives only after every expensive device TU has
# compiled.  Refuse the one combination we have measured to fail before even asking for the SDK; narrower, explicit
# format subsets remain available for focused experiments and benchmarks/sweep_all_formats.sh is the full-coverage
# path (five complete search spaces, five link units, one appended result stream).
#
# Empty MOE_FORMATS means all five in CMake.  Also catch the explicit full set in any order.  Unknown names and
# duplicates remain CMake's validation job: they already fail at configure time rather than after compilation.
if [ "$TARGET" = "test_lowbit_moe_bench" ] && [ "${MOE_ALLOW_ALL_FORMAT_MONOLITH:-0}" != 1 ]; then
  _moe_all_selected=0
  if [ -z "${MOE_FORMATS:-}" ]; then
    _moe_all_selected=1
  else
    _moe_q3=0; _moe_q5=0; _moe_q6=0; _moe_i2=0; _moe_i4=0; _moe_unknown=0; _moe_count=0
    IFS=';' read -r -a _moe_requested <<< "$MOE_FORMATS"
    for _moe_fmt in "${_moe_requested[@]}"; do
      _moe_count=$((_moe_count + 1))
      case "$_moe_fmt" in
        q3) _moe_q3=1 ;; q5) _moe_q5=1 ;; q6) _moe_q6=1 ;; i2) _moe_i2=1 ;; i4) _moe_i4=1 ;;
        *) _moe_unknown=1 ;;
      esac
    done
    if [ "$_moe_count" -eq 5 ] && [ "$_moe_unknown" -eq 0 ] &&
       [ $((_moe_q3 + _moe_q5 + _moe_q6 + _moe_i2 + _moe_i4)) -eq 5 ]; then
      _moe_all_selected=1
    fi
  fi
  if [ "$_moe_all_selected" -eq 1 ]; then
    echo "[build.sh] refusing the known-oversize five-format MoE link before hgcc starts." >&2
    echo "           Run: bash benchmarks/sweep_all_formats.sh" >&2
    echo "           That builds q3/q5/q6/i2/i4 separately and appends every result; coverage is unchanged." >&2
    echo "           MOE_ALLOW_ALL_FORMAT_MONOLITH=1 is only for linker-layout diagnosis on the box." >&2
    exit 2
  fi
fi
if [ "$TARGET" = "test_lowbit_dense_bench" ] ||
   [ "$TARGET" = "test_lowbit_dense_persistent_ab" ] ||
   [ "$TARGET" = "test_lowbit_dense_streamk_ab" ]; then
  # FAIL BEFORE HGCC. A stale generated table otherwise presents as an unrelated CollectiveMma/GemmUniversal
  # template failure, and a bench-side startup banner cannot help because no binary was produced. Rebuild the
  # emitter in a temporary directory and compare its exact output; this validates without making generation a
  # compile-order dependency or maintaining a second runtime/dispatch list.
  python3 "$HERE/ci/check_dense_tactic_table.py" || exit 1
fi
if [ -z "$PPU_SDK_ROOT" ]; then
  echo "[build.sh] PPU_SDK is not set and there is no site default." >&2
  echo "            export PPU_SDK=/path/to/PPU_SDK   (or PPU_HOME, or PPU_SDK_SITE_DEFAULT in the shell profile)" >&2
  exit 1
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

# NO CLEANUP TRAP. There is nothing to restore: this build copies nothing into the submodule and edits none of
# its files. The trap that used to be here put back actlize's examples/CMakeLists.txt and rm -rf'd the overlay,
# and existed only because the build mutated a checkout it did not own.
echo "[build.sh] CUTLASS_PPU_ARCHS=$ARCH"

# NOTE: the former MoE/gs32 *.patch files are now baked into the submodule (fork ppu-w4a16-dev). No patch step.



# --- tile/warp/stages tuning: forward from the environment (defaults match the stock example) ---
TILE_M="${TILE_M:-32}"; TILE_N="${TILE_N:-32}"; WARP_M="${WARP_M:-16}"; WARP_N="${WARP_N:-16}"; STAGES="${STAGES:-3}"
QUANT="${QUANT:-int4}"   # int4 (default), uint2, or uint1 -> test_lowbit_dense_bench's W4/W2/W1 QuantType
TSK="${TSK:-}"           # TileShapeK override (empty: int1->256, int2->128, int4->64). Set only with a matching table
BENCH_GS="${BENCH_GS:-}"  # build ONE dense group size instead of all four. Every generated config wrapper instantiates
                         # each group-size arm whether or not --g can select it, so restricting the group size divides
                         # the tactic kernel count by four. How much
                         # hgcc time that saves is UNMEASURED; nvcc's front end goes 194s -> 136s, but codegen is
                         # where the instantiation count would tell and the front end never reaches it.
                         # Unset keeps the one-binary --g contract. e.g. BENCH_GS=32 ./build.sh
LOWBIT_DENSE_CONFIGS_PER_UNIT="${LOWBIT_DENSE_CONFIGS_PER_UNIT:-4}"
echo "[build.sh] TILE=${TILE_M}x${TILE_N} WARP=${WARP_M}x${WARP_N} STAGES=${STAGES} QUANT=${QUANT} TSK=${TSK} BENCH_GS=${BENCH_GS:-all} DENSE_CONFIGS_PER_UNIT=$LOWBIT_DENSE_CONFIGS_PER_UNIT"

# --- configure & build just our target ---
# OVERRIDABLE, because build.sh rm -rf's this and something that RUNS build.sh to check it must be able to point it
# somewhere disposable. Without the override, ci/box_build_dryrun.sh destroyed the real build tree every time the
# local tier ran -- on the box, that is someone's working build.
# REPO-LOCAL, NOT INSIDE THE SUBMODULE. This defaulted to $ACTLIZE/build_w4a16_compare, i.e. build products in
# third_party/actlize's worktree: it dirties the submodule's git status, `git submodule update` can destroy it,
# and by convention third_party/ holds vendored SOURCE, not our outputs. It is also why every command in this
# repo had to locate a binary with `find third_party/actlize/build_w4a16_compare ...`.
#
# (Historical: there used to be an OVERLAY under examples/99_quactlize_w4a16_compare, copied into the submodule
# and compiled as an actlize example. #34 removed it -- the targets build from this tree now.)
#
# .gitignore already covers build/ and build_*/, so nothing new is needed to keep it untracked.
# build_ppu, NOT build/. setuptools owns build/ (build/lib.* and build/temp.*), and `setup.py clean --all`
# removes it wholesale -- a Python rebuild would silently destroy the PPU tree. A sibling avoids that, and
# .gitignore's existing `build_*/` already covers this name.
#
# The old name was build_w4a16_compare, after an experiment this repo stopped being about; targets now land
# under ppu_targets/ with no examples/ nesting, because they are no longer actlize examples.
BUILD="${PPU_BUILD_DIR:-$HERE/build_ppu}"
# An old tree has products in the submodule. Say so rather than leaving two candidates for `find` to pick from.
for _stale in "$ACTLIZE/build_w4a16_compare" "$HERE/build_w4a16_compare"; do
  if [ -d "$_stale" ] && [ "$BUILD" != "$_stale" ]; then
    echo "[build.sh] removing the stale build tree $_stale"
    rm -rf "$_stale"
  fi
done
# EXPLICIT SOURCE DIRECTORY. `cmake ..` only meant "the actlize root" while $BUILD was inside it; once the build
# directory became overridable, `..` resolved to wherever that happened to be. Naming the source is both correct and
# independent of where the build lands.
rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
# FORWARD THE SWEEP AXIS KNOBS. They were added to CMakeLists.txt and then not wired through here, so narrowing a sweep was
# impossible from build.sh -- the knob existed and could not be reached, which is worse than no knob because it reads as
# available. Keep this explicit list in lockstep with the cache variables advertised by CMake; the advice gate
# below checks the link so a new printed knob cannot become another accepted-but-dropped environment variable.
_MOE_VARS=()
for _v in MOE_FORMATS MOE_TM_LIST MOE_TN_LIST MOE_WM_LIST MOE_STAGES MOE_CORES; do
  if [ -n "${!_v:-}" ]; then _MOE_VARS+=("-D$_v=${!_v}"); echo "[build.sh] $_v=${!_v}"; fi
done
# NO GOOGLETEST CLONE. actlize's CMakeLists.txt:423 clones github.com/google/googletest when
# CUTLASS_ENABLE_GTEST_UNIT_TESTS is on, and :108 defaults it to CUTLASS_ENABLE_TESTS -- which nothing here ever
# turned off. That is why ci/local_gates.py's boxdry step, whose whole tier claims to need no network, sat at
# index-pack with zero output for nine minutes and had to be killed twice. We build no gtest unit tests, so this
# is not a capability being given up; it is a download nobody wanted.
# WHICH SOURCE TREE. Default is OUR root with -DQUACTLIZE_PPU=ON: actlize becomes a subproject and our targets
# build from our own directories. The legacy example-injection path is GONE -- it was kept for one round and
# the user retired it. Local evidence before removal: 35 targets both ways with the same names and nothing
# skipped, and an hgcc command line identical bar the -I set and the source path, with every file the overlay
# used to supply verified reachable through the new device include list.
# BUILD FROM THIS TREE. There is no other path.
#
# EVIDENCE FOR THE SWITCH, and it is stronger than the target parity I first settled for. Against the same stub
# SDK, the generated hgcc command line for test_moe_splitk_bench differs from the legacy path in EXACTLY two
# places, both of which must differ: the include directories are our real benchmarks/ and csrc/ instead of the
# overlay, and the source is its real path instead of a copy. Every flag, every -D (including the ones actlize
# appends after its toolchain: CUTLASS_USE_PACKED_TUPLE and friends), and every arch option are identical.

# NOTHING IN THIS BUILD MAY REACH THE NETWORK. Two mechanisms, and only one of them is on this command line.
#
# THE WANT is removed by CMakeLists.txt:31, `set(CUTLASS_ENABLE_GTEST_UNIT_TESTS OFF CACHE BOOL "" FORCE)`, which
# runs before add_subdirectory(actlize) -- actlize's cmake/googletest.cmake FetchContent_Declares googletest behind
# that option. NOT by the -DCUTLASS_ENABLE_GTEST_UNIT_TESTS=OFF below: FORCE overwrites the cache, so the
# command-line value never decides anything. Both flags here are inert and are kept only because they document the
# intent at the point someone reads this invocation. Established by flipping them: changing the -D to ON changes
# nothing at all, changing CMakeLists.txt:31 to ON breaks the configure. I spent a negative control believing the
# -D was the mechanism, and it passed for the wrong reason.
#
# THE ABILITY is removed by FETCHCONTENT_FULLY_DISCONNECTED=ON, and that is the half the offline claim needs.
# ci/local_gates.py's first line says it "runs every check that can falsify something without a PPU", and the local
# tier reaches this script through boxdry -- so "the local tier is green" is establishable offline only while that
# one option stays off and no second FetchContent_Declare appears anywhere in a submodule we do not control. A flag
# is a want; wants get flipped. With this set, a populate cannot silently become a clone.
#
# VERIFIED, by forcing CMakeLists.txt:31 to ON and reading the diagnostic rather than the exit code:
#     CMake Error at third_party/actlize/cmake/googletest.cmake:49 (add_subdirectory):
#       add_subdirectory given source "/tmp/.../_deps/googletest-src"
# FetchContent_Populate left the directory empty instead of cloning, so the configure died in seconds naming the
# dependency, where the original symptom was nine minutes at index-pack with zero output. A hang is worse than a
# failure: the operator cannot tell "still compiling" from "stuck".
#
# Everything this build needs is a submodule. If something ever legitimately needs fetching, the loud error is the
# correct first outcome -- it forces the choice to be vendored deliberately rather than acquired by a clone.
_CMAKE_SRC="$HERE"; _CMAKE_EXTRA=(-DQUACTLIZE_PPU=ON)
cmake "$_CMAKE_SRC" "${_CMAKE_EXTRA[@]}" -DPPU_SDK_ROOT="$PPU_SDK_ROOT" -DCUTLASS_PPU_ARCHS="$ARCH" \
  -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_GTEST_UNIT_TESTS=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DTILE_M="$TILE_M" -DTILE_N="$TILE_N" -DWARP_M="$WARP_M" -DWARP_N="$WARP_N" -DSTAGES="$STAGES" -DBENCH_QUANT="$QUANT" -DTSK="$TSK" -DBENCH_GS="$BENCH_GS" \
  -DLOWBIT_DENSE_CONFIGS_PER_UNIT="$LOWBIT_DENSE_CONFIGS_PER_UNIT" \
  -DPPU_EXTRA_DEFS="${PPU_DEFS:-}" "${_MOE_VARS[@]+"${_MOE_VARS[@]}"}" \
  >cmake.log 2>&1 || { tail -40 cmake.log; exit 1; }
# CMake owns the literal list and the advice gate checks every name in it. Surface that one source of truth for
# the two sweep targets instead of copying the names into another shell message that can drift.
case "$TARGET" in
  test_lowbit_moe_bench|test_lowbit_moe_decode_bench)
    grep -F "Narrow a MoE axis" cmake.log || {
      echo "  WARNING: CMake did not report the MoE restriction controls" >&2; exit 1; }
    ;;
esac
# cmake's stdout is redirected above, so its message(STATUS ...) never reaches the terminal. Surface the extra
# defines here instead -- telling someone to "check that the line appeared" when it cannot appear is worse than
# not printing it at all.
if [ -n "${PPU_DEFS:-}" ]; then
  echo "PPU_DEFS applied: $PPU_DEFS"
  grep -F "PPU_EXTRA_DEFS ->" cmake.log || echo "  WARNING: cmake did not report PPU_EXTRA_DEFS -- the defines did NOT reach the build"
fi
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
# PRINT THE BINARY. Every command in BOX.md and the docs opened with a `find ... -perm -u+x -print -quit`
# because nothing said where the product landed; a find can also pick up a survivor of an earlier run, which is
# how a "rebuilt" binary turns out to be the old one.
_BIN_PATH=$(find "$BUILD" -type f -name "$TARGET" -perm -u+x -print -quit 2>/dev/null || true)

# CMAKE RECEIVING THE DEFINES IS NOT THE SAME AS THIS TARGET GETTING THEM, and the difference is invisible in a perf
# number. The defines used to be attached to three targets by hand, so
#   PPU_DEFS=PPU_B_CHUNK=1 TARGET=test_scalefirst_bench ./build.sh
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

if [ "$TARGET" = quactlize_ppu ]; then
  BIN="$(find "$BUILD" -name 'libquactlize_ppu.so' -type f | head -1)"
else
  BIN="$(find "$BUILD" -name "$TARGET" -type f -perm -u+x | head -1)"
fi
echo
echo "built: $BIN"
# Per-target run hint. The old line advertised --m=/--mode= for EVERY target, but only the compare example
# parses that shape: on a positional-arg target each flag becomes atoi("--m=2048") == 0, so copying the hint
# silently selects L=0 or Mb=0 and the test passes vacuously. One such copy already cost a box round.
case "$TARGET" in
  quactlize_ppu)
    echo "load:  QUACTLIZE_PPU_LIB=$BIN python -c 'import quactlize; print(quactlize.gguf_backend())'" ;;
  test_moe_grouped_verify)
    echo "run:   $BIN [L] [Mb] [ragged?] [gs]      # POSITIONAL, no --flags"
    echo "       $BIN 8 1                          # Mmax==1, required by PPU_A_CUBE_H=1"
    echo "       MOEG_SMEM=1 MOEG_DUMP=/tmp/d.bin $BIN 8 1     # then MOEG_CHECK=/tmp/d.bin on the other build" ;;
  test_moe_grouped_streamk)
    echo "run:   timeout 180s $BIN                  # fixed S068 + ragged q/fixup/numeric gate"
    echo "       expected: TK256 split_tiles=0; TK64 split_tiles=16 peer_excess=48" ;;
  test_moe_splitk_bench|test_lowbit_moe_bench)
    echo "run:   $BIN <L> <rows-or-tokens> <N> <K> <gs> <mode> [top-k]"
    echo "       $BIN 256 4 512 2048 32 4 8       # pinned token->top-k router; prints Mmax/capacity"
    echo "       SPLITK_CFG='16x128:256 w16x16 s2' SPLITK_S=1 $BIN 64 8 2048 2048 32 3   # select one acu row" ;;
  test_gemv_perf)
    echo "run:   $BIN [shape]                      # GEMV_FMT=/GEMV_CFG= select rows" ;;
  *)
    echo "run:   $BIN --m=2048 --n=4096 --k=4096 --g=128 --mode=1 --iterations=100" ;;
esac

if [ -n "${_BIN_PATH:-}" ]; then
  echo ""
  echo "[build.sh] BINARY: $_BIN_PATH"
  echo "           BIN=\$($0 --print-bin) is not a thing; copy the line above, or:"
  echo "           BIN=$_BIN_PATH"
fi
