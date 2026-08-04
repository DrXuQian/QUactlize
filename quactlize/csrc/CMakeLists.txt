# THE PPU TARGETS, BUILT FROM THIS TREE -- not copied into a submodule first.
#
# WHAT THIS REPLACES. build.sh used to copy 83 files from five of our directories, FLATTENED, into
# third_party/actlize/examples/99_quactlize_w4a16_compare/, sed a line into the submodule's
# examples/CMakeLists.txt to register them, and rm -rf the lot on exit. Sixty-one lines of build.sh existed only
# to manage that injection, plus a cleanup trap and a check proving the overlay was this checkout rather than a
# survivor of an earlier run. Build products landed inside a git submodule's worktree, where `git submodule
# update` can destroy them and where `third_party/` is supposed to hold vendored SOURCE.
#
# WHY IT WAS EVER DONE THAT WAY, and why it does not have to be: being an actlize "example" buys
# cutlass_example_add_executable, whose substance is cutlass_add_executable plus three link libraries and two
# include directories. cutlass_add_executable is defined in actlize's cmake/PPUToolchain.cmake:329 and
# ppu::driver at :199, and actlize's TOP-LEVEL CMakeLists includes that file at line 62 -- so
# add_subdirectory(third_party/actlize) yields both. Nothing in the example wrapper required our sources to live
# inside examples/. It was the fastest way to get a PPU binary once, and it was never revisited.
#
# THE ONE THING THAT IS NOT FREE. The sources #include EACH OTHER BY BARE FILENAME, because the overlay
# flattened five directories into one. Rewriting those includes would be a large and pointless diff, so the same
# flat namespace is reproduced here with include_directories over the same five. The bare filenames in the
# target list below resolve the same way, through QZ_SRC_DIRS in ppu_w4a16_executable -- which additionally
# ERRORS on a name that exists in two of them. The overlay would have copied one over the other and kept
# whichever went last, silently; there is no such collision today and now there cannot be one unnoticed.

# The five directories the overlay used to flatten. ORDER IS NOT SIGNIFICANT -- a name in two of them is an
# error, not a precedence question, which is the point of checking rather than ordering.
get_filename_component(QZ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(QZ_SRC_DIRS
    ${QZ_ROOT}/tests
    ${QZ_ROOT}/benchmarks
    ${QZ_ROOT}/dev
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR})

# Headers resolve the same way sources do. quactlize/include and its gemv_lowbit/ subdirectory are here because
# the overlay copied gemv_lowbit/ as a subdirectory (its library must come whole) while everything else went
# flat -- so both spellings are in use and both must work.
include_directories(
    ${QZ_SRC_DIRS}
    ${QZ_ROOT}/quactlize/include
    ${QZ_ROOT}/quactlize/include/gemv_lowbit
    ${QZ_ROOT}/third_party/actlize/include
    ${QZ_ROOT}/third_party/actlize/tools/util/include)

# ---------------------------------------------------------------------------------------------------------
# THE COMPILE FLAGS COME FROM actlize's DIRECTORY, ASKED FOR RATHER THAN RESTATED.
#
# add_subdirectory creates a CHILD scope: variables set inside actlize's CMakeLists do not propagate back to the
# parent, so a sibling directory under OUR root does not see them. Inside examples/ they were inherited for free,
# which is the one thing being an example genuinely bought and the one thing that is not obvious.
#
# The symptom, if this is missing, is not a build failure. The targets all configure, the target set matches
# exactly (35 == 35, verified), and the compile command carries NO -D at all -- not the CUTLASS ones and not
# PPU_DEFS. Every A/B through PPU_DEFS would then compare a binary against itself and report "no change", which
# is indistinguishable from "the change does nothing". That failure is already recorded in CMakeLists.txt.in for
# a different cause; this would have reintroduced it through the build system instead of the macro plumbing.
#
# get_directory_property(... DEFINITION ...) READS the variable as actlize set it. Copying the flag list here
# instead would be a second definition that drifts the first time actlize's changes -- the same defect as the
# config table that had two hand-maintained copies.
get_directory_property(_QZ_HGCC_FLAGS
    DIRECTORY ${QZ_ROOT}/third_party/actlize DEFINITION CUTLASS_PPU_EXTRA_HGCC_FLAGS)
get_directory_property(_QZ_CXX_FLAGS
    DIRECTORY ${QZ_ROOT}/third_party/actlize DEFINITION CMAKE_CXX_FLAGS)
if(NOT _QZ_HGCC_FLAGS)
  message(FATAL_ERROR
    "CUTLASS_PPU_EXTRA_HGCC_FLAGS is empty as read from actlize's directory. Either add_subdirectory(actlize) "
    "did not run before this, or the variable was renamed there. Building on without it produces targets whose "
    "compile command has no -D at all, which looks like success.")
endif()
set(CUTLASS_PPU_EXTRA_HGCC_FLAGS ${_QZ_HGCC_FLAGS})
set(CMAKE_CXX_FLAGS "${_QZ_CXX_FLAGS}")

# The target definitions are UNCHANGED and still live in CMakeLists.txt.in. Including rather than copying keeps
# one definition of the target set: a second copy is a second place for a target to be added and forgotten,
# which is the defect this whole refactor is about at the level of files.
include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in)
