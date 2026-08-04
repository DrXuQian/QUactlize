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
# THE TOOLCHAIN IS INCLUDED HERE, IN OUR SCOPE. This is the whole of what being an actlize example bought.
#
# add_subdirectory gives actlize a CHILD scope, and everything PPUToolchain.cmake sets -- the hgcc path
# (PPU_DEVICE_HGCC_REAL), the extra device flags (CUTLASS_PPU_EXTRA_HGCC_FLAGS), the arch flags, CMAKE_CXX_FLAGS
# -- stays there. A sibling directory under our root inherits NONE of it.
#
# THE SYMPTOM IS SILENT AND IT IS WORSE THAN A MISSING FLAG. With the variables absent, cutlass_build_dev_kernels
# still emits an add_custom_command whose COMMAND expands to nothing: the generated build.make declares the
# object, prints "[hgcc] <source>", and runs NOTHING. Every target "builds", instantly, producing no objects. The
# first thing that noticed was the link.
#
# Chasing the variables one at a time with get_directory_property is what I tried first, and it is the "second
# copy" defect: a list of names here that has to track a 381-line file nobody will diff. include() executes in
# the CURRENT scope, so including actlize's own toolchain file sets all of it, once, from the one definition.
# Re-including is safe -- it defines functions and sets variables, and CMake permits redefinition.
include(${QZ_ROOT}/third_party/actlize/cmake/PPUToolchain.cmake)
# AND THEN actlize's OWN APPENDS, which happen in its top-level CMakeLists AFTER it includes the toolchain --
# CUTLASS_USE_PACKED_TUPLE / CUTE_USE_PACKED_TUPLE and the CUTLASS_TEST_* set. include() above gives us the
# toolchain's value; only the directory holds the appended one.
#
# THIS IS NOT COSMETIC. CUTLASS_USE_PACKED_TUPLE=1 changes cute::tuple to the standard-layout implementation.
# Compiling our translation units without it while actlize's headers were configured with it is an ABI
# mismatch between two halves of the same binary, and nothing would say so.
get_directory_property(_QZ_ACTLIZE_HGCC
    DIRECTORY ${QZ_ROOT}/third_party/actlize DEFINITION CUTLASS_PPU_EXTRA_HGCC_FLAGS)
if(_QZ_ACTLIZE_HGCC)
  set(CUTLASS_PPU_EXTRA_HGCC_FLAGS ${_QZ_ACTLIZE_HGCC})
endif()
get_directory_property(_QZ_ACTLIZE_CXX
    DIRECTORY ${QZ_ROOT}/third_party/actlize DEFINITION CMAKE_CXX_FLAGS)
if(_QZ_ACTLIZE_CXX)
  set(CMAKE_CXX_FLAGS "${_QZ_ACTLIZE_CXX}")
endif()

# ---------------------------------------------------------------------------------------------------------
# AND THE DEVICE INCLUDE PATH, which include_directories() does NOT reach.
#
# cutlass_build_dev_kernels emits an add_custom_command; a custom command takes no directory properties. Its -I
# list is exactly ${CUTLASS_PPU_DEV_INCLUDE_FLAGS} plus two implicit entries -- the source file's own directory
# and CMAKE_CURRENT_SOURCE_DIR. Under the overlay those two were the SAME flattened directory and therefore
# covered every header we own. Here they are benchmarks/ and csrc/, and nothing else of ours is on the line.
#
# I MISREAD THIS ONCE AND SHOULD NOT HAVE. Diffing the two command lines showed a two-line -I difference and I
# called it "exactly the ones that should differ". The COUNT was unremarkable; the COVERAGE was not -- one -I
# over a flattened tree is not two -I over two of its five directories. The first thing to notice was
# `fatal error: unfused_weight_dequantize.hpp: No such file or directory` on the box.
list(APPEND CUTLASS_PPU_DEV_INCLUDE_FLAGS
    "-I${QZ_ROOT}/quactlize/include"
    "-I${QZ_ROOT}/quactlize/include/gemv_lowbit"
    "-I${QZ_ROOT}/tests"
    "-I${QZ_ROOT}/benchmarks"
    "-I${QZ_ROOT}/dev"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/device")

if(NOT PPU_DEVICE_HGCC_REAL)
  message(FATAL_ERROR
    "PPU_DEVICE_HGCC_REAL is empty after including PPUToolchain.cmake. Without it every device rule expands to "
    "an EMPTY command: the build prints [hgcc] per source, produces no objects, and fails at the link -- which "
    "reads as a link problem and is not one.")
endif()

# The target definitions are UNCHANGED and still live in CMakeLists.txt.in. Including rather than copying keeps
# one definition of the target set: a second copy is a second place for a target to be added and forgotten,
# which is the defect this whole refactor is about at the level of files.
include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in)
