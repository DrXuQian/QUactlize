# Two jobs. Without QUACTLIZE_PPU this is a header-only interface target so the tree configures for tooling
# (compile_commands.json, IDEs) and the include paths are stated once. With -DQUACTLIZE_PPU=ON it is the REAL
# device build: actlize as a subproject, our targets from our own tree.
#
# The comment that used to be here said the device half "is driven by actlize's example machinery through
# build.sh, which is what the PPU toolchain supports". The second clause was never true and is what kept the
# overlay alive: the toolchain lives in actlize/cmake/PPUToolchain.cmake, actlize's top-level CMakeLists includes
# it, and add_subdirectory therefore supplies cutlass_add_executable and ppu::driver with no example machinery
# involved. See quactlize/csrc/CMakeLists.txt.
cmake_minimum_required(VERSION 3.19)
project(quactlize LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(QUACTLIZE_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/quactlize/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
add_library(quactlize INTERFACE)
target_include_directories(quactlize INTERFACE ${QUACTLIZE_INCLUDE})
message(STATUS "quactlize: header-only interface; device targets are built by build.sh via actlize")

# ---------------------------------------------------------------------------------------------------------
option(QUACTLIZE_PPU "Build the PPU device targets (requires the PPU SDK and hgcc)" OFF)
if(QUACTLIZE_PPU)
  # EXAMPLES OFF is the point of the exercise -- we are no longer one. TOOLS must stay ON: our targets link
  # cutlass_tools_util_includes, defined in actlize's tools/util/CMakeLists.txt:31. TESTS and the GTest unit
  # tests stay OFF because the latter git-clones googletest at configure time, which is what made the offline
  # gate tier hang for nine minutes on a tier whose whole claim is that it needs no network.
  set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_TOOLS    ON  CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_GTEST_UNIT_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize actlize)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/quactlize/csrc ppu_targets)
endif()
