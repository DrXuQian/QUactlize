# Thin by design: the device half is driven by actlize's example machinery through build.sh, which is what the PPU
# toolchain supports. This file exists so the tree configures for tooling (compile_commands.json, IDEs) and so the
# include paths are stated once.
cmake_minimum_required(VERSION 3.19)
project(quactlize LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(QUACTLIZE_INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/quactlize/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include)
add_library(quactlize INTERFACE)
target_include_directories(quactlize INTERFACE ${QUACTLIZE_INCLUDE})
message(STATUS "quactlize: header-only interface; device targets are built by build.sh via actlize")
