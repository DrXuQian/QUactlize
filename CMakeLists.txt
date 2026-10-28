cmake_minimum_required(VERSION 3.19)
project(quactlize LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Header interface for consumers and IDEs. Build the installable PPU package
# through build.sh --library; experiment executables are not product targets.
add_library(quactlize INTERFACE)
target_include_directories(quactlize INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/quactlize/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include)

option(QUACTLIZE_PPU "Legacy experimental device targets" OFF)
if(QUACTLIZE_PPU)
    message(FATAL_ERROR "Use build.sh --library for the PPU package. Legacy targets live on experiments/ppu-tuning.")
endif()
