// Mandatory dlopen symbols which the zero-redundancy oracle never calls.
//
// The focused local test links the REAL dense placement and packed-unit
// producer objects beside this file.  ppu_backend::load() deliberately requires
// the five historical device entries before it exposes optional host producer
// seams, so no-op definitions are needed to reach those real producers without
// compiling an unrelated GEMV kernel.  Every stub returns a distinctive error;
// accidentally invoking one fails rather than manufacturing a result.
#include <cstdint>

extern "C" int32_t quactlize_ppu_build_packed_format_v1() { return -1; }
extern "C" int quactlize_ppu_vecdot(
    uint8_t const*, int64_t, uint16_t const*, float*, int, int, int) { return 193; }
extern "C" int quactlize_ppu_vecdot_moe(
    uint8_t const*, int64_t, uint16_t const*, int const*, float*, int, int, int, int, int, int) { return 193; }
extern "C" int quactlize_ppu_dequantize(
    uint8_t const*, int64_t, uint16_t*, int, int) { return 193; }
extern "C" int quactlize_ppu_prepass(
    uint8_t const*, int64_t, uint16_t const*, uint16_t const*, int,
    uint16_t*, uint16_t*, int, int, int) { return 193; }
extern "C" int quactlize_ppu_gemv_lowbit(
    uint16_t const*, uint8_t const*, uint8_t const*, uint16_t const*, uint16_t const*, uint16_t*,
    int, int, int, int, int, int, int const*, int) { return 193; }
