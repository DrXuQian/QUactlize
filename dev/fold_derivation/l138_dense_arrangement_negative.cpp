#include "ppu_placed_arrangement.hpp"

constexpr quactlize_ppu_placed_arrangement_v1 artifact{
    QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1, 2, 64, 1};  // Q3 low F=2, high F=4

static_assert(ppu_arrangements::packed_tensor_matches_exact_reader<11, 256, 256>(&artifact, 4096),
              "L138_EXPECTED_F2_TO_F1_REJECTION");
int main() {}
