#pragma once
#include <type_traits>
#include <utility>
#include "cutlass/cutlass.h"

namespace quactlize::ppu17::detail {
// Optional collective capability; no new scheduler or decision authority.
// The isolated build inserts this call at the existing scheduler/MMA seam.
template<class Collective, class = void> struct KTileCoordinate {
  CUTLASS_DEVICE static void set(Collective&, int) {}
};
template<class Collective> struct KTileCoordinate<Collective,
    std::void_t<decltype(std::declval<Collective&>().set_k_tile_start(0))>> {
  CUTLASS_DEVICE static void set(Collective& collective, int first) {
    collective.set_k_tile_start(first);
  }
};
} // namespace quactlize::ppu17::detail
