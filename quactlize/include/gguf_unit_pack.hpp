#pragma once
// Host producer for the byte-neutral metadata consumed by the fully-quantized GEMMs.  This is shared by the
// torch preparation path and libquactlize_ppu's C ABI: the superblock pairing and [E,unit,N,byte] addressing live
// in one loop, so exporting the producer cannot create a second artifact definition.

#include <cstdint>
#include <cstring>
#include <limits>

#include "gguf_packed_unit.hpp"

namespace gguf_scale {
namespace unit_pack {

// Official GGUF byte locations.  Field placement inside the reordered unit remains owned by
// packed_unit::Unit<T>; these traits name only byte-aligned slices in the source block.
template <KType T> struct Raw;
template <> struct Raw<KType::Q4_K> {
  static constexpr int kBytes=144, kScaleOffset=4, kDOffset=0, kDminOffset=2;
};
template <> struct Raw<KType::Q2_K> {
  static constexpr int kBytes=84, kScaleOffset=0, kDOffset=80, kDminOffset=82;
};
template <> struct Raw<KType::Q3_K> {
  static constexpr int kBytes=110, kScaleOffset=96, kDOffset=108, kDminOffset=-1;
};
template <> struct Raw<KType::Q5_K> {
  static constexpr int kBytes=176, kScaleOffset=4, kDOffset=0, kDminOffset=2;
};
template <> struct Raw<KType::Q6_K> {
  static constexpr int kBytes=210, kScaleOffset=192, kDOffset=208, kDminOffset=-1;
};

template <KType T>
void pack_raw_unit_sb(uint8_t const* block, int sb, uint8_t* unit) {
  using R = Raw<T>;
  using U = packed_unit::Unit<T>;
  cutlass::half_t d, dmin{0.f};
  std::memcpy(&d, block + R::kDOffset, sizeof(d));
  if constexpr (U::kHasMin) std::memcpy(&dmin, block + R::kDminOffset, sizeof(dmin));
  packed_unit::pack_unit_sb<T>(block + R::kScaleOffset, d, dmin, sb, unit);
}

template <KType T>
int64_t bytes(int n, int k) {
  using U = packed_unit::Unit<T>;
  if (n <= 0 || n % 256 || k <= 0 || k % 256) return -1;
  int64_t const superblocks = k / 256;
  if (superblocks % U::kSbPerUnit) return -1;
  int64_t const per_column = (superblocks / U::kSbPerUnit) * U::kUnitTotal;
  if (per_column > std::numeric_limits<int64_t>::max() / n) return -1;
  return int64_t(n) * per_column;
}

// Raw blocks are [E,N,K/256,raw-byte], units are [E,K-unit,N,unit-byte].  Dense is exactly E=1.  Keeping experts
// in this one loop is what makes the grouped entry exercise its expert-base addressing rather than call dense in a
// Python loop.
template <KType T>
void pack(uint8_t const* blocks, uint8_t* units, int n, int k, int experts) {
  using R = Raw<T>;
  using U = packed_unit::Unit<T>;
  int64_t const superblocks = k / 256;
  int64_t const num_units = superblocks / U::kSbPerUnit;
  for (int e = 0; e < experts; ++e)
    for (int col = 0; col < n; ++col)
      for (int64_t unit = 0; unit < num_units; ++unit)
        for (int sb = 0; sb < U::kSbPerUnit; ++sb)
          pack_raw_unit_sb<T>(
              blocks + ((int64_t(e) * n + col) * superblocks + unit * U::kSbPerUnit + sb) * R::kBytes,
              sb,
              units + ((int64_t(e) * num_units + unit) * n + col) * U::kUnitTotal);
}

}  // namespace unit_pack
}  // namespace gguf_scale
