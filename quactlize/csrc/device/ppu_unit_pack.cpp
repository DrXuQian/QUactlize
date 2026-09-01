// Host-only forward producer for the packed metadata channel.  Kept separate from ppu_dense_layout.cu so callers
// which need only units do not inherit xplane templates, and so this ABI can be tested with an ordinary host build.
#include "gguf_unit_pack.hpp"
#include "gguf_vecdot.hpp"
#include "ppu_format_config.hpp"
#include "ppu_placed_arrangement.hpp"
#include "quactlize_ppu_packed.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

extern "C" int quactlize_ppu_prepare_dense_for_tile(
    uint8_t const* low_native, uint8_t const* high_native,
    uint8_t* low_layout, uint8_t* high_layout, int n, int k, int qtype, int tile_k);
extern "C" int quactlize_ppu_recover_dense_for_tile(
    uint8_t const* low_layout, uint8_t const* high_layout,
    uint8_t* low_native, uint8_t* high_native, int n, int k, int qtype, int tile_k);

namespace {

using gguf_scale::KType;

bool product_size(std::initializer_list<size_t> factors, size_t* result) {
  size_t value = 1;
  for (size_t factor : factors) {
    if (factor && value > size_t(-1) / factor) return false;
    value *= factor;
  }
  *result = value;
  return true;
}

bool product_arrangement(
    int qtype, int k,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  if (!arrangement) return false;
  int const layout = qtype == 12
      ? QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1
      : QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1;
  auto const& format = ppu_formats::for_qtype(qtype);
  return arrangement->layout == layout && format.qtype == qtype &&
         ppu_arrangements::matches_compiled_tactic(
             arrangement, qtype, k, format.fully_quantized_tile_k);
}

bool build_owns_qtype(int qtype) {
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT)
  auto const& format = ppu_formats::for_qtype(qtype);
  return format.qtype == qtype && format.packed_format == PPU_PACKED_FORMAT;
#else
  (void)qtype;
  return false;
#endif
}

struct CompleteArtifactSizes {
  size_t raw = 0;
  size_t low = 0;
  size_t high = 0;
  size_t units = 0;
};

template <KType T>
int complete_artifact_sizes(int n, int k, int experts, CompleteArtifactSizes* sizes) {
  using R = gguf_scale::unit_pack::Raw<T>;
  using U = gguf_scale::packed_unit::Unit<T>;
  constexpr int LowBits = (T == KType::Q2_K || T == KType::Q3_K) ? 2 : 4;
  constexpr int HighBits =
      (T == KType::Q3_K || T == KType::Q5_K) ? 1 : (T == KType::Q6_K ? 2 : 0);

  int const superblocks = k / 256;
  if (superblocks % U::kSbPerUnit) return 24;
  int const num_units = superblocks / U::kSbPerUnit;

  size_t plane_elements = 0, low_bits = 0, high_bits = 0;
  if (!product_size({size_t(n), size_t(k)}, &plane_elements) ||
      !product_size({size_t(experts), plane_elements, size_t(LowBits)}, &low_bits) ||
      !product_size({size_t(experts), plane_elements, size_t(HighBits)}, &high_bits) ||
      !product_size(
          {size_t(experts), size_t(n), size_t(superblocks), size_t(R::kBytes)},
          &sizes->raw) ||
      !product_size(
          {size_t(experts), size_t(num_units), size_t(n), size_t(U::kUnitTotal)},
          &sizes->units))
    return 26;

  size_t const max_index = size_t(std::numeric_limits<int64_t>::max());
  if (plane_elements > max_index || low_bits > max_index || high_bits > max_index ||
      sizes->raw > max_index || sizes->units > max_index)
    return 26;
  sizes->low = low_bits / 8;
  sizes->high = high_bits / 8;
  return 0;
}

int complete_artifact_sizes_for_qtype(
    int qtype, int n, int k, int experts, CompleteArtifactSizes* sizes) {
  switch (qtype) {
    case 10: return complete_artifact_sizes<KType::Q2_K>(n, k, experts, sizes);
    case 11: return complete_artifact_sizes<KType::Q3_K>(n, k, experts, sizes);
    case 12: return complete_artifact_sizes<KType::Q4_K>(n, k, experts, sizes);
    case 13: return complete_artifact_sizes<KType::Q5_K>(n, k, experts, sizes);
    case 14: return complete_artifact_sizes<KType::Q6_K>(n, k, experts, sizes);
    default: return 22;
  }
}

struct ByteRange {
  void const* data;
  size_t size;
};

int require_distinct_tensor_ranges(std::initializer_list<ByteRange> ranges) {
  for (auto const& range : ranges) {
    if (!range.size) continue;
    std::uintptr_t const begin = reinterpret_cast<std::uintptr_t>(range.data);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - range.size) return 26;
  }
  for (auto lhs = ranges.begin(); lhs != ranges.end(); ++lhs) {
    if (!lhs->size) continue;
    std::uintptr_t const lhs_begin = reinterpret_cast<std::uintptr_t>(lhs->data);
    std::uintptr_t const lhs_end = lhs_begin + lhs->size;
    for (auto rhs = lhs + 1; rhs != ranges.end(); ++rhs) {
      if (!rhs->size) continue;
      std::uintptr_t const rhs_begin = reinterpret_cast<std::uintptr_t>(rhs->data);
      std::uintptr_t const rhs_end = rhs_begin + rhs->size;
      if (lhs_begin < rhs_end && rhs_begin < lhs_end) return 30;
    }
  }
  return 0;
}

template <KType T>
int prepare(uint8_t const* blocks, uint8_t* units, int n, int k, int experts) {
  using U = gguf_scale::packed_unit::Unit<T>;
  if ((k / 256) % U::kSbPerUnit) return 24;
  gguf_scale::unit_pack::pack<T>(blocks, units, n, k, experts);
  return 0;
}

int prepare_dispatch(uint8_t const* blocks, uint8_t* units, int n, int k, int experts, int qtype) {
  if (!blocks || !units || experts <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256) return 20;
  switch (qtype) {
    case 10: return prepare<KType::Q2_K>(blocks, units, n, k, experts);
    case 11: return prepare<KType::Q3_K>(blocks, units, n, k, experts);
    case 12: return prepare<KType::Q4_K>(blocks, units, n, k, experts);
    case 13: return prepare<KType::Q5_K>(blocks, units, n, k, experts);
    case 14: return prepare<KType::Q6_K>(blocks, units, n, k, experts);
    default: return 22;
  }
}

template <int Bits>
void native_put(uint8_t* plane, int64_t logical, int value) {
  int64_t const bit = logical * Bits;
  plane[bit >> 3] |= uint8_t((value & ((1 << Bits) - 1)) << (bit & 7));
}

template <int Bits>
int native_get(uint8_t const* plane, int64_t logical) {
  int64_t const bit = logical * Bits;
  return (plane[bit >> 3] >> (bit & 7)) & ((1 << Bits) - 1);
}

template <class Field>
void official_field_put(uint8_t* bytes, int group, int value) {
  if constexpr (!cute::is_void_v<Field>) {
    int const mask = (1 << Field::kWidth) - 1;
    bytes[Field::byte_of(group)] |= uint8_t((value & mask) << Field::shift_of(group));
  }
}

template <class Field>
void official_code_put(uint8_t* bytes, int coordinate, int byte_in_word, int value) {
  if constexpr (!cute::is_void_v<Field>) {
    int const mask = (1 << Field::kWidth) - 1;
    bytes[Field::byte_of(coordinate) + byte_in_word] |=
        uint8_t((value & mask) << Field::shift_of(coordinate));
  }
}

template <KType T>
int prepare_fully_quantized(uint8_t const* blocks, uint8_t* low_layout, uint8_t* high_layout,
                            uint8_t* units, int n, int k, int experts, int qtype,
                            quactlize_ppu_placed_arrangement_v2 const* arrangement = nullptr,
                            CompleteArtifactSizes const* checked_sizes = nullptr) {
  using R = gguf_scale::unit_pack::Raw<T>;
  using Tr = gguf_scale::Traits<T>;
  constexpr int LowBits = (T == KType::Q2_K || T == KType::Q3_K) ? 2 : 4;
  constexpr int HighBits = (T == KType::Q3_K || T == KType::Q5_K) ? 1 : (T == KType::Q6_K ? 2 : 0);
  constexpr int CodeBias = T == KType::Q3_K ? 4 : (T == KType::Q6_K ? 32 : 0);
  CompleteArtifactSizes local_sizes;
  if (!checked_sizes) {
    int const rc = complete_artifact_sizes<T>(n, k, experts, &local_sizes);
    if (rc) return rc;
    checked_sizes = &local_sizes;
  }
  size_t const low_per_expert = checked_sizes->low / size_t(experts);
  size_t const high_per_expert = checked_sizes->high / size_t(experts);
  std::vector<uint8_t> low_native(low_per_expert, uint8_t(0));
  std::vector<uint8_t> high_native(HighBits ? high_per_expert : size_t(1), uint8_t(0));
  std::vector<int8_t> codes(256);
  std::vector<cutlass::half_t> scale(Tr::kGroups), zero(Tr::kGroups);
  int const superblocks = k / 256;

  for (int e = 0; e < experts; ++e) {
    std::fill(low_native.begin(), low_native.end(), uint8_t(0));
    std::fill(high_native.begin(), high_native.end(), uint8_t(0));
    for (int col = 0; col < n; ++col) {
      for (int sb = 0; sb < superblocks; ++sb) {
        uint8_t const* block = blocks + ((int64_t(e) * n + col) * superblocks + sb) * R::kBytes;
        gguf_scale::vecdot::unpack_block<T>(block, codes.data(), scale.data(), zero.data());
        for (int j = 0; j < 256; ++j) {
          int const q = int(codes[size_t(j)]) + CodeBias;
          int64_t const logical = int64_t(col) * k + int64_t(sb) * 256 + j;
          native_put<LowBits>(low_native.data(), logical, q);
          if constexpr (HighBits != 0) native_put<HighBits>(high_native.data(), logical, q >> LowBits);
        }
      }
    }
    int const rc = arrangement
        ? quactlize_ppu_prepare_dense_for_arrangement_v2(
              low_native.data(), HighBits ? high_native.data() : nullptr,
              low_layout + size_t(e) * low_per_expert,
              HighBits ? high_layout + size_t(e) * high_per_expert : nullptr,
              n, k, qtype, arrangement)
        : quactlize_ppu_prepare_dense_for_tile(
              low_native.data(), HighBits ? high_native.data() : nullptr,
              low_layout + size_t(e) * low_per_expert,
              HighBits ? high_layout + size_t(e) * high_per_expert : nullptr,
              n, k, qtype, ppu_formats::for_qtype(qtype).fully_quantized_tile_k);
    if (rc) return rc;
  }
  gguf_scale::unit_pack::pack<T>(blocks, units, n, k, experts);
  return 0;
}

template <KType T>
int recover_fully_quantized(uint8_t const* low_layout, uint8_t const* high_layout,
                            uint8_t const* units, uint8_t* recovered,
                            int n, int k, int experts, int qtype,
                            quactlize_ppu_placed_arrangement_v2 const* arrangement = nullptr,
                            CompleteArtifactSizes const* checked_sizes = nullptr) {
  using R = gguf_scale::unit_pack::Raw<T>;
  using U = gguf_scale::packed_unit::Unit<T>;
  using Tr = gguf_scale::Traits<T>;
  using C = gguf_scale::CodeTraits<T>;
  constexpr int LowBits = (T == KType::Q2_K || T == KType::Q3_K) ? 2 : 4;
  constexpr int HighBits = (T == KType::Q3_K || T == KType::Q5_K) ? 1 : (T == KType::Q6_K ? 2 : 0);
  CompleteArtifactSizes local_sizes;
  if (!checked_sizes) {
    int const rc = complete_artifact_sizes<T>(n, k, experts, &local_sizes);
    if (rc) return rc;
    checked_sizes = &local_sizes;
  }
  size_t const low_per_expert = checked_sizes->low / size_t(experts);
  size_t const high_per_expert = checked_sizes->high / size_t(experts);
  std::vector<uint8_t> low_native(low_per_expert);
  std::vector<uint8_t> high_native(HighBits ? high_per_expert : size_t(1));
  int const superblocks = k / 256;
  int const num_units = superblocks / U::kSbPerUnit;
  std::fill(recovered, recovered + checked_sizes->raw, uint8_t(0));
  for (int e = 0; e < experts; ++e) {
    int const rc = arrangement
        ? quactlize_ppu_recover_dense_for_arrangement_v2(
              low_layout + size_t(e) * low_per_expert,
              HighBits ? high_layout + size_t(e) * high_per_expert : nullptr,
              low_native.data(), HighBits ? high_native.data() : nullptr,
              n, k, qtype, arrangement)
        : quactlize_ppu_recover_dense_for_tile(
              low_layout + size_t(e) * low_per_expert,
              HighBits ? high_layout + size_t(e) * high_per_expert : nullptr,
              low_native.data(), HighBits ? high_native.data() : nullptr,
              n, k, qtype, ppu_formats::for_qtype(qtype).fully_quantized_tile_k);
    if (rc) return rc;
    for (int col = 0; col < n; ++col) {
      for (int sb = 0; sb < superblocks; ++sb) {
        uint8_t* block = recovered + ((int64_t(e) * n + col) * superblocks + sb) * R::kBytes;
        uint8_t const* unit = units +
            ((int64_t(e) * num_units + sb / U::kSbPerUnit) * n + col) * U::kUnitTotal
            + (sb % U::kSbPerUnit) * U::kSbBytes;
        std::memcpy(block + R::kDOffset, unit, 2);
        if constexpr (U::kHasMin) std::memcpy(block + R::kDminOffset, unit + 2, 2);

        uint8_t* scale_block = block + R::kScaleOffset;
        for (int g = 0; g < Tr::kGroups; ++g) {
          int const sc = gguf_scale::packed_unit::code_of<T>(unit, g, 0);
          official_field_put<typename Tr::ScLo>(scale_block, g, sc);
          official_field_put<typename Tr::ScHi>(scale_block, g, sc >> 4);
          if constexpr (Tr::kHasMin) {
            int const mn = gguf_scale::packed_unit::code_of<T>(unit, g, 1);
            official_field_put<typename Tr::MnLo>(scale_block, g, mn);
            official_field_put<typename Tr::MnHi>(scale_block, g, mn >> 4);
          }
        }

        for (int j = 0; j < 256; ++j) {
          int64_t const logical = int64_t(col) * k + int64_t(sb) * 256 + j;
          int q = native_get<LowBits>(low_native.data(), logical);
          if constexpr (HighBits != 0) q |= native_get<HighBits>(high_native.data(), logical) << LowBits;
          int const g = j / Tr::kGroupSize;
          int const word = (j % Tr::kGroupSize) / 4;
          int const lane = j & 3;
          int const coordinate = C::word_coord(g, word);
          official_code_put<typename C::Lo>(block + C::kLoOffset, coordinate, lane, q);
          official_code_put<typename C::Hi>(block + C::kHiOffset, coordinate, lane, q >> LowBits);
        }
      }
    }
  }
  return 0;
}

}  // namespace

extern "C" int64_t quactlize_ppu_units_bytes(int n, int k, int qtype) {
  switch (qtype) {
    case 10: return gguf_scale::unit_pack::bytes<KType::Q2_K>(n, k);
    case 11: return gguf_scale::unit_pack::bytes<KType::Q3_K>(n, k);
    case 12: return gguf_scale::unit_pack::bytes<KType::Q4_K>(n, k);
    case 13: return gguf_scale::unit_pack::bytes<KType::Q5_K>(n, k);
    case 14: return gguf_scale::unit_pack::bytes<KType::Q6_K>(n, k);
    default: return -1;
  }
}

extern "C" int quactlize_ppu_prepare_units(
    uint8_t const* blocks, uint8_t* units, int n, int k, int qtype) {
  return prepare_dispatch(blocks, units, n, k, 1, qtype);
}

extern "C" int quactlize_ppu_prepare_units_grouped(
    uint8_t const* blocks, uint8_t* units, int n, int k, int experts, int qtype) {
  return prepare_dispatch(blocks, units, n, k, experts, qtype);
}

extern "C" int quactlize_ppu_prepare_fully_quantized_v1(
    uint8_t const* blocks, uint8_t* low, uint8_t* high, uint8_t* units,
    int n, int k, int experts, int qtype) {
  if (!blocks || !low || !units || experts <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256) return 20;
  if ((qtype == 11 || qtype == 13 || qtype == 14) && !high) return 21;
#define RUN(T) prepare_fully_quantized<KType::T>(blocks, low, high, units, n, k, experts, qtype)
  switch (qtype) {
    case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
    case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 22;
  }
#undef RUN
}

extern "C" int quactlize_ppu_recover_fully_quantized_v1(
    uint8_t const* low, uint8_t const* high, uint8_t const* units, uint8_t* recovered,
    int n, int k, int experts, int qtype) {
  if (!low || !units || !recovered || experts <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256) return 20;
  if ((qtype == 11 || qtype == 13 || qtype == 14) && !high) return 21;
#define RUN(T) recover_fully_quantized<KType::T>(low, high, units, recovered, n, k, experts, qtype)
  switch (qtype) {
    case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
    case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 22;
  }
#undef RUN
}

extern "C" int quactlize_ppu_prepare_fully_quantized_for_arrangement_v2(
    uint8_t const* blocks, uint8_t* low, uint8_t* high, uint8_t* units,
    int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  if (!arrangement) return 23;
  if (!blocks || !low || !units || experts <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256) return 20;
  if (qtype < 10 || qtype > 14) return 22;
  if (!build_owns_qtype(qtype)) return 29;
  bool const has_high = qtype == 11 || qtype == 13 || qtype == 14;
  if (has_high != (high != nullptr)) return 21;
  quactlize_ppu_placed_arrangement_v2 const descriptor = *arrangement;
  if (!product_arrangement(qtype, k, &descriptor)) return 25;
  CompleteArtifactSizes sizes;
  int rc = complete_artifact_sizes_for_qtype(qtype, n, k, experts, &sizes);
  if (rc) return rc;
  rc = require_distinct_tensor_ranges({
      {blocks, sizes.raw}, {low, sizes.low}, {high, sizes.high}, {units, sizes.units}});
  if (rc) return rc;
#define RUN(T) prepare_fully_quantized<KType::T>( \
    blocks, low, high, units, n, k, experts, qtype, &descriptor, &sizes)
  try {
    switch (qtype) {
      case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
      case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 22;
    }
  } catch (std::bad_alloc const&) {
    return 27;
  } catch (std::length_error const&) {
    return 27;
  } catch (...) {
    return 28;
  }
#undef RUN
}

extern "C" int quactlize_ppu_recover_fully_quantized_for_arrangement_v2(
    uint8_t const* low, uint8_t const* high, uint8_t const* units,
    uint8_t* recovered, int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  if (!arrangement) return 23;
  if (!low || !units || !recovered || experts <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256) return 20;
  if (qtype < 10 || qtype > 14) return 22;
  if (!build_owns_qtype(qtype)) return 29;
  bool const has_high = qtype == 11 || qtype == 13 || qtype == 14;
  if (has_high != (high != nullptr)) return 21;
  quactlize_ppu_placed_arrangement_v2 const descriptor = *arrangement;
  if (!product_arrangement(qtype, k, &descriptor)) return 25;
  CompleteArtifactSizes sizes;
  int rc = complete_artifact_sizes_for_qtype(qtype, n, k, experts, &sizes);
  if (rc) return rc;
  rc = require_distinct_tensor_ranges({
      {low, sizes.low}, {high, sizes.high}, {units, sizes.units}, {recovered, sizes.raw}});
  if (rc) return rc;
#define RUN(T) recover_fully_quantized<KType::T>( \
    low, high, units, recovered, n, k, experts, qtype, &descriptor, &sizes)
  try {
    switch (qtype) {
      case 10: return RUN(Q2_K); case 11: return RUN(Q3_K); case 12: return RUN(Q4_K);
      case 13: return RUN(Q5_K); case 14: return RUN(Q6_K); default: return 22;
    }
  } catch (std::bad_alloc const&) {
    return 27;
  } catch (std::length_error const&) {
    return 27;
  } catch (...) {
    return 28;
  }
#undef RUN
}
