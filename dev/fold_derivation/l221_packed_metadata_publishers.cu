// L221 -- exact CuTe publisher multiplicity for the Q4_K Split-K failure row.
//
// Shipping geometry:
//   TileN=64, CTA=128, one 16-byte packed metadata unit per N column.
// The legacy protocol calls get_slice(thread % 64) from all 128 threads.
// Coverage is complete, but every physical shared-memory destination has two
// asynchronous publishers.  Decoder warp 1 owns local columns 32..63, which
// is the exact first-bad boundary observed on all three corrupt partial-plane
// samples (global indices 160, 288 and 416).

#include <array>
#include <cstdio>

#include "cute/atom/copy_atom.hpp"
#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_packed_metadata_ownership.hpp"

using namespace cute;
namespace packed_detail = cutlass::gemm::collective::detail;

namespace {

constexpr int kTileN = 64;
constexpr int kThreads = 128;
constexpr int kUnitBytes = 16;
using Ownership = packed_detail::PackedMetadataColumnOwnership<kTileN, kThreads>;
using Copy = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<cute::uint128_t>, uint8_t>{},
    Layout<Shape<Int<Ownership::owner_threads>, _1>>{},
    Layout<Shape<_1, Int<kUnitBytes>>>{}));

struct Stats {
  int unique = 0;
  int visits = 0;
  int duplicate_visits = 0;
  int min_hits = 99;
  int max_hits = 0;
  int decoder_read_bytes = 0;
  int read_duplicate_writer_overlap = 0;
  int first_second_decoder_warp_column = -1;
};

template <bool OwnerOnly>
Stats enumerate() {
  auto identity = make_identity_tensor(make_shape(Int<kTileN>{}, Int<kUnitBytes>{}));
  std::array<int, kTileN * kUnitBytes> hits{};
  std::array<int, kTileN * kUnitBytes> duplicate_writer{};

  for (int thread = 0; thread < kThreads; ++thread) {
    if constexpr (OwnerOnly) {
      if (!Ownership::owns_physical_thread(thread)) continue;
    }
    auto part = Copy{}.get_slice(Ownership::copy_owner(thread)).partition_S(identity);
    for (int i = 0; i < int(size(part)); ++i) {
      auto coord = part(i);
      int const n = int(get<0>(coord));
      int const b = int(get<1>(coord));
      int const offset = n * kUnitBytes + b;
      ++hits[std::size_t(offset)];
      if (thread >= Ownership::owner_threads)
        duplicate_writer[std::size_t(offset)] = 1;
    }
  }

  Stats z{};
  for (int offset = 0; offset < int(hits.size()); ++offset) {
    int const count = hits[std::size_t(offset)];
    z.unique += count != 0;
    z.visits += count;
    z.duplicate_visits += count > 1 ? count - 1 : 0;
    z.min_hits = count < z.min_hits ? count : z.min_hits;
    z.max_hits = count > z.max_hits ? count : z.max_hits;
  }

  // The production non-split decoder admits only the 64 real owner threads.
  // Each reads the bytes from exactly its own CuTe copy slice.
  for (int thread = 0; thread < Ownership::owner_threads; ++thread) {
    auto part = Copy{}.get_slice(thread).partition_S(identity);
    for (int i = 0; i < int(size(part)); ++i) {
      auto coord = part(i);
      int const n = int(get<0>(coord));
      int const b = int(get<1>(coord));
      int const offset = n * kUnitBytes + b;
      ++z.decoder_read_bytes;
      z.read_duplicate_writer_overlap +=
          duplicate_writer[std::size_t(offset)];
      if (thread >= 32 && z.first_second_decoder_warp_column < 0)
        z.first_second_decoder_warp_column = n;
    }
  }
  return z;
}

void print(char const* variant, Stats const& z) {
  std::printf(
      "L221_PUBLISHERS variant=%s tile_n=%d cta=%d owners=%d "
      "unique=%d visits=%d duplicate_visits=%d hits=%d..%d "
      "decoder_read_bytes=%d read_duplicate_writer_overlap=%d "
      "first_second_decoder_warp_column=%d\n",
      variant, kTileN, kThreads, Ownership::owner_threads,
      z.unique, z.visits, z.duplicate_visits, z.min_hits, z.max_hits,
      z.decoder_read_bytes, z.read_duplicate_writer_overlap,
      z.first_second_decoder_warp_column);
}

}  // namespace

int main() {
  static_assert(Ownership::owner_threads == 64);
  static_assert(Ownership::columns_per_thread == 1);
  static_assert(int(size(Copy{})) == 64);
  auto const legacy = enumerate<false>();
  auto const candidate = enumerate<true>();
  print("legacy-modulo-all", legacy);
  print("owner-only", candidate);
  bool const legacy_red =
      legacy.unique == 1024 && legacy.visits == 2048 &&
      legacy.duplicate_visits == 1024 && legacy.min_hits == 2 &&
      legacy.max_hits == 2 && legacy.decoder_read_bytes == 1024 &&
      legacy.read_duplicate_writer_overlap == 1024 &&
      legacy.first_second_decoder_warp_column == 32;
  bool const candidate_exact =
      candidate.unique == 1024 && candidate.visits == 1024 &&
      candidate.duplicate_visits == 0 && candidate.min_hits == 1 &&
      candidate.max_hits == 1 && candidate.decoder_read_bytes == 1024 &&
      candidate.read_duplicate_writer_overlap == 0 &&
      candidate.first_second_decoder_warp_column == 32;
  std::printf("L221_SUMMARY legacy=%s candidate=%s verdict=%s\n",
              legacy_red ? "RED" : "FALSE-GREEN",
              candidate_exact ? "EXACT" : "FAIL",
              legacy_red && candidate_exact ? "PASS" : "FAIL");
  return legacy_red && candidate_exact ? 0 : 1;
}
