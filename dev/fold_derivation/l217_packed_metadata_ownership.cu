// L217 -- packed metadata copy/decode ownership for CTAs narrower than TileN.
//
// The device failure that motivated this oracle was exact:
//   TM8/TN64/WM8/WN64 -> 32 CTA threads
//   old copy layout    -> one owner per N column (64 owners)
//   owners that exist  -> columns 0..31
//   decoder reads      -> columns 0..63
// Therefore 32 metadata columns were never written, the first bad output was
// N=32, and exactly half of a 1024-column output failed.  The negative build
// below must reproduce that same ownership signature.

#include <array>
#include <cstdio>

#include "cute/atom/copy_atom.hpp"
#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_packed_metadata_ownership.hpp"

using namespace cute;
namespace packed_detail = cutlass::gemm::collective::detail;

#ifndef L217_LEGACY_ONE_COLUMN
#define L217_LEGACY_ONE_COLUMN 0
#endif

namespace {

constexpr int kUnitBytes = 16;

struct Totals {
  int cases = 0;
  int copy_missing = 0;
  int decode_missing = 0;
  int unowned_reads = 0;
  int map_bad = 0;
  int predicate_bad = 0;
  int first_predicate_bad = -1;
};

template <int TileN, int Threads>
Totals verify_case() {
  using Production = packed_detail::PackedMetadataColumnOwnership<TileN, Threads>;
  constexpr int owners = L217_LEGACY_ONE_COLUMN ? TileN : Production::owner_threads;
  constexpr int cpt = L217_LEGACY_ONE_COLUMN ? 1 : Production::columns_per_thread;
  using Copy = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cute::uint128_t>, uint8_t>{},
      Layout<Shape<Int<owners>, _1>>{},
      Layout<Shape<_1, Int<kUnitBytes>>>{}));
  static_assert(int(size(Copy{})) == owners);

  std::array<int, TileN> copied{};
  std::array<int, TileN> decoded{};
  std::array<int, Threads * TileN> copied_by_thread{};
  std::array<uint8_t, TileN * kUnitBytes> pred_src{};
  std::array<uint8_t, TileN * kUnitBytes> pred_dst{};
  auto identity = make_identity_tensor(make_shape(Int<TileN>{}, Int<kUnitBytes>{}));
  auto byte_layout = make_layout(
      make_shape(Int<TileN>{}, Int<kUnitBytes>{}),
      make_stride(Int<kUnitBytes>{}, _1{}));
  auto src_tensor = make_tensor(pred_src.data(), byte_layout);
  auto dst_tensor = make_tensor(pred_dst.data(), byte_layout);
  for (int n = 0; n < TileN; ++n)
    for (int b = 0; b < kUnitBytes; ++b)
      pred_src[std::size_t(n * kUnitBytes + b)] = uint8_t(1 + (n + b) % 251);
  constexpr int residue = TileN - TileN / 4;

  Totals z{};
  z.cases = 1;
  for (int thread = 0; thread < Threads; ++thread) {
    auto thr = Copy{}.get_slice(thread % owners);
    auto part = thr.partition_S(identity);
    for (int i = 0; i < int(size(part)); ++i) {
      auto coord = part(i);
      int const n = int(get<0>(coord));
      if (n < 0 || n >= TileN) {
        ++z.map_bad;
        continue;
      }
      copied[std::size_t(n)] = 1;
      copied_by_thread[std::size_t(thread * TileN + n)] = 1;
      int const owner = thread % owners;
      bool expected = false;
      for (int sub = 0; sub < cpt; ++sub)
        expected |= n == Production::column(owner, sub);
      z.map_bad += !expected;
    }

    auto src_part = thr.partition_S(src_tensor);
    auto dst_part = thr.partition_D(dst_tensor);
    bool const first_column_valid =
        int(get<0>(part(0, 0, 0))) < residue;
    packed_detail::copy_packed_metadata_if<cpt>(
        Copy{}, first_column_valid, residue, src_part, dst_part, part);
  }

  // This is the production decoder's ownership loop. With the corrected cpt,
  // every thread reads exactly the strided column set its TiledCopy slice wrote.
  if constexpr (L217_LEGACY_ONE_COLUMN) {
    for (int thread = 0; thread < Threads; ++thread) {
      for (int n = thread; n < TileN; n += Threads) {
        ++decoded[std::size_t(n)];
        z.unowned_reads +=
            copied_by_thread[std::size_t(thread * TileN + n)] == 0;
      }
    }
  } else {
    for (int thread = 0; thread < Threads && thread < owners; ++thread) {
      for (int sub = 0; sub < cpt; ++sub) {
        int const n = Production::column(thread, sub);
        ++decoded[std::size_t(n)];
        z.unowned_reads +=
            copied_by_thread[std::size_t(thread * TileN + n)] == 0;
      }
    }
  }

  int first_copy_missing = -1;
  for (int n = 0; n < TileN; ++n) {
    if (!copied[std::size_t(n)]) {
      ++z.copy_missing;
      if (first_copy_missing < 0) first_copy_missing = n;
    }
    z.decode_missing += decoded[std::size_t(n)] != 1;
    for (int b = 0; b < kUnitBytes; ++b) {
      uint8_t const want = n < residue
          ? pred_src[std::size_t(n * kUnitBytes + b)] : uint8_t(0);
      if (pred_dst[std::size_t(n * kUnitBytes + b)] != want) {
        ++z.predicate_bad;
        if (z.first_predicate_bad < 0)
          z.first_predicate_bad = n * kUnitBytes + b;
      }
    }
  }
  std::printf(
      "L217_CASE tile_n=%d threads=%d owners=%d cpt=%d copy_missing=%d "
      "first_copy_missing=%d decode_missing=%d unowned_reads=%d map_bad=%d "
      "predicate_bad=%d first_predicate_bad=%d\n",
      TileN, Threads, owners, cpt, z.copy_missing, first_copy_missing,
      z.decode_missing, z.unowned_reads, z.map_bad, z.predicate_bad,
      z.first_predicate_bad);
  return z;
}

void add(Totals& a, Totals const& b) {
  a.cases += b.cases;
  a.copy_missing += b.copy_missing;
  a.decode_missing += b.decode_missing;
  a.unowned_reads += b.unowned_reads;
  a.map_bad += b.map_bad;
  a.predicate_bad += b.predicate_bad;
  if (a.first_predicate_bad < 0 && b.first_predicate_bad >= 0)
    a.first_predicate_bad = b.first_predicate_bad;
}

}  // namespace

int main() {
  Totals z{};
  add(z, verify_case<32, 32>());
  add(z, verify_case<64, 32>());   // exact TM8/TN64/WN64 failure geometry
  add(z, verify_case<128, 32>());
  add(z, verify_case<64, 64>());
  add(z, verify_case<64, 128>());
  add(z, verify_case<128, 256>());
  bool const ok = z.copy_missing == 0 && z.decode_missing == 0 &&
                  z.unowned_reads == 0 && z.map_bad == 0 &&
                  z.predicate_bad == 0;
  std::printf(
      "L217_SUMMARY variant=%s cases=%d copy_missing=%d decode_missing=%d "
      "unowned_reads=%d map_bad=%d predicate_bad=%d verdict=%s\n",
      L217_LEGACY_ONE_COLUMN ? "legacy-one-column" : "derived-ownership",
      z.cases, z.copy_missing, z.decode_missing, z.unowned_reads,
      z.map_bad, z.predicate_bad, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
