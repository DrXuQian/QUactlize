// L207 -- exhaust the generated-operand register maps of the production PPU
// chunked-GDN TiledMma.  This is a host algebra oracle: it instantiates the
// exact __HGGCCC__ builder type but executes no PPU instruction.
//
// The independent anchor below does not call a second CuTe partition method.
// It spells out the public PPU0010 m16n16k16 TN atom map and the production
// 2M x 2N warp topology.  Therefore a self-consistent change to partition_A,
// partition_B, or partition_C cannot silently bless itself.

#include <array>
#include <cstdint>
#include <cstdio>

#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_collective.cuh"

#if !defined(__HGGCCC__)
#error "L207 must instantiate the production __HGGCCC__ CollectiveBuilder branch"
#endif

#if !defined(__CUDA_ARCH__)
namespace {

using Mainloop =
    cutlass::linear_attention::detail::PpuChunkedGdnGlobalDotMainloop;
using Mma = typename Mainloop::TiledMma;
using ProductionTile =
    cutlass::linear_attention::detail::PpuChunkedGdnGlobalDotTile;
using ProductionWarp =
    cutlass::linear_attention::detail::PpuChunkedGdnGlobalDotWarp;

static_assert(cute::size(Mma{}) == 128);
static_assert(cute::size<0>(typename Mma::AtomShape_MNK{}) == 16 &&
              cute::size<1>(typename Mma::AtomShape_MNK{}) == 16 &&
              cute::size<2>(typename Mma::AtomShape_MNK{}) == 16);
static_assert(cute::size<0>(ProductionTile{}) == 64 &&
              cute::size<1>(ProductionTile{}) == 64 &&
              cute::size<2>(ProductionTile{}) == 64);
static_assert(cute::size<0>(ProductionWarp{}) == 32 &&
              cute::size<1>(ProductionWarp{}) == 32 &&
              cute::size<2>(ProductionWarp{}) == 64);

constexpr int kExtent = 64;
constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);
constexpr std::uint64_t kExpectedAHash = UINT64_C(0x29a6a34b79ebcb25);
constexpr std::uint64_t kExpectedBHash = UINT64_C(0x70eb86e99be9bb25);
constexpr std::uint64_t kExpectedCHash = UINT64_C(0xee9011938bb9c325);

struct Coordinate {
  int row;
  int column;
};

void hash_u32(std::uint64_t& hash, std::uint32_t value) {
  for (int byte = 0; byte < 4; ++byte) {
    hash ^= (value >> (8 * byte)) & 0xffu;
    hash *= kFnvPrime;
  }
}

void hash_coordinate(
    std::uint64_t& hash, int thread, int slot, Coordinate coordinate) {
  hash_u32(hash, std::uint32_t(thread));
  hash_u32(hash, std::uint32_t(slot));
  hash_u32(hash, std::uint32_t(coordinate.row));
  hash_u32(hash, std::uint32_t(coordinate.column));
}

// Independent expansion of:
//   atom A/B: ((4,8),(2,2,2)):((32,1),(16,128,8))
//   atom C:   ((4,8),(4,2)):((16,1),(64,8))
// and the production TiledMMA Layout<Shape<2,2,1>>.
// Physical warp order is m + 2*n.  A is consequently duplicated across the
// two N warps, while B is duplicated across the two M warps.
Coordinate expected_operand(bool is_b, int thread, int slot) {
  int const lane = thread % 32;
  int const warp = thread / 32;
  int const warp_m = warp % 2;
  int const warp_n = warp / 2;
  int const warp_axis = is_b ? warp_n : warp_m;
  int const value = slot % 16;
  int const outer_k = slot / 16;
  return {
      lane / 4 + 16 * warp_axis + 8 * ((value / 4) % 2) +
          32 * (value / 8),
      2 * (lane % 4) + (value % 2) + 8 * ((value / 2) % 2) +
          16 * outer_k};
}

Coordinate expected_output(int thread, int slot) {
  int const lane = thread % 32;
  int const warp = thread / 32;
  int const warp_m = warp % 2;
  int const warp_n = warp / 2;
  int const value = slot % 16;
  return {
      lane / 4 + 16 * warp_m + 8 * ((value / 4) % 2) +
          32 * (value / 8),
      lane % 4 + 16 * warp_n + 4 * (value % 4) + 32 * (slot / 16)};
}

struct Coverage {
  int visits = 0;
  int holes = 0;
  int duplicate_coordinates = 0;
  int duplicate_visits = 0;
  int out_of_bounds = 0;
  int min_visits = 0;
  int max_visits = 0;
  int coordinate_mismatches = 0;
  std::uint64_t map_hash = kFnvOffset;
  std::uint64_t expected_hash = kFnvOffset;
};

template <bool IsB, int Threads, bool Transpose, int ReductionStride>
Coverage enumerate_operand() {
  std::array<int, kExtent * kExtent> owners{};
  Coverage result{};
  auto identity = cute::make_identity_tensor(
      cute::Shape<cute::_64, cute::_64>{});

  for (int thread = 0; thread < Threads; ++thread) {
    auto part = [&] {
      if constexpr (IsB) {
        return Mma{}.get_thread_slice(thread).partition_B(identity);
      } else {
        return Mma{}.get_thread_slice(thread).partition_A(identity);
      }
    }();
    static_assert(decltype(cute::size(part))::value == 64,
                  "each resident operand fragment must own 64 BF16 values");
    for (int slot = 0; slot < int(cute::size(part)); ++slot) {
      auto const raw = part(slot);
      Coordinate actual{int(cute::get<0>(raw)), int(cute::get<1>(raw))};
      if constexpr (Transpose) {
        int const tmp = actual.row;
        actual.row = actual.column;
        actual.column = tmp;
      }
      actual.column = (actual.column * ReductionStride) % kExtent;
      Coordinate const expected = expected_operand(IsB, thread, slot);
      result.coordinate_mismatches +=
          actual.row != expected.row || actual.column != expected.column;
      hash_coordinate(result.map_hash, thread, slot, actual);
      hash_coordinate(result.expected_hash, thread, slot, expected);
      if (actual.row < 0 || actual.row >= kExtent ||
          actual.column < 0 || actual.column >= kExtent) {
        ++result.out_of_bounds;
        continue;
      }
      ++owners[std::size_t(actual.row * kExtent + actual.column)];
      ++result.visits;
    }
  }
  result.min_visits = result.visits == 0 ? 0 : result.visits;
  for (int count : owners) {
    result.holes += count == 0;
    result.duplicate_coordinates += count > 2;
    result.duplicate_visits += count > 2 ? count - 2 : 0;
    result.min_visits = count < result.min_visits ? count : result.min_visits;
    result.max_visits = count > result.max_visits ? count : result.max_visits;
  }
  return result;
}

template <int Threads>
Coverage enumerate_output() {
  std::array<int, kExtent * kExtent> owners{};
  Coverage result{};
  auto identity = cute::make_identity_tensor(
      cute::Shape<cute::_64, cute::_64>{});
  for (int thread = 0; thread < Threads; ++thread) {
    auto part = Mma{}.get_thread_slice(thread).partition_C(identity);
    static_assert(decltype(cute::size(part))::value == 32,
                  "each accumulator fragment must own 32 FP32 values");
    for (int slot = 0; slot < int(cute::size(part)); ++slot) {
      auto const raw = part(slot);
      Coordinate const actual{
          int(cute::get<0>(raw)), int(cute::get<1>(raw))};
      Coordinate const expected = expected_output(thread, slot);
      result.coordinate_mismatches +=
          actual.row != expected.row || actual.column != expected.column;
      hash_coordinate(result.map_hash, thread, slot, actual);
      hash_coordinate(result.expected_hash, thread, slot, expected);
      if (actual.row < 0 || actual.row >= kExtent ||
          actual.column < 0 || actual.column >= kExtent) {
        ++result.out_of_bounds;
        continue;
      }
      ++owners[std::size_t(actual.row * kExtent + actual.column)];
      ++result.visits;
    }
  }
  result.min_visits = result.visits == 0 ? 0 : result.visits;
  for (int count : owners) {
    result.holes += count == 0;
    result.duplicate_coordinates += count > 1;
    result.duplicate_visits += count > 1 ? count - 1 : 0;
    result.min_visits = count < result.min_visits ? count : result.min_visits;
    result.max_visits = count > result.max_visits ? count : result.max_visits;
  }
  return result;
}

bool operand_ok(Coverage const& x, std::uint64_t expected_hash) {
  return x.visits == 8192 && x.holes == 0 &&
         x.duplicate_coordinates == 0 && x.duplicate_visits == 0 &&
         x.out_of_bounds == 0 && x.min_visits == 2 && x.max_visits == 2 &&
         x.coordinate_mismatches == 0 && x.map_hash == x.expected_hash &&
         x.map_hash == expected_hash;
}

bool output_ok(Coverage const& x) {
  return x.visits == 4096 && x.holes == 0 &&
         x.duplicate_coordinates == 0 && x.duplicate_visits == 0 &&
         x.out_of_bounds == 0 && x.min_visits == 1 && x.max_visits == 1 &&
         x.coordinate_mismatches == 0 && x.map_hash == x.expected_hash &&
         x.map_hash == kExpectedCHash;
}

void print_operand(char const* name, Coverage const& x) {
  std::printf(
      "%s=(visits=%d,holes=%d,dup_coord=%d,dup_visits=%d,oob=%d,min=%d,max=%d,coord_bad=%d,map=%016llx,anchor=%016llx)",
      name, x.visits, x.holes, x.duplicate_coordinates, x.duplicate_visits,
      x.out_of_bounds, x.min_visits, x.max_visits,
      x.coordinate_mismatches,
      static_cast<unsigned long long>(x.map_hash),
      static_cast<unsigned long long>(x.expected_hash));
}

}  // namespace

int main() {
#if defined(L207_PLANT_THREAD_COUNT)
  constexpr int kThreads = 64;
#else
  constexpr int kThreads = 128;
#endif
#if defined(L207_PLANT_B_TRANSPOSE)
  constexpr bool kTransposeB = true;
#else
  constexpr bool kTransposeB = false;
#endif
#if defined(L207_PLANT_COORDINATE_STRIDE)
  constexpr int kReductionStride = 2;
#else
  constexpr int kReductionStride = 1;
#endif

  Coverage const a =
      enumerate_operand<false, kThreads, false, 1>();
  Coverage const b =
      enumerate_operand<true, kThreads, kTransposeB, kReductionStride>();
  Coverage const c = enumerate_output<kThreads>();
  bool const pass = operand_ok(a, kExpectedAHash) &&
                    operand_ok(b, kExpectedBHash) && output_ok(c);

  std::printf("[l207] %s: source=production-__HGGCCC__-CollectiveBuilder::TiledMma tile=64x64x64 warp=32x32x64 threads=%d resident=A@V ",
              pass ? "PASS" : "FAIL", kThreads);
  print_operand("A", a);
  std::printf(" ");
  print_operand("B", b);
  std::printf(" ");
  print_operand("C", c);
  std::printf(" anchor=public-PPU0010-atom+2Mx2N-warp-topology\n");
  return pass ? 0 : 1;
}
#endif
