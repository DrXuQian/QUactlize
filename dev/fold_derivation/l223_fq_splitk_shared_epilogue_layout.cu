// L223 -- exact CuTe mapping oracle for the legacy Q4_K Split-K shared epilogue.
//
// This binds the frozen TM8/TN64/WM8/WN16 tactic and reproduces the
// EpilogueParallel R2S and S2R tensor partitions without executing a device
// barrier.  It proves or refutes static shared-address aliases, holes and
// register/coordinate permutations before a device ordering experiment is
// allowed to carry the root-cause claim.

#include <array>
#include <cstdint>
#include <cstdio>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"

namespace {
using namespace cute;

constexpr int kTM = 8;
constexpr int kTN = 64;
constexpr int kWM = 8;
constexpr int kWN = 16;
constexpr int kThreads = 128;

using Atom = PPU0010_8x16x16_F32F16F16F32_TN;
using Mma = TiledMMA<
    MMA_Atom<Atom>,
    Layout<Shape<Int<kTM / kWM>, Int<kTN / kWN>, _1>>,
    Tile<Int<(kTM / kWM) * 8>, Int<(kTN / kWN) * 16>, _16>>;
using Fragment = decltype(partition_fragment_C(
    Mma{}, Shape<Int<kTM>, Int<kTN>>{}));

using CopyInst = AutoVectorizingCopyWithAssumedAlignment<128>;
// Expand DefaultGemm_Epilogue_Configuration here instead of including the
// complete GEMM configuration graph.  These are its exact instantiated types
// for ElementAcc=float, TM8, TN64, CTA128, Alignment4 and InstM8.
using SmemLayoutAtom = decltype(composition(
    Swizzle<3, 2, 3>{},
    Layout<Shape<_8, Int<32>>, Stride<Int<32>, _1>>{}));
using SmemLayout = decltype(tile_to_shape(
    SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTN>>{}));
using R2SAtom = Copy_Atom<CopyInst, float>;
using S2RCopy = decltype(make_tiled_copy(
    Copy_Atom<CopyInst, float>{},
    Layout<Shape<Int<8>, Int<16>>, Stride<Int<16>, _1>>{},
    Layout<Shape<_1, Int<4>>>{}));

#ifndef L223_BAD_R2S_ROTATE
#define L223_BAD_R2S_ROTATE 0
#endif
#ifndef L223_BAD_S2R_THREAD_MODULO
#define L223_BAD_S2R_THREAD_MODULO 0
#endif

static_assert(size(Mma{}) == kThreads);
static_assert(size(Fragment{}) * kThreads == kTM * kTN);
static_assert(size(SmemLayout{}) == kTM * kTN);
static_assert(int(S2RCopy::TiledNumThr{}) == kThreads);

float marker(int m, int n) {
  std::uint32_t bits = UINT32_C(0x3f000001) +
                       std::uint32_t(m * kTN + n);
  float value;
  __builtin_memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint32_t bits(float value) {
  std::uint32_t result;
  __builtin_memcpy(&result, &value, sizeof(result));
  return result;
}

template <class Coord>
int linear_coord(Coord coord) {
  return int(get<0>(coord)) * kTN + int(get<1>(coord));
}

int marker_coord(float value) {
  return int(bits(value) - UINT32_C(0x3f000001));
}

}  // namespace

int main() {
  std::array<float, kTM * kTN> shared{};
  std::array<int, kTM * kTN> writers{};
  std::array<int, kTM * kTN> readers{};
  std::array<int, kTM * kTN> shared_logical{};
  shared_logical.fill(-1);

  auto sC = make_tensor(shared.data(), SmemLayout{});
  auto identity = make_identity_tensor(Shape<Int<kTM>, Int<kTN>>{});
  auto tiled_r2s = make_tiled_copy_C(R2SAtom{}, Mma{});

  int r2s_conflicts = 0;
  for (int thread = 0; thread < kThreads; ++thread) {
    auto thread_mma = Mma{}.get_thread_slice(thread);
    auto coordinates = thread_mma.partition_C(identity);
    Fragment accumulators;
    for (int i = 0; i < int(size(accumulators)); ++i) {
      auto coord = coordinates(i);
      accumulators(i) = marker(int(get<0>(coord)), int(get<1>(coord)));
    }

    auto thread_r2s = tiled_r2s.get_thread_slice(thread);
    auto source = thread_r2s.retile_S(accumulators);
    auto destination = thread_r2s.partition_D(sC);
    static_assert(size(decltype(source){}) == size(decltype(destination){}));
    for (int i = 0; i < int(size(source)); ++i) {
      int const offset = int(&destination(i) - shared.data());
      int const source_i =
#if L223_BAD_R2S_ROTATE
          (i + 1) % int(size(source));
#else
          i;
#endif
      int const logical = marker_coord(source(source_i));
      if (offset < 0 || offset >= int(shared.size())) {
        ++r2s_conflicts;
        continue;
      }
      ++writers[std::size_t(offset)];
      if (shared_logical[std::size_t(offset)] >= 0 &&
          shared_logical[std::size_t(offset)] != logical) {
        ++r2s_conflicts;
      }
      shared_logical[std::size_t(offset)] = logical;
      destination(i) = source(source_i);
    }
  }

  int writer_holes = 0;
  int writer_duplicates = 0;
  for (int count : writers) {
    writer_holes += count == 0;
    writer_duplicates += count > 1;
  }

  int s2r_value_bad = 0;
  int s2r_coord_bad = 0;
  auto tiled_s2r = S2RCopy{};
  auto cD = make_identity_tensor(Shape<Int<kTM>, Int<kTN>>{});
  auto tile = make_shape(size<0>(sC), size<1>(sC));
  auto cDt = flat_divide(cD, tile);
  for (int thread = 0; thread < kThreads; ++thread) {
    int const s2r_thread =
#if L223_BAD_S2R_THREAD_MODULO
        thread % 64;
#else
        thread;
#endif
    auto thread_s2r = tiled_s2r.get_thread_slice(s2r_thread);
    auto source = thread_s2r.partition_S(sC);
    auto coordinates = thread_s2r.partition_D(cDt);
    auto output_coordinates = coordinates(_, _, _, 0, 0);
    static_assert(size(decltype(source){}) ==
                  size(decltype(output_coordinates){}));
    for (int i = 0; i < int(size(source)); ++i) {
      int const offset = int(&source(i) - shared.data());
      int const logical = linear_coord(output_coordinates(i));
      if (logical < 0 || logical >= int(readers.size()) ||
          offset < 0 || offset >= int(shared.size())) {
        ++s2r_coord_bad;
        continue;
      }
      ++readers[std::size_t(logical)];
      s2r_coord_bad += shared_logical[std::size_t(offset)] != logical;
      s2r_value_bad += bits(source(i)) !=
                       bits(marker(logical / kTN, logical % kTN));
    }
  }

  int reader_holes = 0;
  int reader_duplicates = 0;
  for (int count : readers) {
    reader_holes += count == 0;
    reader_duplicates += count > 1;
  }

  bool const pass = writer_holes == 0 && writer_duplicates == 0 &&
                    r2s_conflicts == 0 && reader_holes == 0 &&
                    reader_duplicates == 0 && s2r_coord_bad == 0 &&
                    s2r_value_bad == 0;
  std::printf(
      "L223_SHARED_EPILOGUE_LAYOUT writers=%d holes=%d duplicates=%d "
      "r2s_conflicts=%d readers=%d reader_holes=%d reader_duplicates=%d "
      "s2r_coord_bad=%d s2r_value_bad=%d verdict=%s\n",
      kTM * kTN, writer_holes, writer_duplicates, r2s_conflicts,
      kTM * kTN, reader_holes, reader_duplicates,
      s2r_coord_bad, s2r_value_bad, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
