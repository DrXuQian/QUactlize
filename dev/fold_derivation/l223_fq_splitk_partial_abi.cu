// L223 -- end-to-end CuTe proof for the fixed Split-K FP32 partial ABI.
//
// This composes the production TiledMma::partition_C ownership, all CTA tiles,
// every split plane, the compact [M,N,S] CuTe layout, and the standalone
// reducer's linear address.  Coordinate-tagged values make a plane swap or a
// register-coordinate error observable even when coverage remains exact-once.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/detail/ppu_splitk_direct_accumulator_store.hpp"

namespace {
using namespace cute;

constexpr int kTM = 8;
constexpr int kTN = 64;
constexpr int kWM = 8;
constexpr int kWN = 16;

using Atom = PPU0010_8x16x16_F32F16F16F32_TN;
using Mma = TiledMMA<
    MMA_Atom<Atom>,
    Layout<Shape<Int<kTM / kWM>, Int<kTN / kWN>, _1>>,
    Tile<Int<(kTM / kWM) * 8>, Int<(kTN / kWN) * 16>, _16>>;
using Fragment = decltype(partition_fragment_C(
    Mma{}, Shape<Int<kTM>, Int<kTN>>{}));
using PartialStride = Stride<int64_t, _1, int64_t>;

static_assert(size(Mma{}) == 128);
static_assert(size(Fragment{}) == 4);
static_assert(size(Mma{}) * size(Fragment{}) == kTM * kTN);

struct DirectParams {
  float* ptr_D;
  PartialStride dD;
};

#ifndef L223_BAD_PLANE_STRIDE
#define L223_BAD_PLANE_STRIDE 0
#endif
#ifndef L223_BAD_PLANE_SELECT
#define L223_BAD_PLANE_SELECT 0
#endif
#ifndef L223_BAD_REDUCER_PITCH
#define L223_BAD_REDUCER_PITCH 0
#endif

constexpr std::uint32_t kPoison = UINT32_C(0x7fc22323);

std::uint32_t bits(float value) {
  std::uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

float marker(int split, int row, int column, int rows, int columns) {
  std::uint32_t const ordinal =
      std::uint32_t((split * rows + row) * columns + column + 1);
  std::uint32_t const word = UINT32_C(0x3f000000) + ordinal;
  float value;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

struct Result {
  std::int64_t logical_visits = 0;
  std::int64_t physical_holes = 0;
  std::int64_t physical_duplicates = 0;
  std::int64_t value_bad = 0;
  std::int64_t cute_manual_bad = 0;
  std::int64_t fast_reducer_bad = 0;
  std::int64_t invalid_offsets = 0;
  bool compact = false;
};

Result verify(int rows, int columns, int splits) {
  std::size_t const elements = std::size_t(rows) * columns * splits;
  float poison;
  std::memcpy(&poison, &kPoison, sizeof(poison));
  std::vector<float> output(elements, poison);
  std::vector<int> physical_coverage(elements, 0);

  PartialStride stride = cutlass::gemm::kernel::detail::
      make_compact_fp32_partial_stride<PartialStride>(rows, columns);
#if L223_BAD_PLANE_STRIDE
  get<2>(stride) = 0;
#endif
  Result result;
  result.compact = cutlass::gemm::kernel::detail::
      is_compact_fp32_partial_stride(stride, rows, columns);
  DirectParams params{output.data(), stride};

  int const m_tiles = (rows + kTM - 1) / kTM;
  int const n_tiles = (columns + kTN - 1) / kTN;
  for (int split = 0; split < splits; ++split) {
    int const store_plane =
#if L223_BAD_PLANE_SELECT
        (split + 1) % splits;
#else
        split;
#endif
    for (int tile_m = 0; tile_m < m_tiles; ++tile_m) {
      for (int tile_n = 0; tile_n < n_tiles; ++tile_n) {
        int const residue_m = rows - tile_m * kTM;
        int const residue_n = columns - tile_n * kTN;
        auto identity = make_identity_tensor(Shape<Int<kTM>, Int<kTN>>{});
        for (int thread = 0; thread < int(size(Mma{})); ++thread) {
          auto thread_mma = Mma{}.get_thread_slice(thread);
          auto coordinates = thread_mma.partition_C(identity);
          Fragment accumulators;
          for (int i = 0; i < int(size(accumulators)); ++i) {
            auto coord = coordinates(i);
            int const local_m = int(get<0>(coord));
            int const local_n = int(get<1>(coord));
            int const row = tile_m * kTM + local_m;
            int const column = tile_n * kTN + local_n;
            accumulators(i) = marker(split, row, column, rows, columns);
            if (local_m < residue_m && local_n < residue_n) {
              ++result.logical_visits;
              std::int64_t const offset =
                  cutlass::gemm::kernel::detail::fp32_partial_cute_offset(
                      stride, rows, columns, splits,
                      row, column, store_plane);
              if (offset < 0 || std::uint64_t(offset) >= elements) {
                ++result.invalid_offsets;
              } else {
                ++physical_coverage[std::size_t(offset)];
              }
            }
          }
          cutlass::gemm::kernel::detail::store_splitk_accumulators_direct(
              params, make_shape(rows, columns, 5120, splits),
              Shape<Int<kTM>, Int<kTN>, Int<256>>{},
              make_coord(tile_m, tile_n, _, Int<0>{}),
              accumulators, Mma{},
              make_tuple(residue_m, residue_n, 0), store_plane, thread);
        }
      }
    }
  }

  for (int split = 0; split < splits; ++split) {
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        std::size_t const canonical =
            (std::size_t(split) * rows + row) * columns + column;
        std::int64_t const cute_offset =
            cutlass::gemm::kernel::detail::fp32_partial_cute_offset(
                stride, rows, columns, splits, row, column, split);
        std::int64_t reducer_columns = columns;
#if L223_BAD_REDUCER_PITCH
        ++reducer_columns;
#endif
        std::int64_t const reducer_offset =
            cutlass::gemm::kernel::detail::fp32_partial_linear_offset(
                split, row, column, rows, reducer_columns);
        result.cute_manual_bad +=
            cute_offset != std::int64_t(canonical) ||
            reducer_offset != std::int64_t(canonical);
        result.physical_holes +=
            physical_coverage[canonical] == 0;
        result.physical_duplicates +=
            physical_coverage[canonical] > 1;
        result.value_bad +=
            bits(output[canonical]) !=
            bits(marker(split, row, column, rows, columns));
      }
    }
  }

  // The M=1 fast reducer assigns two adjacent N elements to each lane.  Prove
  // that its vector address is the same CuTe coordinate for every S plane.
  if (rows == 1 && columns % 64 == 0) {
    for (int cta = 0; cta < columns / 64; ++cta) {
      for (int thread = 0; thread < 32; ++thread) {
        int const base = (cta * 32 + thread) * 2;
        for (int lane = 0; lane < 2; ++lane) {
          for (int split = 0; split < splits; ++split) {
            std::int64_t const cute_offset =
                cutlass::gemm::kernel::detail::fp32_partial_cute_offset(
                    stride, rows, columns, splits, 0, base + lane, split);
            std::int64_t const fast_offset =
                std::int64_t(split) * rows * columns + base + lane;
            result.fast_reducer_bad += cute_offset != fast_offset;
          }
        }
      }
    }
  }
  return result;
}

}  // namespace

int main() {
  Result decode_s1 = verify(1, 1024, 1);
  Result decode_s2 = verify(1, 1024, 2);
  Result decode_s4 = verify(1, 1024, 4);
  Result tail_s4 = verify(9, 130, 4);
  std::array<Result, 4> results{{decode_s1, decode_s2, decode_s4, tail_s4}};

  std::int64_t visits = 0, expected = 0, holes = 0, duplicates = 0;
  std::int64_t value_bad = 0, cute_manual_bad = 0, fast_bad = 0;
  std::int64_t invalid = 0, compact_bad = 0;
  int const rows[] = {1, 1, 1, 9};
  int const columns[] = {1024, 1024, 1024, 130};
  int const splits[] = {1, 2, 4, 4};
  for (int i = 0; i < 4; ++i) {
    visits += results[std::size_t(i)].logical_visits;
    expected += std::int64_t(rows[i]) * columns[i] * splits[i];
    holes += results[std::size_t(i)].physical_holes;
    duplicates += results[std::size_t(i)].physical_duplicates;
    value_bad += results[std::size_t(i)].value_bad;
    cute_manual_bad += results[std::size_t(i)].cute_manual_bad;
    fast_bad += results[std::size_t(i)].fast_reducer_bad;
    invalid += results[std::size_t(i)].invalid_offsets;
    compact_bad += !results[std::size_t(i)].compact;
  }

  PartialStride physical_s1 = cutlass::gemm::kernel::detail::
      make_compact_fp32_partial_stride<PartialStride>(1, 1024);
  int64_t const physical_s1_pitch = int64_t(get<2>(physical_s1));

  bool const pass = visits == expected && holes == 0 && duplicates == 0 &&
      value_bad == 0 && cute_manual_bad == 0 && fast_bad == 0 &&
      invalid == 0 && compact_bad == 0 && physical_s1_pitch == 1024;
  std::printf(
      "L223_SPLITK_PARTIAL_ABI visits=%lld expected=%lld holes=%lld "
      "duplicates=%lld value_bad=%lld cute_manual_bad=%lld "
      "fast_reducer_bad=%lld invalid=%lld compact_bad=%lld "
      "physical_s1_pitch=%lld verdict=%s\n",
      static_cast<long long>(visits), static_cast<long long>(expected),
      static_cast<long long>(holes), static_cast<long long>(duplicates),
      static_cast<long long>(value_bad),
      static_cast<long long>(cute_manual_bad),
      static_cast<long long>(fast_bad), static_cast<long long>(invalid),
      static_cast<long long>(compact_bad),
      static_cast<long long>(physical_s1_pitch), pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
