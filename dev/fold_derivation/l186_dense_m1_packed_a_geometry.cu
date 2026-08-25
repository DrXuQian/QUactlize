// L186 -- host writer -> PPU0010-read geometry oracle for dense M==1 packed A.
//
// There are intentionally TWO authorities in this proof:
//   * the writer calls the exact production detail::aPackRunOffsetHalfs() helper;
//   * the reader independently replays PPU0010_TSM_LD_SWZL_M8 lane/vreg arithmetic and inverts it.
//
// The second model is verbatim from cute/arch/copy_ppu0010_aiu.hpp and was calibrated against hardware by
// l2/l3/l7/l10/l12/l13/l16/l17 before l84-l86 exported the packed-row construction.  Thus this is not a second
// copy of the writer formula and not a place/recover pair that can agree with its own mistake.  We fill simulated
// shared memory through the production writer map, read it through the independent PPU map, and require every
// logical (pipe,k) value exactly once for every production TileK in {64,128,256}.

#include <array>
#include <cstdio>
#include <vector>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_a_pack.hpp"

namespace {
using namespace cute;

constexpr int kLogicalM = 8;
constexpr int kPhysicalM = 16;
constexpr int kTileN = 128;
constexpr int kWarpM = 8;
constexpr int kWarpN = 32;
constexpr int kStages = 3;
constexpr int kRows = 1;
constexpr int kCubeW = 64;
constexpr int kSlices = 4;
constexpr int kPitch = cutlass::gemm::collective::detail::aPackPitchForRows(kRows);

using Atom = PPU0010_8x16x16_F32F16F16F32_TN;
using Mma = TiledMMA<MMA_Atom<Atom>,
    Layout<Shape<Int<kLogicalM / kWarpM>, Int<kTileN / kWarpN>, _1>>,
    Tile<Int<(kLogicalM / kWarpM) * 8>, Int<(kTileN / kWarpN) * 16>, _16>>;
using Fragment = decltype(make_fragment_like<float>(
    partition_fragment_C(Mma{}, Shape<Int<kLogicalM>, Int<kTileN>>{})));

// The exact source-coordinate route of
// fq_tc_q12_a64_tm8_tn64_tk256_wm8_wn16_s2_bc0_ap1.  The original L186
// reader model started from the already-decoded (cube,stage) pair and therefore
// did not prove that production CuTe partition_S delivered that pair.
constexpr int kExactTileK = 256;
constexpr int kExactStages = 2;
constexpr int kExactCubes = kExactTileK / kCubeW;
constexpr int kExactStagePitch =
    cutlass::gemm::collective::detail::aPackStagePitchHalfs(
        kPitch, kExactCubes, kPhysicalM * kCubeW);
using ExactMma = TiledMMA<MMA_Atom<Atom>,
    Layout<Shape<_1, _4, _1>>, Tile<_8, _64, _16>>;
using ExactSmemAtom = Layout<Shape<_8, _64>, Stride<_64, _1>>;
using ExactLogicalStage = decltype(tile_to_shape(
    ExactSmemAtom{}, make_shape(_8{}, Int<kExactTileK>{})));
using ExactPhysicalStage = decltype(tile_to_shape(
    ExactSmemAtom{}, make_shape(_16{}, Int<kExactTileK>{})));
using ExactSmemLayoutA = decltype(append(
    ExactLogicalStage{},
    make_layout(Int<kExactStages>{},
                Int<cute::cosize_v<ExactPhysicalStage>>{})));
using ExactSmemCopyOp = PPU0010_TSM_LD_SWZL_M8<
    cutlass::half_t, 16, 64, true, false, kExactCubes, kPitch,
    kExactStagePitch>;
using ExactSmemCopyAtom = Copy_Atom<ExactSmemCopyOp, cutlass::half_t>;

int verify_exact_cute_source_coordinates() {
  std::array<cutlass::half_t, 1152> storage{};
  auto s_a = make_tensor(make_smem_ptr(storage.data()), ExactSmemLayoutA{});
  auto tiled_copy = make_tiled_copy_A(ExactSmemCopyAtom{}, ExactMma{});
  int bad = 0;
  for (int warp = 0; warp < 4; ++warp) {
    auto t_cs_a = tiled_copy.get_thread_slice(warp * 32).partition_S(
        make_mix_tensor_like(s_a));
    for (int stage = 0; stage < kExactStages; ++stage) {
      for (int k_block = 0; k_block < int(size<2>(t_cs_a)); ++k_block) {
        auto source = t_cs_a(_, _, k_block, stage);
        auto coord = source.data().coord_;
        // copy_unpack forwards this four-tuple verbatim to
        // PPU0010_TSM_LD_SWZL_M8::copy(coord_w,coord_h,cube,stage).
        bad += int(get<0>(coord)) != (k_block % kSlices) * 16;
        bad += int(get<1>(coord)) != 0;
        bad += int(get<2>(coord)) != k_block / kSlices;
        bad += int(get<3>(coord)) != stage;
      }
    }
  }
  std::printf("[l186:cute] exact=s2/tk256 warps=4 stages=2 k_blocks=%d "
              "coord_bad=%d\n",
              int(size<2>(tiled_copy.get_thread_slice(0).partition_S(
                  make_mix_tensor_like(s_a)))), bad);
  return bad;
}

static_assert(size<0>(typename MMA_Traits<Atom>::Shape_MNK{}) == 8 &&
              size<1>(typename MMA_Traits<Atom>::Shape_MNK{}) == 16 &&
              size<2>(typename MMA_Traits<Atom>::Shape_MNK{}) == 16,
              "geometry oracle must remain the exact shipping m8 atom");
static_assert(kPitch == 64 && kPhysicalM == 16 && kRows == 1 && int(size(Mma{})) == 128,
              "geometry oracle must remain bound to the production Rows1/CTA128 authorities");

// Independent verbatim PPU0010_TSM_LD_SWZL_M8 address model.  Do not replace this with
// detail::aPackRunOffsetHalfs(): their agreement is one of the properties under test.
constexpr int ppu0010_tsm_ld_swzl_m8_word(
    int lane, int vreg, int slice, int cube_height, int* out_row) {
  int const slice_word_base = cube_height * 8 * slice;
  int const slice_start_vec = (((slice & 1) << 1) + ((slice & 2) >> 1)) * 2;
  int const lane_row_idx = lane / 4;  // coord_h == 0 for the first physical m8 read
  int const lane_col_idx = lane % 4;
  int const vreg_row_idx = (vreg / 2) * 8 + lane_row_idx;
  int const vreg_line_idx = vreg_row_idx / 4;
  int const vreg_vec_idx = (vreg_row_idx % 4) * 2 + (vreg % 2);
  int const vreg_vec_idx_swz1 = vreg_vec_idx ^ (vreg_line_idx % 2);
  int const vreg_vec_idx_swz2 = (vreg_vec_idx_swz1 + slice_start_vec) % 8;
  *out_row = vreg_row_idx;
  return slice_word_base + vreg_line_idx * 32 + vreg_vec_idx_swz2 * 4 + lane_col_idx;
}

#ifndef L186_BAD_DESTINATION_DELTA
#define L186_BAD_DESTINATION_DELTA 0
#endif
#ifndef L186_BAD_SLICE_SWAP
#define L186_BAD_SLICE_SWAP 0
#endif

struct Totals {
  long long cells = 0;
  int authority_bad = 0;
  int model_holes = 0;
  int model_duplicates = 0;
  int copy_holes = 0;
  int copy_duplicates = 0;
  int source_holes = 0;
  int source_duplicates = 0;
  int destination_duplicates = 0;
  int destination_oob = 0;
  int reader_holes = 0;
  int reader_duplicates = 0;
  int unread_writes = 0;
  int value_mismatches = 0;
  int output_holes = 0;
  int output_duplicates = 0;

  void add(Totals const& b) {
    cells += b.cells;
    authority_bad += b.authority_bad;
    model_holes += b.model_holes;
    model_duplicates += b.model_duplicates;
    copy_holes += b.copy_holes;
    copy_duplicates += b.copy_duplicates;
    source_holes += b.source_holes;
    source_duplicates += b.source_duplicates;
    destination_duplicates += b.destination_duplicates;
    destination_oob += b.destination_oob;
    reader_holes += b.reader_holes;
    reader_duplicates += b.reader_duplicates;
    unread_writes += b.unread_writes;
    value_mismatches += b.value_mismatches;
    output_holes += b.output_holes;
    output_duplicates += b.output_duplicates;
  }
};

struct PpuReadInverse {
  std::array<int, kCubeW> physical_half{};
  std::array<int, kCubeW> visits{};
};

PpuReadInverse make_ppu_read_inverse(int row_wanted) {
  PpuReadInverse inverse;
  inverse.physical_half.fill(-1);
  for (int lane = 0; lane < 32; ++lane) {
    for (int vreg = 0; vreg < 4; ++vreg) {
      for (int slice = 0; slice < kSlices; ++slice) {
        int row = -1;
        int const physical_word =
            ppu0010_tsm_ld_swzl_m8_word(lane, vreg, slice, kPhysicalM, &row);
        if (row != row_wanted) continue;
        int const logical_word = 8 * slice + 4 * (vreg % 2) + lane % 4;
        for (int half = 0; half < 2; ++half) {
          int const logical_k = 2 * logical_word + half;
          int const physical = 2 * physical_word + half;
          ++inverse.visits[std::size_t(logical_k)];
          if (inverse.physical_half[std::size_t(logical_k)] == -1)
            inverse.physical_half[std::size_t(logical_k)] = physical;
          else if (inverse.physical_half[std::size_t(logical_k)] != physical)
            inverse.physical_half[std::size_t(logical_k)] = -2;
        }
      }
    }
  }
  return inverse;
}

int authority_errors() {
  int bad = 0;
  for (int row_wanted = 0; row_wanted < kPhysicalM; ++row_wanted) {
    for (int slice = 0; slice < kSlices; ++slice) {
      std::array<int, 8> words{};
      int count = 0;
      for (int lane = 0; lane < 32; ++lane) {
        for (int vreg = 0; vreg < 4; ++vreg) {
          int row = -1;
          int const word =
              ppu0010_tsm_ld_swzl_m8_word(lane, vreg, slice, kPhysicalM, &row);
          if (row == row_wanted && count < int(words.size())) words[std::size_t(count++)] = word;
        }
      }
      if (count != 8) {
        ++bad;
        continue;
      }
      int min_word = words[0];
      int max_word = words[0];
      for (int word : words) {
        min_word = word < min_word ? word : min_word;
        max_word = word > max_word ? word : max_word;
      }
      std::array<int, 8> seen{};
      for (int word : words) {
        if (word < min_word || word >= min_word + 8) ++bad;
        else ++seen[std::size_t(word - min_word)];
      }
      for (int visits : seen) bad += visits != 1;
      bad += max_word - min_word != 7;
      bad += cutlass::gemm::collective::detail::aPackRunOffsetHalfs(
                 kPhysicalM, row_wanted, slice) != 2 * min_word;
    }
  }
  return bad;
}

// The CuTe m8 atom publishes v0/v1, but the hardware instruction physically
// reads x4.  This oracle deliberately counts the complete x4 footprint against
// the other pipeline stage's live-row writer.  The historical flat cube pitch
// must be RED; the production stage pitch must have no cross-stage address.
int cross_stage_hardware_collisions(int stage_pitch) {
  constexpr int kAddressLimit = 4096;
  std::array<int, kAddressLimit> writer[2]{};
  std::array<int, kAddressLimit> reader[2]{};
  for (int stage = 0; stage < 2; ++stage) {
    for (int cube = 0; cube < kExactCubes; ++cube) {
      int const base = stage * stage_pitch + cube * kPitch;
      for (int slice = 0; slice < kSlices; ++slice) {
        for (int half_run = 0; half_run < 2; ++half_run) {
          for (int e = 0; e < 8; ++e) {
            int const dst = base +
                cutlass::gemm::collective::detail::aPackRunOffsetHalfs(
                    kPhysicalM, 0, slice) + half_run * 8 + e;
            if (0 <= dst && dst < kAddressLimit) writer[stage][dst] = 1;
          }
        }
        for (int lane = 0; lane < 32; ++lane) {
          for (int vreg = 0; vreg < 4; ++vreg) {
            int row = -1;
            int const word = ppu0010_tsm_ld_swzl_m8_word(
                lane, vreg, slice, kPhysicalM, &row);
            (void)row;
            for (int half = 0; half < 2; ++half) {
              int const src = base + 2 * word + half;
              if (0 <= src && src < kAddressLimit) reader[stage][src] = 1;
            }
          }
        }
      }
    }
  }
  int collisions = 0;
  for (int i = 0; i < kAddressLimit; ++i) {
    collisions += reader[0][i] && writer[1][i];
    collisions += reader[1][i] && writer[0][i];
  }
  return collisions;
}

constexpr int marker(int pipe, int tile_k, int logical_k) {
  return 1 + pipe * tile_k + logical_k;
}

template <int TileK>
Totals verify_k() {
  static_assert(TileK == 64 || TileK == 128 || TileK == 256);
  constexpr int cubes = TileK / kCubeW;
  constexpr int stage_pitch =
      cutlass::gemm::collective::detail::aPackStagePitchHalfs(
          kPitch, cubes, kPhysicalM * kCubeW);
  constexpr int span_raw = stage_pitch * (kStages - 1) +
      kPitch * (cubes - 1) + kPhysicalM * kCubeW;
  constexpr int span = ((span_raw + 63) / 64) * 64;
  constexpr int copies_per_pipe = cubes * kRows * kSlices * 2;
  constexpr int threads = int(size(Mma{}));

  Totals z{};
  z.cells = 1LL * kStages * TileK;
  z.authority_bad = authority_errors();
  auto const inverse = make_ppu_read_inverse(0);
  for (int k = 0; k < kCubeW; ++k) {
    z.model_holes += inverse.visits[std::size_t(k)] == 0 ||
                     inverse.physical_half[std::size_t(k)] < 0;
    z.model_duplicates += inverse.visits[std::size_t(k)] > 1;
  }

  std::vector<int> shared(std::size_t(span), -1);
  std::vector<int> destination(std::size_t(span), 0);
  std::vector<int> reader_destination(std::size_t(span), 0);
  std::vector<int> source(std::size_t(kStages * TileK), 0);
  std::vector<int> copy_visits(std::size_t(kStages * copies_per_pipe), 0);

  // Exact production ownership: CTA128 threads stride the logical uint128 copy domain.  This is deliberately not a
  // direct nested cube/run loop; changing the CTA or copy-domain denominator must surface as a hole/duplicate.
  for (int pipe = 0; pipe < kStages; ++pipe) {
    for (int thread = 0; thread < threads; ++thread) {
      for (int logical_thread = thread; logical_thread < copies_per_pipe; logical_thread += threads) {
        ++copy_visits[std::size_t(pipe * copies_per_pipe + logical_thread)];
        int const cube = logical_thread / (kRows * kSlices * 2);
        int const row = (logical_thread % (kRows * kSlices * 2)) / (kSlices * 2);
        int const run = (logical_thread % (kSlices * 2)) / 2;
        int const half = logical_thread % 2;
        int const physical_half = half ^ ((row / 4) & 1);
#if L186_BAD_SLICE_SWAP
        int const writer_run = run == 1 ? 2 : (run == 2 ? 1 : run);
#else
        int const writer_run = run;
#endif
        for (int element = 0; element < 8; ++element) {
          int const k = cube * kCubeW + run * 16 + half * 8 + element;
          ++source[std::size_t(pipe * TileK + k)];
          int const dst = kPitch * cube + stage_pitch * pipe +
              cutlass::gemm::collective::detail::aPackRunOffsetHalfs(
                  kPhysicalM, row, writer_run) +
              L186_BAD_DESTINATION_DELTA * (pipe == 1) + physical_half * 8 + element;
          if (dst < 0 || dst >= span) {
            ++z.destination_oob;
          } else {
            ++destination[std::size_t(dst)];
            shared[std::size_t(dst)] = marker(pipe, TileK, k);
          }
        }
      }
    }
  }

  for (int visits : copy_visits) {
    z.copy_holes += visits == 0;
    z.copy_duplicates += visits > 1;
  }
  for (int visits : source) {
    z.source_holes += visits == 0;
    z.source_duplicates += visits > 1;
  }
  for (int visits : destination) z.destination_duplicates += visits > 1;

  // Independent PPU0010 reader: no production run-offset helper appears below this point.
  for (int pipe = 0; pipe < kStages; ++pipe) {
    for (int cube = 0; cube < cubes; ++cube) {
      for (int local_k = 0; local_k < kCubeW; ++local_k) {
        int const physical = inverse.physical_half[std::size_t(local_k)];
        if (physical < 0) {
          ++z.reader_holes;
          continue;
        }
        int const src = kPitch * cube + stage_pitch * pipe + physical;
        if (src < 0 || src >= span) {
          ++z.reader_holes;
          continue;
        }
        ++reader_destination[std::size_t(src)];
        int const logical_k = cube * kCubeW + local_k;
        z.value_mismatches += shared[std::size_t(src)] != marker(pipe, TileK, logical_k);
      }
    }
  }
  for (int i = 0; i < span; ++i) {
    z.reader_holes += destination[std::size_t(i)] > 0 && reader_destination[std::size_t(i)] == 0;
    z.reader_duplicates += reader_destination[std::size_t(i)] > 1;
    z.unread_writes += destination[std::size_t(i)] > 0 && reader_destination[std::size_t(i)] != 1;
  }
  return z;
}

Totals verify_output_ownership() {
  Totals z{};
  std::array<int, kLogicalM * kTileN> output{};
  auto identity = make_identity_tensor(Shape<Int<kLogicalM>, Int<kTileN>>{});
  for (int thread = 0; thread < int(size(Mma{})); ++thread) {
    auto part = Mma{}.get_thread_slice(thread).partition_C(identity);
    Fragment fragment;
    auto physical_to_fragment = right_inverse(fragment.layout());
    for (int slot = 0; slot < int(size(fragment)); ++slot) {
      auto coord = part(physical_to_fragment(slot));
      int const m = int(get<0>(coord));
      int const n = int(get<1>(coord));
      if (0 <= m && m < kLogicalM && 0 <= n && n < kTileN)
        ++output[std::size_t(m * kTileN + n)];
    }
  }
  for (int visits : output) {
    z.output_holes += visits == 0;
    z.output_duplicates += visits > 1;
  }
  return z;
}

bool clean(Totals const& z) {
  return z.authority_bad == 0 && z.model_holes == 0 && z.model_duplicates == 0 &&
         z.copy_holes == 0 && z.copy_duplicates == 0 &&
         z.source_holes == 0 && z.source_duplicates == 0 &&
         z.destination_duplicates == 0 && z.destination_oob == 0 &&
         z.reader_holes == 0 && z.reader_duplicates == 0 && z.unread_writes == 0 &&
         z.value_mismatches == 0 && z.output_holes == 0 && z.output_duplicates == 0;
}

}  // namespace

int main() {
  Totals z{};
  z.add(verify_k<64>());
  z.add(verify_k<128>());
  z.add(verify_k<256>());
  z.add(verify_output_ownership());
  int const cute_coord_bad = verify_exact_cute_source_coordinates();
  int const historical_cross = cross_stage_hardware_collisions(
      kPitch * kExactCubes);
  int const production_cross = cross_stage_hardware_collisions(
      kExactStagePitch);
  std::printf(
      "[l186:physical-footprint] logical=x2 physical=x4 "
      "historical_stage_pitch=%d historical_cross=%d "
      "production_stage_pitch=%d production_cross=%d\n",
      kPitch * kExactCubes, historical_cross,
      kExactStagePitch, production_cross);
  std::printf(
      "[l186:geometry] tk=64,128,256 cells=%lld authority_bad=%d model_holes=%d model_duplicates=%d "
      "copy_holes=%d copy_duplicates=%d source_holes=%d source_duplicates=%d "
      "destination_duplicates=%d destination_oob=%d reader_holes=%d reader_duplicates=%d "
      "unread_writes=%d value_mismatches=%d output_holes=%d output_duplicates=%d\n",
      z.cells, z.authority_bad, z.model_holes, z.model_duplicates,
      z.copy_holes, z.copy_duplicates, z.source_holes, z.source_duplicates,
      z.destination_duplicates, z.destination_oob, z.reader_holes, z.reader_duplicates,
      z.unread_writes, z.value_mismatches, z.output_holes, z.output_duplicates);
  bool const ok = clean(z) && cute_coord_bad == 0 &&
      historical_cross > 0 && production_cross == 0;
  std::printf(
      "[l186:geometry] %s: production writer -> independent hardware-calibrated PPU0010 reader is "
      "exact-once and value-correct\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
