// L224 -- exact A-register prepare/consume lifetime for the failing
// Q4_K/A64 TM8/TN64/TK256/WM8/WN16/Stages2 packed-A specialization.
//
// Keep this a host CuTe oracle. The full shipping collective forces PPU
// device bodies through nvcc and cannot be iterated on the host. L186 binds
// the shipping specialization to the exact Mma/SmemCopyAtom aliases repeated
// below; this file composes those aliases and exhaustively compares the
// physical register offsets written by prepare(next) with those consumed by
// the current MMA delivery.

#include <array>
#include <cstdio>
#include <vector>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_a_schedule.hpp"

namespace {
using namespace cute;

using Atom = PPU0010_8x16x16_F32F16F16F32_TN;
using MainPermK = Int<cutlass::MixGemmMmaPermK<4, 256, 1>::value>;
static_assert(MainPermK{} == _64{},
              "Q4/TK256/F1 shipping MMA permutation must remain K64");
using Mma = TiledMMA<MMA_Atom<Atom>,
    Layout<Shape<_1, _4, _1>>, Tile<_8, _64, MainPermK>>;
using SmemAtomA = Layout<Shape<_8, _64>, Stride<_64, _1>>;
using SmemStageA = decltype(tile_to_shape(
    SmemAtomA{}, make_shape(_8{}, _256{})));
using SmemLayoutA = decltype(append(
    SmemStageA{}, make_layout(_2{}, Int<4096>{})));
using SmemCopyOpA = PPU0010_TSM_LD_SWZL_M8<
    cutlass::half_t, 16, 64, true, false, 4, 64, 1216>;
using SmemCopyAtomA = Copy_Atom<SmemCopyOpA, cutlass::half_t>;

// Exact shipping B types.  Q4/A64 gives one 32-byte resident delivery
// (64 int4 values), repeated four times over TK256.  The fixed int8 shadow
// loader has a physical m16 shape even when the compute atom is m8.
using SmemAtomB = Layout<Shape<_8, _64>, Stride<_64, _1>>;
using SmemLayoutB = decltype(tile_to_shape(
    SmemAtomB{}, make_shape(_64{}, _256{}, _2{})));
using ShadowAtom = PPU0010_16x16x32_S32S8S8S32_TN;
using ShadowMma = TiledMMA<MMA_Atom<ShadowAtom>,
    Layout<Shape<_1, _4, _1>>, Tile<_16, _64, _32>>;
using SmemCopyOpB =
    PPU0010_TSM_LD_SWZL<int8_t, 64, 32, true, false, 4>;
using SmemCopyAtomB = Copy_Atom<SmemCopyOpB, int8_t>;

// Exact shipping global-to-shared B copy.  Q4/A64 has a 32-byte contiguous
// artifact delivery: one 64x64-int4 cube, repeated four times over TK256.
// The AIU atom itself has one logical issuing thread.  L224 also proves what
// happens when the shipping collective nevertheless passes CTA thread ids
// 0..31 to this one-thread tiled copy before copy_aiu's warp-0 guard.
using GmemCopyOpB = PPU0010_AIU_LOAD<
    C<64 * 32 * 8>, cutlass::int4b_t, false>;
using GmemCopyAtomB = Copy_Atom<GmemCopyOpB, cutlass::int4b_t>;
using GmemTiledCopyB = decltype(make_tiled_copy(
    GmemCopyAtomB{},
    Layout<Shape<_1, _1>, Stride<_1, _1>>{},
    Layout<Shape<_64, _64>>{}));

using SmemAtomScale = Layout<Shape<_8, _1>>;
using SmemLayoutScale = decltype(tile_to_shape(
    SmemAtomScale{}, make_shape(_64{}, _8{}, _2{})));
using SmemLayoutScaleRaw =
    Layout<Shape<_64, _16, _2>, Stride<_16, _1, _1024>>;

struct OutputOwner {
  int thread = -1;
  int fragment = -1;
};

template <class Tensor>
void mark_mode2(Tensor const& tensor, int k, std::array<int, 8192>& out) {
  for (int i = 0; i < int(size<0>(tensor)); ++i) {
    for (int j = 0; j < int(size<1>(tensor)); ++j) {
      int const offset = int(tensor.layout()(make_coord(i, j, k)));
      if (0 <= offset && offset < int(out.size())) {
        ++out[std::size_t(offset)];
      }
    }
  }
}

template <class Tensor>
void mark_flat(Tensor const& tensor, int base,
               std::array<int, 8192>& out) {
  for (int i = 0; i < int(size(tensor)); ++i) {
    int const offset = base + int(tensor.layout()(i));
    if (0 <= offset && offset < int(out.size())) {
      ++out[std::size_t(offset)];
    }
  }
}

int differences(std::array<int, 8192> const& a,
                std::array<int, 8192> const& b) {
  int result = 0;
  for (int i = 0; i < int(a.size()); ++i) {
    result += a[std::size_t(i)] != b[std::size_t(i)];
  }
  return result;
}

int overlap(std::array<int, 8192> const& a,
            std::array<int, 8192> const& b) {
  int result = 0;
  for (int i = 0; i < int(a.size()); ++i) {
    result += a[std::size_t(i)] != 0 && b[std::size_t(i)] != 0;
  }
  return result;
}

constexpr int q4_code(int n, int k) {
  return (((13 * n + 7 * k + 3) & 7) - 3) & 15;
}

constexpr int active_offset(int superblock) {
  return (37 * superblock + 11) & 255;
}

constexpr int active_sign(int superblock) {
  return (superblock & 1) ? -1 : 1;
}

// Independent replay of PPU0010_TSM_LD_SWZL<int8_t,64,32,true,false,4>.
// The CuTe source coordinate supplies (coord_h, (coord_w,cube), stage);
// this returns the physical b32 word inside the complete two-stage B buffer.
constexpr int ppu_b_word(int lane, int vreg, int coord_w, int coord_h,
                         int cube, int stage) {
  constexpr int CubeH = 64;
  constexpr int CubeWBytes = 32;
  constexpr int Cubes = 4;
  int const stage_cube_base = CubeH * CubeWBytes * (cube + stage * Cubes);
  int const slice = coord_w / 32;
  int const slice_word_base = (stage_cube_base / 4) + CubeH * 8 * slice;
  int const slice_start_vec =
      (((slice & 1) << 1) + ((slice & 2) >> 1)) * 2;
  int const lane_row = lane / 4 + coord_h;
  int const lane_col = lane % 4;
  int const vreg_row = (vreg / 2) * 8 + lane_row;
  int const line = vreg_row / 4;
  int const vec = (vreg_row % 4) * 2 + (vreg % 2);
  int const swizzled = ((vec ^ (line % 2)) + slice_start_vec) % 8;
  return slice_word_base + line * 32 + swizzled * 4 + lane_col;
}

template <class Tensor>
void show(char const* name, Tensor const& tensor) {
  std::printf("L224 %s size=%d cosize=%d layout=", name,
              int(size(tensor)), int(cosize(tensor.layout())));
  print(tensor.layout());
  std::printf("\n");
}

template <class Tensor>
void show_values(char const* name, Tensor const& tensor, int limit = 8) {
  std::printf("L224 %s values=", name);
  for (int i = 0; i < int(size(tensor)) && i < limit; ++i) {
    if (i != 0) std::printf(",");
    print(tensor(i));
  }
  std::printf("\n");
}
}  // namespace

int main() {
  Mma mma;
  auto thr = mma.get_thread_slice(0);
  auto s_a = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                         SmemLayoutA{});
  auto t_cr_a = thr.partition_fragment_A(s_a(_, _, Int<0>{}));
  auto a_copy = make_tiled_copy_A(SmemCopyAtomA{}, mma);
  auto a_view = a_copy.get_thread_slice(0).retile_D(t_cr_a);

  // The failing 32-output footprint spans two N warps.  Bind every warp's
  // collective source view, not only thread 0's register layout: A must be
  // broadcast-identical across the four N warps while B must select the
  // corresponding N band.  A logical proof for one warp cannot establish
  // either relation.
  auto a_identity = make_identity_tensor(shape(s_a));
  auto a_src_w0 = a_copy.get_thread_slice(0).partition_S(a_identity);
  auto a_src_w1 = a_copy.get_thread_slice(32).partition_S(a_identity);
  auto a_src_w2 = a_copy.get_thread_slice(64).partition_S(a_identity);
  auto a_src_w3 = a_copy.get_thread_slice(96).partition_S(a_identity);

  auto s_b = make_tensor(make_smem_ptr((cutlass::int4b_t*)nullptr),
                         SmemLayoutB{});
  auto s_b_s8 = recast<int8_t>(s_b);
  auto t_cr_b_mma = thr.partition_fragment_B(s_b(_, _, Int<0>{}));
  ShadowMma shadow_mma;
  auto t_cr_b_load = shadow_mma.get_thread_slice(0).partition_fragment_B(
      s_b_s8(_, _, Int<0>{}));
  auto b_copy = make_tiled_copy_B(SmemCopyAtomB{}, shadow_mma);
  auto b_view = b_copy.get_thread_slice(0).retile_D(t_cr_b_load);
  // Production partitions make_mix_tensor_like(sB_s8), not the original
  // int4 tensor.  Keep the coordinate oracle on that exact recast view.
  auto b_identity = make_identity_tensor(shape(s_b_s8));
  auto b_src_w0 = b_copy.get_thread_slice(0).partition_S(b_identity);
  auto b_src_w1 = b_copy.get_thread_slice(32).partition_S(b_identity);
  auto b_src_w2 = b_copy.get_thread_slice(64).partition_S(b_identity);
  auto b_src_w3 = b_copy.get_thread_slice(96).partition_S(b_identity);

  // Complete B physical footprint.  This is stronger than comparing stage
  // strides: take the production CuTe source coordinate for every warp,
  // copy block and stage, replay all four physical x4 register reads, and
  // count the addressed bytes. Every byte in each 8-KiB B stage must be read
  // exactly once, and the two stage byte sets must be disjoint.
  constexpr int BStageBytes = 64 * 32 * 4;
  constexpr int BTotalBytes = BStageBytes * 2;
  std::array<int, BTotalBytes> b_physical_reads{};
  std::array<int, BTotalBytes> b_stage_reads[2]{};
  int b_physical_oob = 0;
  int b_coord_bad = 0;
  for (int warp = 0; warp < 4; ++warp) {
    auto source = b_copy.get_thread_slice(warp * 32).partition_S(b_identity);
    for (int stage = 0; stage < 2; ++stage) {
      for (int block = 0; block < 4; ++block) {
        auto delivery = source(_, _, block, stage);
        auto coord = delivery(0);
        int const coord_h = int(get<0>(coord));
        int const coord_w = int(get<1, 0>(coord));
        int const cube = int(get<1, 1>(coord));
        int const coord_stage = int(get<2>(coord));
        b_coord_bad += coord_h != warp * 16;
        b_coord_bad += coord_w != 0;
        b_coord_bad += cube != block;
        b_coord_bad += coord_stage != stage;
        for (int lane = 0; lane < 32; ++lane) {
          for (int vreg = 0; vreg < 4; ++vreg) {
            int const word = ppu_b_word(
                lane, vreg, coord_w, coord_h, cube, coord_stage);
            for (int byte = 0; byte < 4; ++byte) {
              int const address = 4 * word + byte;
              if (0 <= address && address < BTotalBytes) {
                ++b_physical_reads[std::size_t(address)];
                ++b_stage_reads[stage][std::size_t(address)];
              } else
                ++b_physical_oob;
            }
          }
        }
      }
    }
  }
  int b_physical_holes = 0;
  int b_physical_duplicates = 0;
  int b_cross_stage_overlap = 0;
  for (int address = 0; address < BTotalBytes; ++address) {
    b_physical_holes += b_physical_reads[std::size_t(address)] == 0;
    b_physical_duplicates += b_physical_reads[std::size_t(address)] != 1;
  }
  for (int address = 0; address < BTotalBytes; ++address) {
    b_cross_stage_overlap +=
        b_stage_reads[0][std::size_t(address)] != 0 &&
        b_stage_reads[1][std::size_t(address)] != 0;
  }
  // RED control: replacing the stage pitch by one half-stage makes the
  // physical stage sets intersect. Keep it in the same enumerator domain.
  int b_bad_pitch_overlap = 0;
  for (int stage0_address = 0; stage0_address < BStageBytes;
       ++stage0_address) {
    int const bad_stage1_address = stage0_address + BStageBytes / 2;
    b_bad_pitch_overlap +=
        b_stage_reads[0][std::size_t(stage0_address)] != 0 &&
        bad_stage1_address < BStageBytes &&
        b_stage_reads[0][std::size_t(bad_stage1_address)] != 0;
  }
  bool const b_shared_physical_exact = b_coord_bad == 0 &&
      b_physical_oob == 0 && b_physical_holes == 0 &&
      b_physical_duplicates == 0 && b_cross_stage_overlap == 0 &&
      b_bad_pitch_overlap > 0;
  std::printf(
      "L224 B-shared-physical bytes=%d coord_bad=%d oob=%d holes=%d "
      "nononce=%d cross_stage_overlap=%d negative_half_pitch_overlap=%d "
      "verdict=%s\n",
      BTotalBytes, b_coord_bad, b_physical_oob, b_physical_holes,
      b_physical_duplicates, b_cross_stage_overlap, b_bad_pitch_overlap,
      b_shared_physical_exact ? "STRICTLY_DISJOINT" : "BAD");

  // Packed scale/zero publication has three different SharedStorage members:
  // raw native bytes, decoded scale and decoded zero. Within each member the
  // CuTe stage mode must be bijective and disjoint. Different members are
  // separate C++ objects, so there is no cross-member alias to infer.
  std::array<int, int(cosize(SmemLayoutScale{}))> scale_visits{};
  std::array<int, int(cosize(SmemLayoutScaleRaw{}))> raw_visits{};
  int scale_oob = 0;
  int raw_oob = 0;
  for (int stage = 0; stage < 2; ++stage) {
    for (int group = 0; group < 8; ++group) {
      for (int n = 0; n < 64; ++n) {
        int const offset = int(SmemLayoutScale{}(make_coord(n, group, stage)));
        if (0 <= offset && offset < int(scale_visits.size()))
          ++scale_visits[std::size_t(offset)];
        else
          ++scale_oob;
      }
    }
    for (int byte = 0; byte < 16; ++byte) {
      for (int n = 0; n < 64; ++n) {
        int const offset = int(SmemLayoutScaleRaw{}(
            make_coord(n, byte, stage)));
        if (0 <= offset && offset < int(raw_visits.size()))
          ++raw_visits[std::size_t(offset)];
        else
          ++raw_oob;
      }
    }
  }
  int scale_nononce = 0;
  int raw_nononce = 0;
  for (int visits : scale_visits) scale_nononce += visits != 1;
  for (int visits : raw_visits) raw_nononce += visits != 1;
  constexpr int ScaleStageElements = 64 * 8;
  constexpr int RawStageBytes = 64 * 16;
  bool const metadata_shared_exact = scale_oob == 0 && raw_oob == 0 &&
      scale_nononce == 0 && raw_nononce == 0 &&
      stride<2>(SmemLayoutScale{}) == Int<ScaleStageElements>{} &&
      stride<2>(SmemLayoutScaleRaw{}) == Int<RawStageBytes>{};
  std::printf(
      "L224 metadata-shared scale_elements=%d raw_bytes=%d "
      "scale_oob=%d scale_nononce=%d raw_oob=%d raw_nononce=%d "
      "scale_stage_pitch=%d raw_stage_pitch=%d verdict=%s\n",
      int(scale_visits.size()), int(raw_visits.size()), scale_oob,
      scale_nononce, raw_oob, raw_nononce,
      int(stride<2>(SmemLayoutScale{})),
      int(stride<2>(SmemLayoutScaleRaw{})),
      metadata_shared_exact ? "STRICTLY_DISJOINT" : "BAD");

  // Bind the global AIU issuer map, independently of the shared->register
  // consumer map above. Copy_Traits<GmemCopyOpB>::ThrID is Layout<_1> because
  // one AIU issue is one opaque logical copy.  The shipping helper currently
  // reaches it from every physical lane of warp 0.  That is numerically
  // idempotent only if every lane resolves the exact same source/destination
  // coordinate and descriptor. CuTe proves the coordinate half here; AiuDesc
  // is initialized from CTA-invariant problem fields and is audited at the
  // source seam. Do not infer a warp-synchronous issuer contract from
  // Layout<_1>: one physical thread is sufficient for this AIU operation.
  GmemTiledCopyB gmem_b_copy;
  auto g_b_identity = make_identity_tensor(Shape<_64, _256, _1>{});
  auto s_b_identity = make_identity_tensor(Shape<_64, _256, _2>{});
  auto g_b_lane0 = gmem_b_copy.get_thread_slice(0).partition_S(g_b_identity);
  auto s_b_lane0 = gmem_b_copy.get_thread_slice(0).partition_D(s_b_identity);
  int gmem_b_lane_mismatch = 0;
  int gmem_b_lane_oob = 0;
  for (int lane = 0; lane < 32; ++lane) {
    auto lane_src = gmem_b_copy.get_thread_slice(lane).partition_S(g_b_identity);
    auto lane_dst = gmem_b_copy.get_thread_slice(lane).partition_D(s_b_identity);
    for (int i = 0; i < int(size(lane_src)); ++i) {
      int const src = int(crd2idx(lane_src(i), shape(g_b_identity)));
      int const ref = int(crd2idx(g_b_lane0(i), shape(g_b_identity)));
      gmem_b_lane_mismatch += lane != 0 && src != ref;
      gmem_b_lane_oob += src < 0 || src >= int(size(g_b_identity));
    }
    for (int i = 0; i < int(size(lane_dst)); ++i) {
      int const dst = int(crd2idx(lane_dst(i), shape(s_b_identity)));
      int const ref = int(crd2idx(s_b_lane0(i), shape(s_b_identity)));
      gmem_b_lane_mismatch += lane != 0 && dst != ref;
      gmem_b_lane_oob += dst < 0 || dst >= int(size(s_b_identity));
    }
    if (lane < 4 || lane == 31) {
      std::printf("L224 B-AIU-issuer lane=%d src0=", lane);
      print(lane_src(0));
      std::printf(" dst0=");
      print(lane_dst(0));
      std::printf("\n");
    }
  }
  bool const gmem_b_warp_issuer_exact =
      size(typename GmemCopyAtomB::ThrID{}) == _1{} &&
      size<0>(typename GmemTiledCopyB::TiledLayout_TV{}) == _1{} &&
      gmem_b_lane_mismatch == 0 && gmem_b_lane_oob == 0;
  std::printf(
      "L224 B-AIU-issuer atom_threads=%d tiled_threads=%d "
      "nonowner_coordinate_mismatches=%d coordinate_oob=%d verdict=%s\n",
      int(size(typename GmemCopyAtomB::ThrID{})),
      int(size<0>(typename GmemTiledCopyB::TiledLayout_TV{})),
      gmem_b_lane_mismatch, gmem_b_lane_oob,
      gmem_b_warp_issuer_exact ? "IDENTICAL_COORDINATES" :
                                 "MAP_CHANGED");

  // CuTe's atom has one logical issuer.  Model the helper's physical guard
  // separately: the shipping baseline admits all 32 lanes of warp 0, while
  // PPU_AIU_SINGLE_LOGICAL_ISSUER admits only CTA thread 0.  The planted
  // two-lane guard is the exact negative -- identical coordinates do not make
  // two physical opaque-copy issues satisfy a one-thread atom contract.
  constexpr int BaselinePhysicalIssuers = 32;
  constexpr int CandidatePhysicalIssuers = 1;
  constexpr int PlantedTwoPhysicalIssuers = 2;
  int const logical_aiu_issuers =
      int(size(typename GmemCopyAtomB::ThrID{}));
  bool const gmem_b_single_issuer_contract_exact =
      gmem_b_warp_issuer_exact && logical_aiu_issuers == 1 &&
      BaselinePhysicalIssuers != logical_aiu_issuers &&
      CandidatePhysicalIssuers == logical_aiu_issuers &&
      PlantedTwoPhysicalIssuers != logical_aiu_issuers;
  std::printf(
      "L224 B-AIU-issuer-cardinality logical=%d baseline_physical=%d "
      "candidate_physical=%d planted_two_physical=%d negative_two=%s "
      "verdict=%s\n",
      logical_aiu_issuers, BaselinePhysicalIssuers,
      CandidatePhysicalIssuers, PlantedTwoPhysicalIssuers,
      PlantedTwoPhysicalIssuers == logical_aiu_issuers ? "FALSE_GREEN" :
                                                        "RED",
      gmem_b_single_issuer_contract_exact ? "ONE_PHYSICAL_ISSUER" :
                                            "ISSUER_CONTRACT_BAD");

  constexpr int MmaAtoms = decltype(size<2>(t_cr_a))::value;
  constexpr int ABlocks = decltype(size<2>(a_view))::value;
  constexpr int BAtoms = decltype(size<2>(t_cr_b_mma))::value;
  constexpr int BBlocks = decltype(size<2>(b_view))::value;
  constexpr int BAtomsPerCopy = BAtoms / BBlocks;
  using Schedule = cutlass::gemm::collective::detail::MixedARegisterSchedule<
      MmaAtoms, ABlocks, BBlocks>;
  static_assert(MmaAtoms == 16, "shipping A MMA atom count changed");
  static_assert(ABlocks == 4, "shipping A copy block count changed");
  static_assert(BAtoms == 16, "shipping B MMA atom count changed");
  static_assert(BBlocks == 4, "shipping B copy block count changed");
  static_assert(BAtomsPerCopy == 4,
                "shipping B atoms-per-copy count changed");
  static_assert(Schedule::AAtomsPerCopy == 4 &&
                Schedule::BAtomsPerCopy == 4,
                "L224 must remain the exact M8/TK256 register schedule");

  // Register offsets are lane-local.  Proving thread 0 alone is therefore
  // insufficient: the proof domain is (physical CTA thread, local register
  // offset). Reconstruct the exact production slices for all 128 threads.
  // Production deliberately passes warp*32 to each warp-collective TSM copy,
  // while the destination fragments come from the physical thread id.
  int all_thread_a_map_bad = 0;
  int all_thread_a_prepare_consume_overlap = 0;
  int all_thread_b_load_bad = 0;
  int all_thread_b_convert_bad = 0;
  int all_thread_b_prepare_consume_overlap = 0;
  int planted_same_block_overlap = 0;
  for (int thread = 0; thread < int(size(Mma{})); ++thread) {
    int const warp_collective_thread = (thread / 32) * 32;
    auto thread_mma = mma.get_thread_slice(thread);
    auto thread_a = thread_mma.partition_fragment_A(s_a(_, _, Int<0>{}));
    auto thread_a_view = a_copy.get_thread_slice(warp_collective_thread)
                             .retile_D(thread_a);

    for (int a = 0; a < ABlocks; ++a) {
      std::array<int, 8192> prepared{};
      std::array<int, 8192> consumed{};
      mark_mode2(thread_a_view, a, prepared);
      for (int atom = a * Schedule::AAtomsPerCopy;
           atom < (a + 1) * Schedule::AAtomsPerCopy; ++atom) {
        mark_mode2(thread_a, atom, consumed);
      }
      all_thread_a_map_bad += differences(prepared, consumed) != 0;
    }
    for (int consumed_b = 0; consumed_b < BBlocks; ++consumed_b) {
      int const prepare_b = (consumed_b + 1) % BBlocks;
      std::array<int, 8192> prepared_next{};
      std::array<int, 8192> consumed_now{};
      std::array<int, 8192> planted_same{};
      for (int a = 0; a < ABlocks; ++a) {
        int const first_b =
            (a * Schedule::AAtomsPerCopy) / Schedule::BAtomsPerCopy;
        if (first_b == prepare_b)
          mark_mode2(thread_a_view, a, prepared_next);
        if (first_b == consumed_b)
          mark_mode2(thread_a_view, a, planted_same);
      }
      for (int atom = consumed_b * Schedule::BAtomsPerCopy;
           atom < (consumed_b + 1) * Schedule::BAtomsPerCopy; ++atom) {
        mark_mode2(thread_a, atom, consumed_now);
      }
      all_thread_a_prepare_consume_overlap +=
          overlap(prepared_next, consumed_now);
      // Built-in RED control through the same set comparison: preparing the
      // current block instead of the next block must collide for every lane.
      planted_same_block_overlap += overlap(planted_same, consumed_now);
    }

    auto thread_b_mma = thread_mma.partition_fragment_B(
        s_b(_, _, Int<0>{}));
    auto thread_b_load = shadow_mma.get_thread_slice(thread)
        .partition_fragment_B(s_b_s8(_, _, Int<0>{}));
    auto thread_b_view = b_copy.get_thread_slice(warp_collective_thread)
                             .retile_D(thread_b_load);
    cute::for_each(cute::make_int_sequence<BBlocks>{}, [&] (auto b_) {
      constexpr int B = decltype(b_)::value;
      constexpr int NextB = (B + 1) % BBlocks;
      std::array<int, 8192> load{};
      std::array<int, 8192> copy_view{};
      mark_mode2(thread_b_load, B, load);
      mark_mode2(thread_b_view, B, copy_view);
      all_thread_b_load_bad += differences(load, copy_view) != 0;

      auto cvt_in = recast<cutlass::int4b_t>(
          thread_b_load(_, _, Int<B>{}));
      constexpr int FirstAtom = B * BAtomsPerCopy;
      int const base = int(thread_b_mma.layout()(
          make_coord(0, 0, Int<FirstAtom>{})));
      std::array<int, 8192> converted{};
      std::array<int, 8192> expected{};
      mark_flat(cvt_in, base, converted);
      for (int atom = FirstAtom; atom < FirstAtom + BAtomsPerCopy; ++atom)
        mark_mode2(thread_b_mma, atom, expected);
      all_thread_b_convert_bad += differences(converted, expected) != 0;

      auto next_in = recast<cutlass::int4b_t>(
          thread_b_load(_, _, Int<NextB>{}));
      constexpr int NextFirstAtom = NextB * BAtomsPerCopy;
      int const next_base = int(thread_b_mma.layout()(
          make_coord(0, 0, Int<NextFirstAtom>{})));
      std::array<int, 8192> prepared_next{};
      mark_flat(next_in, next_base, prepared_next);
      all_thread_b_prepare_consume_overlap +=
          overlap(prepared_next, expected);
    });
  }
  bool const all_thread_register_exact =
      all_thread_a_map_bad == 0 &&
      all_thread_a_prepare_consume_overlap == 0 &&
      all_thread_b_load_bad == 0 && all_thread_b_convert_bad == 0 &&
      all_thread_b_prepare_consume_overlap == 0 &&
      planted_same_block_overlap > 0;
  std::printf(
      "L224 all-thread-register domain=128xlocal-register-space "
      "A_map_bad=%d A_prepare_consume_overlap=%d B_load_bad=%d "
      "B_convert_bad=%d B_prepare_consume_overlap=%d "
      "negative_same_block_overlap=%d verdict=%s\n",
      all_thread_a_map_bad, all_thread_a_prepare_consume_overlap,
      all_thread_b_load_bad, all_thread_b_convert_bad,
      all_thread_b_prepare_consume_overlap, planted_same_block_overlap,
      all_thread_register_exact ? "STRICTLY_DISJOINT" : "BAD");

  show("A-fragment", t_cr_a);
  show("A-copy-view", a_view);
  show("B-MMA-fragment", t_cr_b_mma);
  show("B-load-fragment", t_cr_b_load);
  show("B-copy-view", b_view);
  show("A-source-w0", a_src_w0);
  show("A-source-w1", a_src_w1);
  show("A-source-w2", a_src_w2);
  show("A-source-w3", a_src_w3);
  show("B-source-w0", b_src_w0);
  show("B-source-w1", b_src_w1);
  show("B-source-w2", b_src_w2);
  show("B-source-w3", b_src_w3);
  show_values("A-source-w0", a_src_w0);
  show_values("A-source-w1", a_src_w1);
  show_values("A-source-w2", a_src_w2);
  show_values("A-source-w3", a_src_w3);
  show_values("B-source-w0", b_src_w0);
  show_values("B-source-w1", b_src_w1);
  show_values("B-source-w2", b_src_w2);
  show_values("B-source-w3", b_src_w3);

  // Bind the observed aligned 32-output footprint to the production
  // partition_C map. At M=1 each warp owns 16 live columns, so one bad
  // 32-column band is exactly two adjacent N warps; it is not evidence for a
  // single thread, register, or producer warp by itself.
  auto c_identity = make_identity_tensor(Shape<_8, _64>{});
  std::array<OutputOwner, 64> output_owners{};
  std::array<int, 4> live_per_warp{};
  int output_duplicates = 0;
  for (int thread = 0; thread < int(size(Mma{})); ++thread) {
    auto coordinates = mma.get_thread_slice(thread).partition_C(c_identity);
    for (int fragment = 0; fragment < int(size(coordinates)); ++fragment) {
      auto mn = coordinates(fragment);
      int const m = int(get<0>(mn));
      int const n = int(get<1>(mn));
      if (m != 0) continue;
      if (n < 0 || n >= int(output_owners.size()) ||
          output_owners[std::size_t(n)].thread >= 0) {
        ++output_duplicates;
        continue;
      }
      output_owners[std::size_t(n)] = OutputOwner{thread, fragment};
      ++live_per_warp[std::size_t(thread / 32)];
    }
  }
  int output_holes = 0;
  int output_band_bad = 0;
  for (int n = 0; n < int(output_owners.size()); ++n) {
    output_holes += output_owners[std::size_t(n)].thread < 0;
    output_band_bad +=
        output_owners[std::size_t(n)].thread / 32 != n / 16;
  }
  bool const output_ownership_exact = output_holes == 0 &&
      output_duplicates == 0 && output_band_bad == 0 &&
      live_per_warp == std::array<int, 4>{{16, 16, 16, 16}};
  std::printf(
      "L224 output-ownership live_per_warp=%d,%d,%d,%d "
      "bands=N0-15:W0,N16-31:W1,N32-47:W2,N48-63:W3 "
      "aligned32=two-adjacent-N-warps holes=%d duplicates=%d band_bad=%d "
      "verdict=%s\n",
      live_per_warp[0], live_per_warp[1], live_per_warp[2],
      live_per_warp[3], output_holes, output_duplicates, output_band_bad,
      output_ownership_exact ? "EXACT" : "BAD");

  int identity_bad = 0;
  for (int a = 0; a < ABlocks; ++a) {
    std::array<int, 8192> prepared{};
    std::array<int, 8192> consumed{};
    mark_mode2(a_view, a, prepared);
    for (int atom = a * Schedule::AAtomsPerCopy;
         atom < (a + 1) * Schedule::AAtomsPerCopy; ++atom) {
      mark_mode2(t_cr_a, atom, consumed);
    }
    int const diff = differences(prepared, consumed);
    identity_bad += diff != 0;
    std::printf("L224 block-map A_copy=%d MMA_atoms=[%d,%d) diff=%d\n",
                a, a * Schedule::AAtomsPerCopy,
                (a + 1) * Schedule::AAtomsPerCopy, diff);
  }

  // Reconstruct transform_B_kblock's actual destination expression.  The
  // converter starts at the first main-B atom for one delivery and overlays
  // the recast load-fragment layout on that pointer.  This must cover exactly
  // four main-B atoms, with no overlap with the delivery currently consumed.
  int b_load_identity_bad = 0;
  int b_convert_identity_bad = 0;
  int b_convert_duplicates = 0;
  int b_prepare_consume_overlap = 0;
  std::array<int, 8192> all_converted{};
  cute::for_each(cute::make_int_sequence<BBlocks>{}, [&] (auto b_) {
    constexpr int B = decltype(b_)::value;
    constexpr int NextB = (B + 1) % BBlocks;
    std::array<int, 8192> load{};
    std::array<int, 8192> copy_view{};
    mark_mode2(t_cr_b_load, B, load);
    mark_mode2(b_view, B, copy_view);
    int const load_diff = differences(load, copy_view);
    b_load_identity_bad += load_diff != 0;

    auto cvt_in = recast<cutlass::int4b_t>(
        t_cr_b_load(_, _, Int<B>{}));
    constexpr int FirstAtom = B * BAtomsPerCopy;
    int const base = int(t_cr_b_mma.layout()(
        make_coord(0, 0, Int<FirstAtom>{})));
    std::array<int, 8192> converted{};
    mark_flat(cvt_in, base, converted);
    std::array<int, 8192> expected{};
    for (int atom = FirstAtom; atom < FirstAtom + BAtomsPerCopy; ++atom) {
      mark_mode2(t_cr_b_mma, atom, expected);
    }
    int const convert_diff = differences(converted, expected);
    b_convert_identity_bad += convert_diff != 0;
    for (int i = 0; i < int(converted.size()); ++i) {
      b_convert_duplicates += converted[std::size_t(i)] > 1;
      all_converted[std::size_t(i)] += converted[std::size_t(i)];
    }

    auto next_in = recast<cutlass::int4b_t>(
        t_cr_b_load(_, _, Int<NextB>{}));
    constexpr int NextFirstAtom = NextB * BAtomsPerCopy;
    int const next_base = int(t_cr_b_mma.layout()(
        make_coord(0, 0, Int<NextFirstAtom>{})));
    std::array<int, 8192> prepared_next{};
    mark_flat(next_in, next_base, prepared_next);
    int const current_overlap = overlap(prepared_next, expected);
    b_prepare_consume_overlap += current_overlap;
    std::printf(
        "L224 B-block=%d load_view_diff=%d convert_atoms=[%d,%d) "
        "convert_diff=%d prepare_next=%d current_overlap=%d\n",
        B, load_diff, FirstAtom, FirstAtom + BAtomsPerCopy, convert_diff,
        NextB, current_overlap);
  });
  int b_convert_holes = 0;
  int b_convert_global_duplicates = 0;
  for (int i = 0; i < int(size(t_cr_b_mma)); ++i) {
    int const physical = int(t_cr_b_mma.layout()(i));
    b_convert_holes += all_converted[std::size_t(physical)] == 0;
    b_convert_global_duplicates +=
        all_converted[std::size_t(physical)] != 1;
  }
  bool const b_register_exact = b_load_identity_bad == 0 &&
      b_convert_identity_bad == 0 && b_convert_duplicates == 0 &&
      b_convert_holes == 0 && b_convert_global_duplicates == 0 &&
      b_prepare_consume_overlap == 0;
  std::printf(
      "L224 B-register load_identity_bad=%d convert_identity_bad=%d "
      "local_duplicates=%d holes=%d global_nononce=%d "
      "prepare_consume_overlap=%d verdict=%s\n",
      b_load_identity_bad, b_convert_identity_bad, b_convert_duplicates,
      b_convert_holes, b_convert_global_duplicates,
      b_prepare_consume_overlap, b_register_exact ? "EXACT" : "BAD");

  int max_overlap = 0;
  int total_overlap = 0;
  for (int consumed_b = 0; consumed_b < BBlocks; ++consumed_b) {
    int const prepare_b = (consumed_b + 1) % BBlocks;
    std::array<int, 8192> prepared{};
    std::array<int, 8192> consumed{};
    for (int a = 0; a < ABlocks; ++a) {
      int const first_b =
          (a * Schedule::AAtomsPerCopy) / Schedule::BAtomsPerCopy;
      if (first_b == prepare_b) mark_mode2(a_view, a, prepared);
    }
    for (int atom = consumed_b * Schedule::BAtomsPerCopy;
         atom < (consumed_b + 1) * Schedule::BAtomsPerCopy; ++atom) {
      mark_mode2(t_cr_a, atom, consumed);
    }
    int const current = overlap(prepared, consumed);
    max_overlap = current > max_overlap ? current : max_overlap;
    total_overlap += current;
    std::printf("L224 transition consume_b=%d prepare_b=%d overlap=%d\n",
                consumed_b, prepare_b, current);
  }
  std::printf(
      "L224 schedule mma_atoms=%d A_blocks=%d B_blocks=%d "
      "A_atoms_per_copy=%d B_atoms_per_copy=%d max_overlap=%d total=%d\n",
      MmaAtoms, ABlocks, BBlocks, Schedule::AAtomsPerCopy,
      Schedule::BAtomsPerCopy, max_overlap, total_overlap);

  // Frozen ca01dc6 signature: S4 plane 2 spans superblocks [10,15) and one
  // captured 32-output band contained 6.0 instead of 1.0. Replacing sb13's
  // complete A tile with its predecessor is one exact explanation for that
  // value pair. It is not a classifier for the whole incident family: later
  // captures also contained deltas -4 and -6. Enumerate every complete A-tile
  // substitution below so this oracle cannot silently promote one compatible
  // signature into the root cause.
  constexpr int n = 32;
  int expected = 0;
  int stale13 = 0;
  int stale_matches = 0;
  for (int sb = 10; sb < 15; ++sb) {
    int const current = active_sign(sb) *
        q4_code(n, sb * 256 + active_offset(sb));
    int const previous = active_sign(sb - 1) *
        q4_code(n, sb * 256 + active_offset(sb - 1));
    expected += current;
    std::printf(
        "L224 plane2 n=%d sb=%d current=%d previous_tile=%d\n",
        n, sb, current, previous);
  }
  for (int replaced = 10; replaced < 15; ++replaced) {
    int candidate = expected;
    candidate -= active_sign(replaced) *
        q4_code(n, replaced * 256 + active_offset(replaced));
    candidate += active_sign(replaced - 1) *
        q4_code(n, replaced * 256 + active_offset(replaced - 1));
    stale_matches += candidate == 6;
    if (replaced == 13) stale13 = candidate;
    std::printf("L224 stale-replace sb=%d plane2=%d\n", replaced,
                candidate);
  }
  std::printf(
      "L224 incident expected=%d observed=6 stale_sb13=%d "
      "stale_matches=%d verdict=%s\n",
      expected, stale13, stale_matches,
      expected == 1 && stale13 == 6 && stale_matches == 1
          ? "ONE_SIGNATURE_ADMITS_PREVIOUS_A_TILE"
          : "INCIDENT_NOT_CLASSIFIED");

  int adjacent_minus5 = 0;
  int adjacent_plus5 = 0;
  int any_minus4 = 0;
  int any_minus6 = 0;
  for (int col = 0; col < 8; ++col) {
    for (int dst = 1; dst < 20; ++dst) {
      int const current = active_sign(dst) *
          q4_code(col, dst * 256 + active_offset(dst));
      int const previous = active_sign(dst - 1) *
          q4_code(col, dst * 256 + active_offset(dst - 1));
      adjacent_minus5 += previous - current == -5;
      adjacent_plus5 += previous - current == 5;
      for (int src = 0; src < 20; ++src) {
        if (src == dst) continue;
        int const replacement = active_sign(src) *
            q4_code(col, dst * 256 + active_offset(src));
        any_minus4 += replacement - current == -4;
        any_minus6 += replacement - current == -6;
      }
    }
  }
  bool const family_not_stale_a =
      adjacent_minus5 > 0 && adjacent_plus5 > 0 && any_minus4 > 0 &&
      any_minus6 == 0;
  std::printf(
      "L224 complete-A-substitution adjacent_delta_minus5=%d "
      "adjacent_delta_plus5=%d any_source_delta_minus4=%d "
      "any_source_delta_minus6=%d verdict=%s\n",
      adjacent_minus5, adjacent_plus5, any_minus4, any_minus6,
      family_not_stale_a ? "STALE_A_NOT_COMPLETE_FAMILY_ROOT" :
                           "FIXTURE_CLASSIFIER_CHANGED");
  std::printf("L224 identity_bad=%d verdict=%s\n", identity_bad,
              max_overlap == 0 && identity_bad == 0 && b_register_exact &&
                      all_thread_register_exact &&
                      b_shared_physical_exact && metadata_shared_exact &&
                      gmem_b_single_issuer_contract_exact &&
                      expected == 1 &&
                      stale13 == 6 && stale_matches == 1 &&
                      family_not_stale_a && output_ownership_exact
                  ? "REGISTER_LIFETIME_EXACT"
                  : "PREPARE_OR_BLOCK_MAP_BAD");
  return max_overlap == 0 && identity_bad == 0 && b_register_exact &&
                 all_thread_register_exact &&
                 b_shared_physical_exact && metadata_shared_exact &&
                 gmem_b_single_issuer_contract_exact &&
                 expected == 1 &&
                 stale13 == 6 && stale_matches == 1 && family_not_stale_a &&
                 output_ownership_exact
      ? 0
      : 1;
}
