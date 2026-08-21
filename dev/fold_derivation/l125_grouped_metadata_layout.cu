// L125 -- THE #112/G5 ZERO-PLANE ADDRESS CHAIN IS HOST-DECIDABLE.
//
// cp.async is a byte copy: it neither computes nor changes an address.  This
// oracle therefore exhausts all 256 experts through the production metadata
// tile helper, the production partition_S/partition_D maps, and an explicit
// host scatter/recover.  Unique raw 16-bit tags and an independent int64
// compact-layout formula are the anchor; agreement between two CuTe objects
// alone is not accepted as evidence.
//
// The exact CollectiveOp::load_init cannot itself be called from a host TU:
// without __HGGCCC__ the transitive PPU/CUTLASS bodies expose device
// intrinsics (the first failure is ppu_mma_multistage.hpp's __syncthreads),
// while with __HGGCCC__ it is device-only.  L125 therefore has two arms:
//   * L125_TYPE_ONLY binds the lightweight objects to the exact shipping G5
//     types under the device front-end;
//   * the runtime arm calls the same CUTE_HOST_DEVICE tile helper used by
//     production load_init and exhausts its pure layout algebra on the host.

#if defined(L125_TYPE_ONLY)

#include <cstdio>
#include <type_traits>

#include "m8n16_g5_contract.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"

namespace md = cutlass::gemm::collective::detail;
using Shipping = m8n16_g5_contract::M8;
using Mainloop = typename Shipping::Mainloop;
using RuntimeMetadata = md::MixedMetadataPolicy<
    cutlass::half_t, m8n16_g5_layout_spec::kScaleTileK, true,
    md::FlatMetadataAddress>;
using RuntimePlan = typename RuntimeMetadata::template ScaleCopy<
    m8n16_g5_layout_spec::kN, m8n16_g5_layout_spec::kCtaThreads>;
using RuntimeCopy = decltype(cute::make_tiled_copy(
    cute::Copy_Atom<cute::PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cutlass::half_t>{},
    cute::Layout<cute::Shape<cute::Int<RuntimePlan::thread_layout_h>,
                             cute::Int<RuntimePlan::thread_layout_w>>>{},
    cute::Layout<cute::Shape<cute::Int<RuntimePlan::values_per_thread>, cute::_1>>{}));
using Descriptor = typename Shipping::Policy::Descriptor;
using RuntimeSmem = decltype(cute::tile_to_shape(
    cute::Layout<cute::Shape<cute::_8, cute::_1>>{},
    cute::Shape<cute::_32, cute::_2, cute::_3>{}));

static_assert(md::kStridedMetadataTileApi == 2);
static_assert(Descriptor::quant_mode == ppu_mixed_policy::QuantMode::FinegrainedScaleZero);
static_assert(Descriptor::tactic_tile_k == 64 && Descriptor::artifact_tile_k == 64 &&
              Descriptor::artifact_low_fold == 1 && Descriptor::stages == 3 &&
              !Descriptor::interleaved);
static_assert(std::is_same_v<typename Descriptor::ScaleTileShapeType,
                             cute::Shape<cute::_32, cute::_2>>);
static_assert(std::is_same_v<typename Descriptor::MetadataPolicyType, RuntimeMetadata>);
static_assert(std::is_same_v<typename Mainloop::MetadataPolicy, RuntimeMetadata>);
static_assert(std::is_same_v<typename Mainloop::ScaleCopyPlan, RuntimePlan>);
static_assert(std::is_same_v<typename Mainloop::GmemTiledCopyScale, RuntimeCopy>);
static_assert(std::is_same_v<typename Mainloop::GmemTiledCopyZero, RuntimeCopy>);
static_assert(std::is_same_v<typename Mainloop::FusedScaleHalfLayout, RuntimeSmem>);
static_assert(Mainloop::Scale_TileN == m8n16_g5_layout_spec::kN);
static_assert(Mainloop::Scale_TileK == m8n16_g5_layout_spec::kScaleTileK);
static_assert(int(cute::size(typename Mainloop::TiledMma{})) ==
              m8n16_g5_layout_spec::kCtaThreads);
static_assert(RuntimePlan::thread_slots == 8 &&
              RuntimePlan::values_per_thread == 8);
static_assert(!Mainloop::is_fused_scale_zero && !Mainloop::is_packed_scale &&
              Mainloop::has_zero_channel);

#ifndef L125_SELECTED_WN
#define L125_SELECTED_WN 32
#endif
using SelectedWarp = cute::Shape<cute::_8, cute::Int<L125_SELECTED_WN>, cute::_64>;
using SelectedPolicy = ppu_mixed_policy::MainloopPolicy<
    Shipping::QuantMode, typename Shipping::BaseSchedule,
    typename Shipping::Tile, typename Shipping::Scale, SelectedWarp,
    Shipping::Stages, Shipping::AiuInterleaved, typename Shipping::ElementB>;
static_assert(std::is_same_v<typename SelectedPolicy::CollectiveOp, Mainloop>,
              "L125 selected policy is not the shipping G5 metadata type");

int main() {
  std::printf("[l125:type] exact G5: FinegrainedScaleZero gs32 "
              "tile=8x32x64 warp=8x32x64 stages=3 int4 CTA32; "
              "metadata plan H4xW2 slots8 values8 PASS\n");
  std::printf("[l125:model] S/Z source=GmemTiledCopyScale/GmemTiledCopyZero "
              "class=CuTe-tiled-copy/Copy_Traits host-modelled; "
              "GmemTiledCopyScalePacked=NOT-SELECTED scalar-global=NONE naked-asm=NONE\n");
  return 0;
}

#else

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cute/arch/copy_ppu.hpp"
#include "cute/atom/copy_traits_ppu.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "grouped_schedule_decode.hpp"
#include "m8n16_g5_layout_spec.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"

namespace md = cutlass::gemm::collective::detail;
namespace spec = m8n16_g5_layout_spec;
using namespace cute;
using Element = cutlass::half_t;
using ScaleTile = Shape<Int<spec::kN>, Int<spec::kScaleTileK>>;
using Metadata = md::MixedMetadataPolicy<
    Element, spec::kScaleTileK, true, md::FlatMetadataAddress>;
using Plan = typename Metadata::template ScaleCopy<spec::kN, spec::kCtaThreads>;
using GmemCopy = decltype(make_tiled_copy(
    Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
    Layout<Shape<Int<Plan::thread_layout_h>, Int<Plan::thread_layout_w>>>{},
    Layout<Shape<Int<Plan::values_per_thread>, _1>>{}));
using SmemLayout = decltype(tile_to_shape(
    Layout<Shape<_8, _1>>{},
    make_shape(Int<spec::kN>{}, Int<spec::kScaleTileK>{}, Int<spec::kStages>{})));

static_assert(md::kStridedMetadataTileApi == 2);
static_assert(spec::kExperts == 256 && spec::kN == 32 && spec::kK == 256 &&
              spec::kGroupSize == 32 && spec::kScaleK == 8 &&
              spec::kScaleTileK == 2 && spec::kStages == 3 &&
              spec::kCtaThreads == 32);
static_assert(Plan::thread_layout_h == 4 && Plan::thread_layout_w == 2 &&
              Plan::thread_slots == 8 && Plan::values_per_thread == 8 &&
              Plan::Coverage::value);
static_assert(int(size(GmemCopy{})) == Plan::thread_slots);
static_assert(sizeof(Element) == sizeof(std::uint16_t));

constexpr int kPlane = spec::kN * spec::kScaleK;
constexpr int kAll = spec::kExperts * kPlane;
constexpr int kGuard = 1024;
constexpr std::uint16_t kPoison = 0xa55au;
using MetadataStride = Stride<_1, int64_t, int64_t>;

MetadataStride tight_metadata_stride() {
  return make_stride(_1{}, int64_t(spec::kN), int64_t(kPlane));
}

std::uint16_t raw(Element const& value) {
  std::uint16_t out;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

void set_raw(Element& value, std::uint16_t bits) {
  std::memcpy(&value, &bits, sizeof(bits));
}

std::uint16_t tag(int expert, int group, int n) {
  return std::uint16_t(expert * kPlane + group * spec::kN + n);
}

struct ExpertStats {
  int scheduler_bad = 0;
  int gep_bad = 0;
  int gz_bad = 0;
  int partition_holes = 0;
  int partition_dups = 0;
  int visit_bad = 0;
  int recover_bad = 0;
  int non_target_poison_bad = 0;
  int min_gz = kAll;
  int max_gz = -1;
};

ExpertStats census_expert(std::vector<Element>& source, int expert) {
  ExpertStats out;
  auto const decoded = quactlize::grouped_schedule::decode_uniform_z(expert, 1);
  out.scheduler_bad += decoded.expert != expert || decoded.slice != 0;

  Element const* explicit_base = source.data() + int64_t(expert) * kPlane;
  for (int group = 0; group < spec::kScaleK; ++group) {
    for (int n = 0; n < spec::kN; ++n) {
      auto const* p = explicit_base + int64_t(group) * spec::kN + n;
      out.gep_bad += p - source.data() != int64_t(expert) * kPlane + group * spec::kN + n;
      out.gep_bad += raw(*p) != tag(expert, group, n);
    }
  }

  auto gZ = md::make_metadata_tile<ScaleTile>(
      source.data(), tight_metadata_stride(),
      spec::kN, int64_t(spec::kScaleK), spec::kExperts,
      decoded.expert, 0);
  auto const* gz_base = raw_pointer_cast(gZ.data());
  out.gz_bad += gz_base - source.data() != int64_t(expert) * kPlane;
  for (int tile = 0; tile < spec::kScaleK / spec::kScaleTileK; ++tile) {
    for (int group_in_tile = 0; group_in_tile < spec::kScaleTileK; ++group_in_tile) {
      int const group = tile * spec::kScaleTileK + group_in_tile;
      for (int n = 0; n < spec::kN; ++n) {
        auto const* p = raw_pointer_cast(&gZ(n, group_in_tile, tile));
        int const off = int(p - source.data());
        out.min_gz = std::min(out.min_gz, off);
        out.max_gz = std::max(out.max_gz, off);
        out.gz_bad += off != expert * kPlane + group * spec::kN + n;
        out.gz_bad += raw(*p) != tag(expert, group, n);
      }
    }
  }

  std::array<int, kPlane> logical_hits{};
  GmemCopy copy;
  for (int slot = 0; slot < Plan::thread_slots; ++slot) {
    auto src = copy.get_slice(slot).partition_S(gZ);
    for (int i = 0; i < int(size(src)); ++i) {
      auto const* p = raw_pointer_cast(&src(i));
      int const inner = int(p - explicit_base);
      if (inner < 0 || inner >= kPlane) {
        ++out.partition_holes;
      } else {
        ++logical_hits[inner];
      }
    }
  }
  for (int hits : logical_hits) {
    out.partition_holes += hits == 0;
    out.partition_dups += hits > 1 ? hits - 1 : 0;
  }

  std::array<int, kPlane> physical_hits{};
  std::array<Element, int(cosize(SmemLayout{}))> smem{};
  for (int metadata_tile = 0;
       metadata_tile < spec::kScaleK / spec::kScaleTileK; ++metadata_tile) {
    for (auto& value : smem) set_raw(value, kPoison);
    std::array<int, int(cosize(SmemLayout{}))> smem_hits{};
    auto sZ = make_tensor(make_smem_ptr(smem.data()), SmemLayout{});
    int const stage = metadata_tile % spec::kStages;
    for (int thread = 0; thread < spec::kCtaThreads; ++thread) {
      int const slot = thread % Plan::thread_slots;
      auto slice = copy.get_slice(slot);
      auto src = slice.partition_S(gZ)(_, _, _, metadata_tile);
      auto dst = slice.partition_D(sZ)(_, _, _, stage);
      if (int(size(src)) != Plan::values_per_thread ||
          int(size(dst)) != Plan::values_per_thread) {
        ++out.visit_bad;
        continue;
      }
      for (int i = 0; i < int(size(src)); ++i) {
        auto const* sp = raw_pointer_cast(&src(i));
        auto* dp = raw_pointer_cast(&dst(i));
        int const inner = int(sp - explicit_base);
        if (inner < 0 || inner >= kPlane) {
          ++out.visit_bad;
        } else {
          ++physical_hits[inner];
          int const smem_offset = int(dp - smem.data());
          if (smem_offset < 0 || smem_offset >= int(smem_hits.size())) {
            ++out.visit_bad;
            continue;
          }
          ++smem_hits[smem_offset];
          std::memcpy(dp, sp, sizeof(Element));
        }
      }
    }
    for (int group_in_tile = 0; group_in_tile < spec::kScaleTileK; ++group_in_tile) {
      int const group = metadata_tile * spec::kScaleTileK + group_in_tile;
      for (int n = 0; n < spec::kN; ++n) {
        out.recover_bad += raw(sZ(n, group_in_tile, stage)) != tag(expert, group, n);
      }
    }
    for (int st = 0; st < spec::kStages; ++st) {
      for (int group_in_tile = 0; group_in_tile < spec::kScaleTileK; ++group_in_tile) {
        for (int n = 0; n < spec::kN; ++n) {
          int const hits = smem_hits[int(&sZ(n, group_in_tile, st) - smem.data())];
          out.visit_bad += hits != (st == stage ? spec::kCtaThreads / Plan::thread_slots : 0);
          if (st != stage) out.non_target_poison_bad += raw(sZ(n, group_in_tile, st)) != kPoison;
        }
      }
    }
  }
  for (int hits : physical_hits) out.visit_bad += hits != spec::kCtaThreads / Plan::thread_slots;
  return out;
}

int main() {
  std::vector<Element> source(kAll + kGuard);
  for (int e = 0; e < spec::kExperts; ++e)
    for (int g = 0; g < spec::kScaleK; ++g)
      for (int n = 0; n < spec::kN; ++n)
        set_raw(source[e * kPlane + g * spec::kN + n], tag(e, g, n));
  for (int i = kAll; i < int(source.size()); ++i) set_raw(source[i], kPoison);

  int tag_roundtrip_bad = 0;
  for (int e = 0; e < spec::kExperts; ++e) {
    for (int inner = 0; inner < kPlane; ++inner) {
      std::uint16_t const bits = raw(source[e * kPlane + inner]);
      tag_roundtrip_bad += int(bits) / kPlane != e || int(bits) % kPlane != inner;
    }
  }
  int scheduler_sweep_bad = 0;
  for (int splits = 1; splits <= 8; ++splits) {
    for (int e = 0; e < spec::kExperts; ++e) {
      for (int slice = 0; slice < splits; ++slice) {
        auto const decoded = quactlize::grouped_schedule::decode_uniform_z(e * splits + slice, splits);
        scheduler_sweep_bad += decoded.expert != e || decoded.slice != slice;
      }
    }
  }

  ExpertStats total;
  for (int e = 0; e < spec::kExperts; ++e) {
    auto const s = census_expert(source, e);
    total.scheduler_bad += s.scheduler_bad;
    total.gep_bad += s.gep_bad;
    total.gz_bad += s.gz_bad;
    total.partition_holes += s.partition_holes;
    total.partition_dups += s.partition_dups;
    total.visit_bad += s.visit_bad;
    total.recover_bad += s.recover_bad;
    total.non_target_poison_bad += s.non_target_poison_bad;
    std::printf("[l125:e] e=%3d scheduler=%3d explicit=[%5d,%5d] "
                "gZ=[%5d,%5d] partition=%s visits=%s recover=%s\n",
                e, quactlize::grouped_schedule::decode_uniform_z(e, 1).expert,
                e * kPlane, e * kPlane + kPlane - 1, s.min_gz, s.max_gz,
                (s.partition_holes || s.partition_dups) ? "BAD" : "256/256",
                s.visit_bad ? "BAD" : "4..4", s.recover_bad ? "BAD" : "256/256");
  }

  int low_expert_bad = 0, high_expert_bad = 0, folded_elements = 0;
  int wrong_l_stride = 0;
  int transpose_bad = 0;
  for (int e = 0; e < spec::kExperts; ++e) {
    int const planted = e < 128 ? e : e - 64;
    int this_fold_bad = 0;
    for (int inner = 0; inner < kPlane; ++inner) {
      this_fold_bad += raw(source[planted * kPlane + inner]) !=
                       std::uint16_t(e * kPlane + inner);
      wrong_l_stride += raw(source[e * (kPlane / 2) + inner]) !=
                        std::uint16_t(e * kPlane + inner);
    }
    folded_elements += this_fold_bad;
    low_expert_bad += e < 128 && this_fold_bad != 0;
    high_expert_bad += e >= 128 && this_fold_bad != kPlane;

    std::array<unsigned char, kPlane> transpose_seen{};
    for (int g = 0; g < spec::kScaleK; ++g) {
      for (int n = 0; n < spec::kN; ++n) {
        int const wrong = n * spec::kScaleK + g;
        ++transpose_seen[wrong];
        transpose_bad += raw(source[e * kPlane + wrong]) != tag(e, g, n);
      }
    }
    for (auto hits : transpose_seen) transpose_bad += hits != 1;
  }

  int raw_oob = 0, raw_in = 0;
  std::array<int, kPlane> raw_hits{};
  {
    auto gZ = md::make_metadata_tile<ScaleTile>(
        source.data(), tight_metadata_stride(),
        spec::kN, int64_t(spec::kScaleK), spec::kExperts, 255, 0);
    Element const* plane = source.data() + 255 * kPlane;
    for (int thread = 0; thread < spec::kCtaThreads; ++thread) {
      auto src = GmemCopy{}.get_slice(thread).partition_S(gZ);
      for (int i = 0; i < int(size(src)); ++i) {
        int const inner = int(raw_pointer_cast(&src(i)) - plane);
        if (inner >= 0 && inner < kPlane) {
          ++raw_in;
          ++raw_hits[inner];
        } else {
          ++raw_oob;
        }
      }
    }
  }
  int raw_holes = 0, raw_duplicate_coords = 0, raw_max_hit = 0;
  for (int hits : raw_hits) {
    raw_holes += hits == 0;
    raw_duplicate_coords += hits > 1;
    raw_max_hit = std::max(raw_max_hit, hits);
  }

  int tile0_holes = 0, tile0_duplicate_coords = 0;
  {
    std::array<int, kPlane> hits{};
    for (int metadata_tile = 0; metadata_tile < spec::kScaleK / spec::kScaleTileK;
         ++metadata_tile) {
      (void)metadata_tile;
      for (int slot = 0; slot < Plan::thread_slots; ++slot) {
        auto gZ = md::make_metadata_tile<ScaleTile>(
            source.data(), tight_metadata_stride(),
            spec::kN, int64_t(spec::kScaleK), spec::kExperts, 0, 0);
        auto src = GmemCopy{}.get_slice(slot).partition_S(gZ)(_, _, _, 0);
        for (int i = 0; i < int(size(src)); ++i) {
          ++hits[int(raw_pointer_cast(&src(i)) - source.data())];
        }
      }
    }
    for (int count : hits) {
      tile0_holes += count == 0;
      tile0_duplicate_coords += count > 1;
    }
  }

  int duplicate_holes = 0, duplicate_collisions = 0;
  for (int e = 0; e < spec::kExperts; ++e) {
    std::array<int, kPlane> owner{};
    owner.fill(-1);
    for (int src = 0; src < kPlane; ++src) {
      int const dst = src == kPlane - 1 ? kPlane - 2 : src;
      duplicate_collisions += owner[dst] >= 0;
      owner[dst] = src;
    }
    for (int src : owner) duplicate_holes += src < 0;
  }

  bool const positive = tag_roundtrip_bad == 0 && scheduler_sweep_bad == 0 &&
      total.scheduler_bad == 0 && total.gep_bad == 0 &&
      total.gz_bad == 0 && total.partition_holes == 0 && total.partition_dups == 0 &&
      total.visit_bad == 0 && total.recover_bad == 0 && total.non_target_poison_bad == 0;
  bool const folded_red = low_expert_bad == 0 && high_expert_bad == 0 &&
                          folded_elements == 128 * kPlane;
  bool const stride_red = wrong_l_stride == (spec::kExperts - 1) * kPlane;
  bool const transpose_red = transpose_bad == spec::kExperts * (kPlane - 2);
  bool const raw_red = raw_in == 640 && raw_oob == 384 && raw_holes == 0 &&
                       raw_duplicate_coords == 192 && raw_max_hit == 4;
  bool const tile0_red = tile0_holes == 192 && tile0_duplicate_coords == 64;
  bool const duplicate_red = duplicate_holes == spec::kExperts &&
                             duplicate_collisions == spec::kExperts;

  std::printf("[l125] positive tag-roundtrip=%d scheduler-S1..8=%d scheduler=%d gep=%d "
              "gZ=%d holes=%d dups=%d visit=%d recover=%d non-target-poison=%d -> %s\n",
              tag_roundtrip_bad, scheduler_sweep_bad, total.scheduler_bad, total.gep_bad, total.gz_bad,
              total.partition_holes, total.partition_dups, total.visit_bad,
              total.recover_bad, total.non_target_poison_bad, positive ? "PASS" : "FAIL");
  // This is the independent calibration point for the CuTe model.  The source
  // address comes from the public tight metadata ABI (explicit int64
  // e*scale_k*N + group*N + n), and every element carries that coordinate as
  // a unique raw-16 tag.  partition_S/partition_D must scatter those tags and
  // the explicit logical indexing above must recover them.  Thus a matching
  // CuTe source/destination pair cannot pass merely by sharing one wrong map.
  std::printf("[l125:anchor] explicit-int64-tight-ABI+unique-raw16-tags "
              "scatter/recover=%s (independent of Copy_Traits agreement)\n",
              total.gep_bad == 0 && total.recover_bad == 0 ? "PASS" : "FAIL");
  std::printf("[l125:boundary] zero-plane address chain is entirely modelled: "
              "make_metadata_tile(dS=tight) -> GmemTiledCopyZero.partition_S -> "
              "partition_D; cp.async is the terminal byte-copy and contributes "
              "no address algebra; no scalar/naked-asm metadata read exists\n");
  std::printf("[l125:red] e>=128->e-64 elements=%d low_experts_bad=%d "
              "high_experts_inexact=%d expected=32768 -> %s\n",
              folded_elements, low_expert_bad, high_expert_bad,
              folded_red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l125:red] half-L-stride mismatches=%d expected=65280 -> %s\n",
              wrong_l_stride, stride_red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l125:red] bijective-NG-transpose mismatches=%d expected=65024 -> %s\n",
              transpose_bad, transpose_red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l125:red] raw-thread-slices in=%d oob=%d holes=%d duplicate-coords=%d "
              "max-hit=%d expected=640/384/0/192/4 -> %s\n",
              raw_in, raw_oob, raw_holes, raw_duplicate_coords, raw_max_hit,
              raw_red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l125:red] all-four-use-tile0 holes=%d duplicate-coords=%d "
              "expected=192/64 -> %s\n",
              tile0_holes, tile0_duplicate_coords, tile0_red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l125:red] duplicate-owner holes=%d collisions=%d expected=256/256 -> %s\n",
              duplicate_holes, duplicate_collisions,
              duplicate_red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l125] scope=zero-plane-only B-addressing=NOT-COVERED result=%s\n",
              positive && folded_red && stride_red && transpose_red && raw_red && tile0_red && duplicate_red
                  ? "PASS" : "FAIL");
  return positive && folded_red && stride_red && transpose_red && raw_red && tile0_red && duplicate_red ? 0 : 1;
}

#endif
