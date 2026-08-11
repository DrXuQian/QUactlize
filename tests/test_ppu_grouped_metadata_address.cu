// G5 metadata-address probe for the real grouped ordinary/F=1 ScaleZero path.
//
// This is diagnostic, not a numerical oracle.  The controlled q==8 fixture
// makes B's contribution irrelevant and traces only the zero plane at the
// expert boundary 127/128/129.  One launch records four progressively later
// observations: an explicit int64 GEP, the CuTe gZ slice, partition_S's source
// pointer, and the value that the production cp.async copy left in smem.
// Finding the existing mismatch is therefore a successful diagnosis; only an
// incomplete/self-inconsistent trace makes this executable fail.

// All 256 experts deliberately have M=1.  Besides exercising the real expert
// scheduler, that keeps the grouped launcher on its uniform path so it cannot
// reuse the trace workspace for the ragged m-tile prefix.

// PPU_DEFS=PPU_METADATA_ADDR_PROBE=1 TARGET=test_ppu_grouped_metadata_address ./build.sh

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "moe_grouped_ppu.cuh"
#include "ppu_group_schedule.hpp"

#if !defined(PPU_METADATA_ADDR_PROBE) || (PPU_METADATA_ADDR_PROBE == 0)
#error test_ppu_grouped_metadata_address requires PPU_METADATA_ADDR_PROBE=1 in both host and device compiles
#endif

namespace {

using half_t = cutlass::half_t;
using int4_t = cutlass::int4b_t;
using QM = moe_grouped_ppu::QuantMode;
using GS = moe_grouped_ppu::GroupShape;
using DStride = moe_grouped_ppu::DStride;
using Trace = cutlass::gemm::kernel::GroupedMetadataAddressProbe;
using ShapeRecord = cutlass::gemm::kernel::GroupedMetadataShapeRecord;
using CopyRecord = cutlass::gemm::kernel::GroupedMetadataCopyRecord;

constexpr int kE = 256;
constexpr int kM = 1;
constexpr int kN = 32;
constexpr int kK = 256;
constexpr int kGs = 32;
constexpr int kScaleK = kK / kGs;
constexpr int kPlaneElements = kN * kScaleK;
constexpr int kTM = 8;
constexpr int kTN = 32;
constexpr int kTK = 64;
constexpr int kWM = 8;
constexpr int kWN = 32;
constexpr int kStages = 3;
constexpr int kExperts[3] = {127, 128, 129};

using Schedule = ppu_group_schedule::FinegrainedSchedule<kGs>;
using Tile = cute::Shape<cute::Int<kTM>, cute::Int<kTN>, cute::Int<kTK>>;
using Warp = cute::Shape<cute::Int<kWM>, cute::Int<kWN>, cute::Int<kTK>>;
using Scale = cute::Shape<cute::Int<kTN>, cute::_2>;

// The high byte names the expert and the low byte names the element inside
// that expert's tight (N,scale_k) plane.  These are raw fp16 bits; no floating
// arithmetic is performed, so even NaN encodings remain useful payload tags.
constexpr std::uint16_t tag_bits(int expert, int linear) {
  return std::uint16_t((expert << 8) | linear);
}

int expert_slot(int expert) {
  for (int i = 0; i < 3; ++i) if (kExperts[i] == expert) return i;
  return -1;
}

struct ExpertStats {
  int explicit_addr_bad = 0;
  int explicit_value_bad = 0;
  int gz_addr_bad = 0;
  int gz_value_bad = 0;
  int partition_outside = 0;
  int partition_value_bad = 0;
  int partition_holes = 0;
  int partition_duplicates = 0;
  int cp_async_bad = 0;
  int destination_bad = 0;
  std::array<int, 256> gz_source_experts{};
  std::array<int, 256> partition_source_experts{};
  std::array<int, 256> cp_async_source_experts{};
  std::map<std::int64_t, int> gz_address_deltas;
  std::string seam;
};

void print_distribution(char const* label, std::array<int, 256> const& counts) {
  std::printf(" %s={", label);
  bool first = true;
  for (int e = 0; e < 256; ++e) {
    if (!counts[e]) continue;
    std::printf("%s%d:%d", first ? "" : ",", e, counts[e]);
    first = false;
  }
  std::printf("}");
}

void print_delta_distribution(
    char const* label, std::map<std::int64_t, int> const& counts) {
  std::printf(" %s={", label);
  bool first = true;
  for (auto const& [delta, count] : counts) {
    std::printf("%s%lld:%d", first ? "" : ",",
                static_cast<long long>(delta), count);
    first = false;
  }
  std::printf("}");
}

int run_probe() {
  // All pointers are real and correctly sized even though probe mode exits
  // before the GEMM.  That keeps load_init/descriptors identical to shipping.
  std::vector<half_t> A(std::size_t(kE) * kK, half_t(1.0f));
  std::vector<std::uint8_t> B(std::size_t(kE) * kN * kK / 2, 0x88u); // q==8
  std::vector<half_t> scales(std::size_t(kE) * kPlaneElements, half_t(1.0f));
  std::vector<half_t> zeros(std::size_t(kE) * kPlaneElements);
  for (int e = 0; e < kE; ++e) {
    for (int linear = 0; linear < kPlaneElements; ++linear) {
      zeros[std::size_t(e) * kPlaneElements + linear] =
          half_t::bitcast(tag_bits(e, linear));
    }
  }

  std::vector<GS> shapes(kE, cute::make_shape(kM, kN, kK));
  std::vector<int> group_m(kE, kM);
  std::vector<half_t*> ptrs(kE);
  std::vector<DStride> strides(kE);
  cutlass::DeviceAllocation<half_t> dA(A.size());
  cutlass::DeviceAllocation<std::uint8_t> dB(B.size());
  cutlass::DeviceAllocation<half_t> dScale(scales.size());
  cutlass::DeviceAllocation<half_t> dZero(zeros.size());
  cutlass::DeviceAllocation<half_t> dD(std::size_t(kE) * kN);
  cutlass::DeviceAllocation<GS> dShapes(kE);
  cutlass::DeviceAllocation<int> dGroupM(kE);
  cutlass::DeviceAllocation<half_t*> dPtrs(kE);
  cutlass::DeviceAllocation<DStride> dStrides(kE);
  cutlass::DeviceAllocation<Trace> dTrace(1);

  dA.copy_from_host(A.data());
  dB.copy_from_host(B.data());
  dScale.copy_from_host(scales.data());
  dZero.copy_from_host(zeros.data());
  dShapes.copy_from_host(shapes.data());
  dGroupM.copy_from_host(group_m.data());
  for (int e = 0; e < kE; ++e) {
    ptrs[e] = dD.get() + std::size_t(e) * kN;
    strides[e] = cutlass::make_cute_packed_stride(
        DStride{}, cute::make_shape(kM, kN, 1));
  }
  dPtrs.copy_from_host(ptrs.data());
  dStrides.copy_from_host(strides.data());
  Trace trace{};
  dTrace.copy_from_host(&trace);

  char const* previous = std::getenv("MOEG_PROBE");
  std::string previous_value = previous ? previous : "";
  ::setenv("MOEG_PROBE", "2", 1);
  int const failures_before = moe_grouped_ppu::moeg_fail_count();
  bool const launched = moe_grouped_ppu::launch<
      QM::FinegrainedScaleZero, Schedule, Tile, Scale, Warp,
      kStages, false, int4_t>(
          dA.get(), reinterpret_cast<int4_t const*>(dB.get()),
          dScale.get(), dZero.get(), dPtrs.get(), dStrides.get(),
          dGroupM.get(), kM, kN, kK, kE, kGs, dShapes.get(),
          shapes.data(), nullptr, reinterpret_cast<char*>(dTrace.get()),
          sizeof(Trace), nullptr);
  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  if (previous) ::setenv("MOEG_PROBE", previous_value.c_str(), 1);
  else ::unsetenv("MOEG_PROBE");
  int const launch_failures = moe_grouped_ppu::moeg_fail_count() - failures_before;
  dTrace.copy_to_host(&trace);

  if (!launched || launch_failures != 0) {
    std::printf("[G5:ADDR] INCOMPLETE launch=%d launch_failures=%d\n",
                int(launched), launch_failures);
    return 1;
  }

  constexpr std::uint32_t expected_shape = 3u * kPlaneElements;
  std::uint32_t const expected_copy =
      3u * trace.thread_slots * trace.metadata_tiles * trace.values_per_thread;
  bool complete =
      trace.magic == cutlass::gemm::kernel::kGroupedMetadataProbeMagic &&
      trace.version == cutlass::gemm::kernel::kGroupedMetadataProbeVersion &&
      trace.overflow == 0 && trace.configuration_errors == 0 &&
      trace.shape_count == expected_shape && trace.copy_count == expected_copy &&
      trace.cta_threads == 32 && trace.thread_slots == 8 &&
      trace.metadata_tiles == 4 && trace.values_per_thread == 8;
  for (int i = 0; i < 3; ++i) complete &= trace.expert_ctas[i] == 1;

  std::array<std::array<int, kPlaneElements>, 3> shape_seen{};
  std::array<std::array<ShapeRecord, kPlaneElements>, 3> shape_records{};
  std::array<std::array<int, kPlaneElements>, 3> partition_seen{};
  std::array<std::set<std::tuple<int, int, int>>, 3> copy_keys;
  std::array<std::array<std::set<std::uint64_t>, 4>, 3> destination_sets;
  std::array<ExpertStats, 3> stats{};
  std::uintptr_t const zero_base = reinterpret_cast<std::uintptr_t>(dZero.get());

  for (std::uint32_t i = 0; i < trace.shape_count && i < expected_shape; ++i) {
    ShapeRecord const& r = trace.shape[i];
    int const slot = expert_slot(r.scheduler_expert);
    if (slot < 0 || r.local_n < 0 || r.local_n >= kN ||
        r.scale_group < 0 || r.scale_group >= kScaleK) {
      complete = false;
      continue;
    }
    int const linear = r.scale_group * kN + r.local_n;
    ++shape_seen[slot][linear];
    shape_records[slot][linear] = r;
    std::uintptr_t const expected_addr = zero_base +
        2u * (std::size_t(r.scheduler_expert) * kPlaneElements + linear);
    std::uint16_t const expected_bits = tag_bits(r.scheduler_expert, linear);
    stats[slot].explicit_addr_bad += r.explicit_addr != expected_addr;
    stats[slot].explicit_value_bad += r.explicit_bits != expected_bits;
    stats[slot].gz_addr_bad += r.gz_addr != expected_addr;
    stats[slot].gz_value_bad += r.gz_bits != expected_bits;
    ++stats[slot].gz_source_experts[r.gz_bits >> 8];
    stats[slot].gz_address_deltas[
        std::int64_t(r.gz_addr) - std::int64_t(expected_addr)]++;
  }
  for (int slot = 0; slot < 3; ++slot) {
    for (int linear = 0; linear < kPlaneElements; ++linear) {
      if (shape_seen[slot][linear] != 1) complete = false;
    }
  }

  // Counts locate the failing seam, but the requested diagnosis is about the
  // address itself.  Print deterministic boundary witnesses plus the first
  // sixteen divergent CuTe elements for every expert.  In particular this
  // makes e=128's within-plane split visible instead of collapsing it into an
  // average "read expert" value.
  constexpr int sentinels[] = {0, 31, 32, 127, 128, 223, 255};
  for (int slot = 0; slot < 3; ++slot) {
    std::set<int> detail_linear(std::begin(sentinels), std::end(sentinels));
    for (int linear = 0; linear < kPlaneElements && detail_linear.size() < 23; ++linear) {
      ShapeRecord const& r = shape_records[slot][linear];
      std::uintptr_t const expected_addr = zero_base +
          2u * (std::size_t(kExperts[slot]) * kPlaneElements + linear);
      if (r.gz_addr != expected_addr ||
          r.gz_bits != tag_bits(kExperts[slot], linear)) {
        detail_linear.insert(linear);
      }
    }
    for (int linear : detail_linear) {
      ShapeRecord const& r = shape_records[slot][linear];
      std::printf(
          "[G5:ADDR][shape-detail] scheduler_e=%d group=%d n=%d "
          "explicit=%#llx/%#06x gz_base=%#llx gz=%#llx/%#06x gz_tag_e=%u\n",
          r.scheduler_expert, r.scale_group, r.local_n,
          static_cast<unsigned long long>(r.explicit_addr), unsigned(r.explicit_bits),
          static_cast<unsigned long long>(r.gz_base_addr),
          static_cast<unsigned long long>(r.gz_addr), unsigned(r.gz_bits),
          unsigned(r.gz_bits >> 8));
    }
  }

  std::array<int, 3> copy_detail_lines{};
  for (std::uint32_t i = 0; i < trace.copy_count && i < expected_copy; ++i) {
    CopyRecord const& r = trace.copy[i];
    int const slot = expert_slot(r.scheduler_expert);
    if (slot < 0 || r.thread_idx < 0 || r.thread_idx >= int(trace.thread_slots) ||
        r.copy_slot != r.thread_idx || r.metadata_tile < 0 ||
        r.metadata_tile >= int(trace.metadata_tiles) || r.value_idx < 0 ||
        r.value_idx >= int(trace.values_per_thread)) {
      complete = false;
      continue;
    }
    auto key = std::make_tuple(r.thread_idx, r.metadata_tile, r.value_idx);
    if (!copy_keys[slot].insert(key).second) complete = false;
    destination_sets[slot][r.metadata_tile].insert(r.destination_addr);

    std::uintptr_t const plane_begin = zero_base +
        2u * std::size_t(r.scheduler_expert) * kPlaneElements;
    std::uintptr_t const plane_end = plane_begin + 2u * kPlaneElements;
    bool const in_plane = r.partition_addr >= plane_begin &&
                          r.partition_addr < plane_end &&
                          ((r.partition_addr - plane_begin) % 2u == 0);
    stats[slot].partition_outside += !in_plane;
    int linear = -1;
    if (in_plane) {
      linear = int((r.partition_addr - plane_begin) / 2u);
      ++partition_seen[slot][linear];
      stats[slot].partition_value_bad +=
          r.partition_bits != tag_bits(r.scheduler_expert, linear);
    }
    ++stats[slot].partition_source_experts[r.partition_bits >> 8];
    ++stats[slot].cp_async_source_experts[r.cp_async_bits >> 8];
    stats[slot].cp_async_bad += r.cp_async_bits != r.partition_bits;

    bool const bad = !in_plane ||
        (linear >= 0 && r.partition_bits != tag_bits(r.scheduler_expert, linear)) ||
        r.cp_async_bits != r.partition_bits;
    if (bad && copy_detail_lines[slot] < 16) {
      std::printf("[G5:ADDR][copy-detail] scheduler_e=%d slot=%d tile=%d value=%d "
                  "src=%#llx src_e=%u src_linear=%u smem=%#llx smem_e=%u smem_linear=%u\n",
                  r.scheduler_expert, r.copy_slot, r.metadata_tile, r.value_idx,
                  static_cast<unsigned long long>(r.partition_addr),
                  unsigned(r.partition_bits >> 8), unsigned(r.partition_bits & 0xff),
                  static_cast<unsigned long long>(r.destination_addr),
                  unsigned(r.cp_async_bits >> 8), unsigned(r.cp_async_bits & 0xff));
      ++copy_detail_lines[slot];
    }
  }

  for (int slot = 0; slot < 3; ++slot) {
    for (int linear = 0; linear < kPlaneElements; ++linear) {
      stats[slot].partition_holes += partition_seen[slot][linear] == 0;
      stats[slot].partition_duplicates += partition_seen[slot][linear] > 1;
    }
    for (int tile = 0; tile < int(trace.metadata_tiles); ++tile) {
      if (destination_sets[slot][tile].size() !=
          std::size_t(trace.thread_slots * trace.values_per_thread)) {
        ++stats[slot].destination_bad;
      }
    }
    if (copy_keys[slot].size() !=
        std::size_t(trace.thread_slots * trace.metadata_tiles * trace.values_per_thread)) {
      complete = false;
    }

    if (stats[slot].explicit_addr_bad || stats[slot].explicit_value_bad)
      stats[slot].seam = "EXPLICIT_INT64_GEP_OR_FIXTURE";
    else if (stats[slot].gz_addr_bad || stats[slot].gz_value_bad)
      stats[slot].seam = "GZ_SHAPE_SLICE_OR_LOWERING";
    else if (stats[slot].partition_outside || stats[slot].partition_value_bad ||
             stats[slot].partition_holes || stats[slot].partition_duplicates)
      stats[slot].seam = "PARTITION_S";
    else if (stats[slot].cp_async_bad || stats[slot].destination_bad)
      stats[slot].seam = "CP_ASYNC_OR_SMEM_DELIVERY";
    else
      stats[slot].seam = "AFTER_CP_ASYNC_METADATA_CONSUMER";
  }

  std::printf("[G5:ADDR] trace magic=%#x version=%u shape=%u/%u copy=%u/%u "
              "overflow=%u config_errors=%u cta_threads=%u slots=%u tiles=%u values=%u\n",
              trace.magic, trace.version, trace.shape_count, expected_shape,
              trace.copy_count, expected_copy, trace.overflow,
              trace.configuration_errors, trace.cta_threads, trace.thread_slots,
              trace.metadata_tiles, trace.values_per_thread);
  for (int slot = 0; slot < 3; ++slot) {
    ExpertStats const& s = stats[slot];
    std::printf("[G5:ADDR] expert=%d scheduler_ctas=%u explicit_addr_bad=%d explicit_value_bad=%d "
                "gz_addr_bad=%d gz_value_bad=%d partition_outside=%d partition_value_bad=%d "
                "partition_holes=%d partition_duplicates=%d cp_async_bad=%d destination_bad=%d seam=%s",
                kExperts[slot], trace.expert_ctas[slot], s.explicit_addr_bad,
                s.explicit_value_bad, s.gz_addr_bad, s.gz_value_bad,
                s.partition_outside, s.partition_value_bad, s.partition_holes,
                s.partition_duplicates, s.cp_async_bad, s.destination_bad,
                s.seam.c_str());
    print_distribution("partition_experts", s.partition_source_experts);
    print_distribution("cp_async_experts", s.cp_async_source_experts);
    print_distribution("gz_experts", s.gz_source_experts);
    print_delta_distribution("gz_addr_delta_bytes", s.gz_address_deltas);
    std::printf("\n");
  }
  std::printf("[G5:ADDR] scope=B_NOT_COVERED q==8 nulls B; this probe proves only zero-plane addressing\n");
  std::printf("[G5:ADDR] %s: all four observation layers were exercised for experts 127/128/129\n",
              complete ? "COMPLETE" : "INCOMPLETE");
  return complete ? 0 : 1;
}

} // namespace

int main() {
  std::printf("== G5 grouped metadata address probe ==\n");
  return run_probe();
}
