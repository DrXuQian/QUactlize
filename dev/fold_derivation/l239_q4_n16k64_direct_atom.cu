// L239 -- host-only production-type oracle for one direct Q4 N16 x K64
// delivery atom.
//
// This is deliberately narrower than L238.  It binds the real delivery
// policy to the concrete plain shared representation proposed for the direct
// reader:
//
//   logical Q4 atom : N16 x K64              = 1024 int4 = 512 B
//   physical shared: [K/16=4][2*N=32] u32    = 128 words = 512 B
//   S2R delivery   : 32 lanes x one uint128  = 512 B
//
// The lane map is the row-major flattening of the physical tensor:
//   row  = lane / 8
//   word = 4 * (lane % 8) + vector_word
// so every lane reads four adjacent u32 words from a 16-byte-aligned start.

#if defined(L239_COMPILER_PROBE)

#include <cuda_fp16.h>

__global__ void l239_compiler_probe(half const* x, half* y) {
  int const i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i == 0) y[0] = __hadd(x[0], x[0]);
}

int main() { return 0; }

#else

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_q4_n16k64_delivery.hpp"

namespace {

namespace bd =
    cutlass::gemm::collective::detail::quactlize_b_delivery;
namespace direct =
    cutlass::gemm::collective::detail::quactlize_q4_n16k64_delivery;
using namespace cute;

constexpr int kN = 16;
constexpr int kLogicalK = 64;
constexpr int kBitsPerCode = 4;
constexpr int kPhysicalKRows = kLogicalK / 16;
constexpr int kPhysicalNWords = 2 * kN;
constexpr int kLanes = 32;
constexpr int kWordsPerLane = 4;
constexpr int kVectorBytes = 16;
constexpr int kAlignmentBytes = direct::kSharedAlignmentBytes;
constexpr int kPhysicalWords = kPhysicalKRows * kPhysicalNWords;
constexpr int kStageBytes = kPhysicalWords * int(sizeof(std::uint32_t));
constexpr int kLogicalBytes = kN * kLogicalK * kBitsPerCode / 8;

using Physical = direct::PhysicalShared<kN, kLogicalK>;
using PhysicalSharedLayout = typename Physical::Layout;

// This is intentionally a different type and coordinate system.  It is the
// logical fp16 converter/MMA destination, never the physical shared source.
using LogicalDestinationLayout =
    Layout<Shape<_16, _64>, Stride<_64, _1>>;

using DirectShared = typename Physical::Contract;
using AiuProvider = direct::AiuPlainProvider<kN, kLogicalK>;
using CpProvider = direct::CpAsyncProvider<kN, kLogicalK, 32>;
using AiuPlainWriter = typename AiuProvider::WriterType;
using CpAsyncWriter = typename CpProvider::WriterType;
using ProductionUniversalReader = typename AiuProvider::ReaderType;
using UniversalVectorAtom = typename ProductionUniversalReader::CopyAtom;
using UniversalLaneCopy = typename ProductionUniversalReader::Copy;

template <class Shared_>
struct UniversalReaderFor {
  using S2R = bd::UniversalS2R;
  using Shared = Shared_;
  using CopyAtom = UniversalVectorAtom;
  using TiledCopy = UniversalLaneCopy;
};

#if defined(L239_PLANT_STAGE_BYTES_MISMATCH)
using ReaderShared = bd::PhysicalSharedContract<
    PhysicalSharedLayout, std::uint32_t, kStageBytes - kAlignmentBytes,
    kN, kLogicalK, kAlignmentBytes>;
#elif defined(L239_PLANT_N_ATOM_MISMATCH)
using ReaderShared = bd::PhysicalSharedContract<
    PhysicalSharedLayout, std::uint32_t, kStageBytes,
    32, kLogicalK, kAlignmentBytes>;
#elif defined(L239_PLANT_K_ATOM_MISMATCH)
using ReaderShared = bd::PhysicalSharedContract<
    PhysicalSharedLayout, std::uint32_t, kStageBytes,
    kN, 32, kAlignmentBytes>;
#elif defined(L239_PLANT_ALIGNMENT_MISMATCH)
using ReaderShared = bd::PhysicalSharedContract<
    PhysicalSharedLayout, std::uint32_t, kStageBytes,
    kN, kLogicalK, 16>;
#elif defined(L239_PLANT_SOURCE_DEST_LAYOUT_MIX)
using ReaderShared = bd::PhysicalSharedContract<
    LogicalDestinationLayout, std::uint32_t, kStageBytes,
    kN, kLogicalK, kAlignmentBytes>;
#else
using ReaderShared = DirectShared;
#endif

#if defined(L239_PLANT_STAGE_BYTES_MISMATCH) || \
    defined(L239_PLANT_N_ATOM_MISMATCH) || \
    defined(L239_PLANT_K_ATOM_MISMATCH) || \
    defined(L239_PLANT_ALIGNMENT_MISMATCH) || \
    defined(L239_PLANT_SOURCE_DEST_LAYOUT_MIX)
using UniversalReader = UniversalReaderFor<ReaderShared>;
using AiuPlainChain = bd::BoundBDelivery<AiuPlainWriter, UniversalReader>;
using CpAsyncChain = bd::BoundBDelivery<CpAsyncWriter, UniversalReader>;
#else
using UniversalReader = ProductionUniversalReader;
using AiuPlainChain = typename AiuProvider::Binding;
using CpAsyncChain = typename CpProvider::Binding;
#endif

// Production-shaped repetition witness: the same N16xK64 atom composes a
// 64x256 logical tile without changing either endpoint's physical contract.
using WidePhysical = direct::PhysicalShared<64, 256>;
using WideAiuProvider = direct::AiuPlainProvider<64, 256>;
#if defined(L239_PLANT_THREAD_COUNT_MISMATCH)
using WideCpProvider = direct::CpAsyncProvider<64, 256, 96>;
#else
using WideCpProvider = direct::CpAsyncProvider<64, 256, 128>;
using WideCp64Provider = direct::CpAsyncProvider<64, 256, 64>;
using WideCp256Provider = direct::CpAsyncProvider<64, 256, 256>;
#endif
static_assert(WidePhysical::physical_k_rows == 16);
static_assert(WidePhysical::physical_n_words == 128);
static_assert(WidePhysical::stage_bytes == 8192);
static_assert(WideAiuProvider::WriterType::cube_count == 4);
static_assert(std::is_same_v<typename WideAiuProvider::Shared,
                             typename WideCpProvider::Shared>);
#if !defined(L239_PLANT_THREAD_COUNT_MISMATCH)
static_assert(WideCpProvider::WriterType::thread_count == 128);
static_assert(WideCpProvider::WriterType::vector_rounds == 4);
static_assert(WideCpProvider::WriterType::thread_rows == 4);
static_assert(WideCpProvider::WriterType::thread_n_vectors == 32);
static_assert(WideCp64Provider::WriterType::vector_rounds == 8);
static_assert(WideCp64Provider::WriterType::thread_rows == 2);
static_assert(WideCp256Provider::WriterType::vector_rounds == 2);
static_assert(WideCp256Provider::WriterType::thread_rows == 8);
#endif

static_assert(kStageBytes == 512 && kLogicalBytes == 512,
              "the physical and logical N16xK64 Q4 atoms must both be 512 B");
static_assert(Physical::stage_bytes == kStageBytes);
static_assert(kPhysicalWords == 128);
static_assert(sizeof_bits<cute::uint128_t>::value == 128);
static_assert(UniversalVectorAtom::NumValSrc == kWordsPerLane &&
              UniversalVectorAtom::NumValDst == kWordsPerLane,
              "one Universal uint128 copy must expose four u32 values");
static_assert(kLanes * kVectorBytes == kStageBytes);
static_assert(kVectorBytes * 8 / kBitsPerCode == 32,
              "one lane must receive exactly 32 Q4 codes");
static_assert(cosize_v<PhysicalSharedLayout> == kPhysicalWords);
using ExpectedAiuInst = PPU0010_AIU_LOAD<
    C<4096>, std::uint32_t, false, false>;
static_assert(std::is_same_v<typename AiuPlainWriter::CopyInst,
                             ExpectedAiuInst>,
              "the AIU writer must be the real PPU0010 b32 no-trans linear op");
static_assert(!AiuPlainWriter::swizzled);
static_assert(direct::kSharedAlignmentBytes == 32);
static_assert(std::is_same_v<typename CpAsyncWriter::CopyInst,
                             PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>>);
static_assert(std::is_same_v<typename ProductionUniversalReader::CopyInst,
                             UniversalCopy<cute::uint128_t>>);
static_assert(DirectShared::n_atom == kN &&
              DirectShared::logical_k_atom == kLogicalK &&
              DirectShared::alignment_bytes == kAlignmentBytes);
static_assert(std::is_same_v<typename AiuPlainChain::Shared,
                             typename CpAsyncChain::Shared>,
              "AIU-plain and cp.async must publish the identical shared ABI");
static_assert(AiuPlainChain::single_issuer &&
              CpAsyncChain::thread_partitioned,
              "the two writers share bytes, not issue ownership");

bool prove_lane_map() {
  std::array<int, kPhysicalWords> hits{};
  bool exact = true;

  PhysicalSharedLayout physical{};
  typename UniversalLaneCopy::TiledLayout_TV tiled{};
  for (int lane = 0; lane < kLanes; ++lane) {
    int const row = lane / 8;
    int const word0 = kWordsPerLane * (lane % 8);
    int const byte0 = int(physical(make_coord(row, word0))) *
                      int(sizeof(std::uint32_t));
    exact &= byte0 % kVectorBytes == 0;

    for (int v = 0; v < kWordsPerLane; ++v) {
      int const physical_offset =
          int(physical(make_coord(row, word0 + v)));
      int const tiled_offset = int(tiled(make_coord(lane, v)));
      exact &= physical_offset == kWordsPerLane * lane + v;
      exact &= tiled_offset == physical_offset;
      if (physical_offset < 0 || physical_offset >= kPhysicalWords) {
        exact = false;
      } else {
        ++hits[std::size_t(physical_offset)];
      }
    }
  }

  for (int count : hits) exact &= count == 1;
  return exact;
}

template <class Physical_, class Writer_>
bool prove_cta_stage_map() {
  constexpr int kThreads = Writer_::thread_count;
  constexpr int kRounds = Writer_::vector_rounds;
  constexpr int kWords = Physical_::physical_words;
  static_assert(kRounds * kThreads * kWordsPerLane == kWords);

  std::array<int, kWords> src_hits{};
  std::array<int, kWords> dst_hits{};
  bool exact = true;
  typename Physical_::Layout physical{};
  typename Writer_::Copy copy{};
  auto identity =
      make_identity_tensor(typename Physical_::CoordinateShape{});

  // Exercise the actual CuTe get_slice/partition_S path that the future
  // collective will call.  No modulo remapping is admitted: every physical
  // CTA thread is a distinct coordinate in Copy's ThreadLayout.
  for (int thread = 0; thread < kThreads; ++thread) {
    auto src = copy.get_slice(thread).partition_S(identity);
    auto dst = copy.get_slice(thread).partition_D(identity);
    exact &= int(size(src)) == kRounds * kWordsPerLane;
    exact &= int(size(dst)) == int(size(src));
    for (int i = 0; i < int(size(src)); ++i) {
      auto const src_coord = src(i);
      auto const dst_coord = dst(i);
      int const row = int(get<0>(src_coord));
      int const nword = int(get<1>(src_coord));
      int const dst_row = int(get<0>(dst_coord));
      int const dst_nword = int(get<1>(dst_coord));
      int const flat = int(physical(make_coord(row, nword)));
      int const dst_flat =
          int(physical(make_coord(dst_row, dst_nword)));
      // A full ownership denominator is insufficient if the TiledCopy
      // permutes source and destination differently.  The direct offline ABI
      // requires the same physical [K/16][2N] coordinate at both endpoints.
      exact &= flat == dst_flat;
      if (i % kWordsPerLane == 0) {
        exact &= (flat * int(sizeof(std::uint32_t))) % kVectorBytes == 0;
        exact &= (dst_flat * int(sizeof(std::uint32_t))) %
                     kVectorBytes == 0;
      }
      exact &= row >= 0 && row < Physical_::physical_k_rows;
      exact &= nword >= 0 && nword < Physical_::physical_n_words;
      exact &= dst_row >= 0 && dst_row < Physical_::physical_k_rows;
      exact &= dst_nword >= 0 &&
               dst_nword < Physical_::physical_n_words;
      if (flat < 0 || flat >= kWords) {
        exact = false;
      } else {
        ++src_hits[std::size_t(flat)];
      }
      if (dst_flat < 0 || dst_flat >= kWords) {
        exact = false;
      } else {
        ++dst_hits[std::size_t(dst_flat)];
      }
    }
  }

  for (int count : src_hits) exact &= count == 1;
  for (int count : dst_hits) exact &= count == 1;
  return exact;
}

template <class Physical_, class Writer_>
bool prove_aiu_stage_map() {
  constexpr int kWords = Physical_::physical_words;
  std::array<int, kWords> src_hits{};
  std::array<int, kWords> dst_hits{};
  bool exact = true;
  typename Physical_::Layout physical{};
  typename Writer_::Copy copy{};
  auto identity =
      make_identity_tensor(typename Physical_::CoordinateShape{});

  // The AIU atom is opaque and has one logical issuer, but one copy() still
  // has to repeat that atom across every N16 cohort in the full stage.  Prove
  // the actual TiledCopy partition rather than inferring that repetition from
  // cube_count alone.
  auto src = copy.get_slice(0).partition_S(identity);
  auto dst = copy.get_slice(0).partition_D(identity);
  exact &= int(size(src)) == kWords;
  exact &= int(size(dst)) == kWords;
  for (int i = 0; i < int(size(src)); ++i) {
    auto const src_coord = src(i);
    auto const dst_coord = dst(i);
    int const src_flat = int(physical(src_coord));
    int const dst_flat = int(physical(dst_coord));
    exact &= src_flat == dst_flat;
    if (src_flat < 0 || src_flat >= kWords) {
      exact = false;
    } else {
      ++src_hits[std::size_t(src_flat)];
    }
    if (dst_flat < 0 || dst_flat >= kWords) {
      exact = false;
    } else {
      ++dst_hits[std::size_t(dst_flat)];
    }
  }
  for (int count : src_hits) exact &= count == 1;
  for (int count : dst_hits) exact &= count == 1;
  return exact;
}

}  // namespace

int main() {
  bool const lane_map_exact = prove_lane_map();
  bool const cta_stage_exact =
      prove_cta_stage_map<WidePhysical,
                          typename WideCpProvider::WriterType>();
  bool const aiu_stage_exact =
      prove_aiu_stage_map<WidePhysical,
                          typename WideAiuProvider::WriterType>();
  std::printf(
      "L239 DIRECT_ATOM n=%d logical_k=%d physical=4x32u32 "
      "bytes=%d lanes=%d vector_bytes=%d codes_per_lane=%d alignment=%d "
      "coverage=%s result=%s\n",
      kN, kLogicalK, kStageBytes, kLanes, kVectorBytes,
      kVectorBytes * 8 / kBitsPerCode, kAlignmentBytes,
      lane_map_exact ? "128/128" : "BAD",
      lane_map_exact ? "PASS" : "FAIL");
  std::printf(
      "L239 DIRECT_CHAIN writer=aiu-plain reader=universal issue=single "
      "shared=same result=PASS\n");
  std::printf(
      "L239 DIRECT_CHAIN writer=cp-async reader=universal issue=threads "
      "shared=same result=PASS\n");
  std::printf(
      "L239 CTA_STAGE n=64 logical_k=256 threads=128 rounds=4 "
      "coverage=%s result=%s\n",
      cta_stage_exact ? "2048/2048" : "BAD",
      cta_stage_exact ? "PASS" : "FAIL");
  std::printf(
      "L239 AIU_STAGE n=64 logical_k=256 issuer=1 cubes=4 "
      "coverage=%s result=%s\n",
      aiu_stage_exact ? "2048/2048" : "BAD",
      aiu_stage_exact ? "PASS" : "FAIL");
  std::printf(
      "L239 N16_K64_DIRECT_ATOM %s chains=2 lanes=32 vectors=32 "
      "words=128 cta_threads=128 reds=6\n",
      lane_map_exact && cta_stage_exact && aiu_stage_exact ? "PASS" :
                                                            "FAIL");
  return lane_map_exact && cta_stage_exact && aiu_stage_exact ? 0 : 1;
}

#endif
