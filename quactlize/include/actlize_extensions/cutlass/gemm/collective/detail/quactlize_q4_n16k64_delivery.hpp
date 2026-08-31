#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cute/arch/copy_ppu.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_ppu.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"

#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_b_delivery_policy.hpp"

// Production-adjacent, but deliberately unselected, Q4 delivery providers.
//
// The two writers below publish one identical *plain* shared-memory ABI:
//
//   logical B tile : [N, K] Q4
//   physical words : [K / 16][2 * N] uint32_t
//   CuTe coordinate: (k16, n_word), stride (2 * N, 1)
//
// In other words, the right-hand physical coordinate is contiguous and four
// adjacent u32 words are one uint128 transaction.  One N16 x K64 atom is
// exactly 4 x 32 u32 = 512 bytes.  This header only makes the real CuTe/PPU
// types constructible and binds their shared contract; no dispatch policy or
// collective selects either provider yet.

namespace cutlass::gemm::collective::detail::quactlize_q4_n16k64_delivery {

namespace bd =
    cutlass::gemm::collective::detail::quactlize_b_delivery;

inline constexpr int kLogicalBits = 4;
inline constexpr int kNAtom = 16;
inline constexpr int kLogicalKAtom = 64;
inline constexpr int kCodesPerU32 = 32 / kLogicalBits;
inline constexpr int kKCodesPerPhysicalRow = 16;
inline constexpr int kWordsPerNPerPhysicalRow =
    kKCodesPerPhysicalRow / kCodesPerU32;
inline constexpr int kWordsPerVector = 4;
inline constexpr int kVectorBytes = 16;
// This is the PPU0010 AIU shared-base requirement, not an inference from the
// uint128 reader.  DefaultGemm_AIU_Operand publishes align_bytes=32 for
// PPU0010, and the shipping collective records the corresponding box failure:
// a merely 16-B-aligned smem_b produced `AIU_ld TSM size out of range`
// (quactlize_mma_mixed_input.hpp, SharedStorage alignment note).  The
// Universal/cp.async endpoint needs only 16 B, so the common contract retains
// the stronger writer requirement.
inline constexpr int kSharedAlignmentBytes = 32;
inline constexpr int kReaderThreads = 32;

static_assert(kWordsPerNPerPhysicalRow == 2);
static_assert(kReaderThreads * kVectorBytes ==
              kNAtom * kLogicalKAtom * kLogicalBits / 8);

template <int TileN, int TileK>
struct PhysicalShared {
  static_assert(TileN > 0 && TileN % kNAtom == 0,
                "Q4 N16xK64 delivery requires TileN divisible by 16");
  static_assert(TileK > 0 && TileK % kLogicalKAtom == 0,
                "Q4 N16xK64 delivery requires TileK divisible by 64");

  using Element = std::uint32_t;
  static constexpr int physical_k_rows = TileK / kKCodesPerPhysicalRow;
  static constexpr int physical_n_words =
      kWordsPerNPerPhysicalRow * TileN;
  static constexpr int physical_words =
      physical_k_rows * physical_n_words;
  static constexpr std::size_t stage_bytes =
      std::size_t(physical_words) * sizeof(Element);
  static constexpr std::size_t logical_bytes =
      std::size_t(TileN) * std::size_t(TileK) * kLogicalBits / 8;

  // This is exactly the C-style [physical_k_rows][physical_n_words]
  // coordinate order used by the no-transpose AIU writer.  Keep n_word at
  // stride one; do not reuse this source layout as the converter/MMA register
  // destination layout.
  using ArrayShape = cute::Shape<cute::Int<physical_k_rows>,
                                 cute::Int<physical_n_words>>;
  using CoordinateShape = ArrayShape;
  using Layout = cute::Layout<
      CoordinateShape,
      cute::Stride<cute::Int<physical_n_words>, cute::_1>>;

  using Contract = bd::PhysicalSharedContract<
      Layout, Element, stage_bytes, kNAtom, kLogicalKAtom,
      kSharedAlignmentBytes>;

  static_assert(stage_bytes == logical_bytes,
                "Q4 physical shared bytes must equal logical NK/2");
  static_assert(physical_k_rows % (kLogicalKAtom /
                                   kKCodesPerPhysicalRow) == 0,
                "Q4 physical K rows must contain whole K64 atoms");
  static_assert(cute::cosize_v<Layout> == physical_words,
                "Q4 plain shared layout must be a compact exact cover");
};

// One opaque AIU instruction publishes one TileK x N16-word slice.  This is
// intentionally the same spelling as DeepGEMM's plain writer:
//
//   DefaultGemm_AIU_Operand<PPU0010, u32, Trans=false,
//                           K/16, 2*N, ..., Swzl=false>
//
// It lowers through the PPU0010 b32 no-transpose linear instruction.  Larger
// TileN repeats the same [TileK/16][32] cube; there is no coordinate-order
// equivalence argument hiding a different `.trans` instruction here.
template <int TileN, int TileK>
struct AiuPlainWriter {
  using Physical = PhysicalShared<TileN, TileK>;
  using Element = typename Physical::Element;
  using G2S = bd::AiuPlainG2S;
  using Shared = typename Physical::Contract;

  // These are the PPU0010 branch of DefaultGemm_AIU_Operand with
  // Block_MN=K/16, Block_K=2*N and Swzl=false, spelled locally to avoid
  // importing the entire GEMM collective dependency graph into a detail
  // provider header.
  static constexpr int cube_n_words = 32;  // min(2*N*4 B, 128 B) / 4 B
  static constexpr int cube_k_rows = Physical::physical_k_rows;
  static constexpr int cube_bits =
      cube_k_rows * cube_n_words * cute::sizeof_bits_v<Element>;
  static constexpr int cube_count = Physical::physical_n_words / cube_n_words;
  static constexpr bool swizzled = false;

  using CopyInst = cute::PPU0010_AIU_LOAD<
      cute::C<cube_bits>, Element, false, false>;
  using CopyAtom = cute::Copy_Atom<CopyInst, Element>;
  using Copy = decltype(cute::make_tiled_copy(
      CopyAtom{},
      cute::Layout<cute::Shape<cute::_1, cute::_1>,
                   cute::Stride<cute::_1, cute::_1>>{},
      cute::Layout<cute::Shape<cute::Int<cube_k_rows>,
                               cute::Int<cube_n_words>>>{}));
  using SmemLayoutAtom = cute::Layout<
      cute::Shape<cute::Int<cube_k_rows>, cute::Int<cube_n_words>>,
      cute::Stride<cute::Int<cube_n_words>, cute::_1>>;

  static_assert(cube_n_words == 32,
                "one Q4 AIU-plain cube must cover exactly N16");
  static_assert(cube_k_rows == Physical::physical_k_rows);
  static_assert(kSharedAlignmentBytes == 32,
                "Q4 plain shared contract must retain PPU0010 AIU alignment");
  static_assert(std::is_same_v<
      SmemLayoutAtom,
      cute::Layout<
          cute::Shape<cute::Int<Physical::physical_k_rows>, cute::_32>,
          cute::Stride<cute::_32, cute::_1>>>,
      "AIU-plain resident cube must be [K/16][N16*2] u32");
  static_assert(cube_bits / 8 ==
                kNAtom * TileK * kLogicalBits / 8,
                "AIU-plain cube byte count must match one logical N16 slice");
  static_assert(bd::G2STraits<G2S>::issue_scope ==
                bd::IssueScope::OpaqueSingleIssuer);
};

// The cp.async alternative decomposes the complete plain stage across the
// exact CTA thread count supplied by the eventual TiledMma.  This parameter is
// intentionally mandatory: a 32-thread TiledCopy called by a 128-thread CTA
// would make get_slice(32..127) outside the declared thread domain.  Every
// admitted CTA thread owns one uint128 vector per round, and the number of
// stage vectors must be an exact multiple of CTAThreads.
template <int TileN, int TileK, int CTAThreads>
struct CpAsyncWriter {
  using Physical = PhysicalShared<TileN, TileK>;
  using Element = typename Physical::Element;
  using G2S = bd::CpAsyncG2S;
  using Shared = typename Physical::Contract;

  static_assert(CTAThreads >= 32 && CTAThreads <= 256 &&
                    CTAThreads % 32 == 0,
                "Q4 cp.async CTA thread count must be 32..256 in whole warps");
  static constexpr int thread_count = CTAThreads;
  static constexpr int stage_vectors =
      int(Physical::stage_bytes) / kVectorBytes;
  static constexpr int vectors_per_physical_row =
      Physical::physical_n_words / kWordsPerVector;
  static_assert(stage_vectors >= thread_count &&
                    stage_vectors % thread_count == 0,
                "Q4 cp.async CTA thread count must exactly partition stage vectors");
  static constexpr int vector_rounds = stage_vectors / thread_count;
  static constexpr int thread_rows =
      thread_count <= vectors_per_physical_row
          ? 1
          : thread_count / vectors_per_physical_row;
  static constexpr int thread_n_vectors = thread_count / thread_rows;
  static_assert(thread_rows * thread_n_vectors == thread_count);
  static_assert(Physical::physical_k_rows % thread_rows == 0 &&
                    vectors_per_physical_row % thread_n_vectors == 0,
                "Q4 cp.async CTA thread layout must exactly tile physical K/N-vector axes");

  using CopyInst = cute::PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using CopyAtom = cute::Copy_Atom<CopyInst, Element>;
  using ThreadLayout = cute::Layout<
      cute::Shape<cute::Int<thread_rows>,
                  cute::Int<thread_n_vectors>>,
      cute::Stride<cute::Int<thread_n_vectors>, cute::_1>>;
  using ValueLayout = cute::Layout<
      cute::Shape<cute::_1, cute::Int<kWordsPerVector>>>;
  using Copy = decltype(cute::make_tiled_copy(
      CopyAtom{}, ThreadLayout{}, ValueLayout{}));

  static_assert(CopyAtom::NumValSrc == kWordsPerVector &&
                CopyAtom::NumValDst == kWordsPerVector,
                "one Q4 cp.async transaction must expose four u32 words");
  static_assert(bd::G2STraits<G2S>::issue_scope ==
                bd::IssueScope::ThreadPartitioned);
};

// Plain shared to raw-register delivery.  The destination is deliberately a
// uint32 register fragment.  Converter/MMA code must construct its own logical
// destination and may only then recast these words to Q4; the physical shared
// layout is not an MMA rest-stride.
template <int TileN, int TileK>
struct UniversalReader {
  using Physical = PhysicalShared<TileN, TileK>;
  using Element = typename Physical::Element;
  using S2R = bd::UniversalS2R;
  using Shared = typename Physical::Contract;

  using CopyInst = cute::UniversalCopy<cute::uint128_t>;
  using CopyAtom = cute::Copy_Atom<CopyInst, Element>;
  using Copy = decltype(cute::make_tiled_copy(
      CopyAtom{}, cute::Layout<cute::Shape<cute::_32>>{},
      cute::Layout<cute::Shape<cute::_4>>{}));

  static_assert(CopyAtom::NumValSrc == kWordsPerVector &&
                CopyAtom::NumValDst == kWordsPerVector,
                "one Q4 Universal read must expose four u32 words");
};

template <class Writer_, int TileN, int TileK>
struct Provider {
  using Physical = PhysicalShared<TileN, TileK>;
  using WriterType = Writer_;
  using ReaderType = UniversalReader<TileN, TileK>;
  using Binding = bd::BoundBDelivery<WriterType, ReaderType>;
  using Shared = typename Binding::Shared;

  static_assert(std::is_same_v<Shared, typename Physical::Contract>,
                "Q4 direct provider must preserve the physical shared ABI");
};

template <int TileN, int TileK>
using AiuPlainProvider =
    Provider<AiuPlainWriter<TileN, TileK>, TileN, TileK>;

template <int TileN, int TileK, int CTAThreads>
using CpAsyncProvider =
    Provider<CpAsyncWriter<TileN, TileK, CTAThreads>, TileN, TileK>;

// The minimum atom is the ABI authority used by offline mapping and later
// fragment proofs.  Both candidate writers must name this exact type, not two
// merely equal-looking contracts.
using AtomPhysical = PhysicalShared<kNAtom, kLogicalKAtom>;
using AtomAiuPlain = AiuPlainProvider<kNAtom, kLogicalKAtom>;
using AtomCpAsync = CpAsyncProvider<kNAtom, kLogicalKAtom, 32>;
static_assert(AtomPhysical::stage_bytes == 512);
static_assert(std::is_same_v<typename AtomAiuPlain::Shared,
                             typename AtomCpAsync::Shared>);
static_assert(AtomAiuPlain::Binding::single_issuer);
static_assert(AtomCpAsync::Binding::thread_partitioned);

}  // namespace cutlass::gemm::collective::detail::quactlize_q4_n16k64_delivery
