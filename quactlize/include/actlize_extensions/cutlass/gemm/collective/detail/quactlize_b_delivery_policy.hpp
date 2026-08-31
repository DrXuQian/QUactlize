#pragma once

#include <cstddef>
#include <type_traits>

// Compile-time vocabulary for selecting the physical B-delivery chain.
//
// This header deliberately does not include CuTe, CUTLASS copy atoms, or PPU
// instruction definitions.  It describes only the contract between the two
// independently replaceable halves of delivery:
//
//   global memory --G2S--> shared representation --S2R--> register fragment
//
// A later builder integration supplies the concrete GmemTiledCopy,
// SmemLayoutAtom, and S2R copy types.  Keeping those hardware types out of this
// policy makes an invalid writer/reader pairing fail before either branch is
// instantiated and keeps the production type unchanged until it is explicitly
// wired to ProductionBDelivery below.

namespace cutlass::gemm::collective::detail::quactlize_b_delivery {

enum class SharedEncoding {
  Invalid,
  AiuSwizzled,
  Plain,
};

// AIU writes are opaque, logically single-issuer operations.  A cp.async
// tiled copy is instead partitioned over its physical publishing threads.
// This distinction must eventually select the issue function as well as the
// copy atom; using copy_aiu() for a thread-partitioned copy would publish only
// a fraction of the shared tile.
enum class IssueScope {
  Invalid,
  OpaqueSingleIssuer,
  ThreadPartitioned,
};

// The encoding is the common ABI between a writer and a reader.  Stage shape,
// byte count, base alignment, and concrete address map remain properties of
// the eventual hardware provider and must be proved equal there.
template <SharedEncoding Encoding_>
struct SharedContract {
  static constexpr SharedEncoding encoding = Encoding_;
};

// The concrete shared-memory ABI bound by a builder after TileN/TileK and the
// physical transport have been selected.  Layout_ and Element_ remain opaque
// here on purpose: this header can validate a provider before including any
// CuTe or PPU instruction definition, while a production instantiation can
// still pass the exact CuTe layout and transport element types.
//
// BytesPerStage is the complete resident B byte count for one pipeline stage,
// not the allocation across all stages.  NAtom and LogicalKAtom are artifact
// facts: tactic TN/TK may repeat them, but may not redefine them.  Alignment
// is part of the contract because a writer that admits an address which the
// reader's vector operation cannot consume is not a compatible provider.
template <class Layout_, class Element_, std::size_t BytesPerStage_,
          int NAtom_, int LogicalKAtom_, int AlignmentBytes_>
struct PhysicalSharedContract {
  static_assert(BytesPerStage_ > 0,
                "B delivery requires a positive per-stage byte count");
  static_assert(NAtom_ > 0 && LogicalKAtom_ > 0,
                "B delivery requires positive N/K atoms");
  static_assert(AlignmentBytes_ > 0,
                "B delivery requires a positive shared alignment");
  static_assert((AlignmentBytes_ & (AlignmentBytes_ - 1)) == 0,
                "B delivery shared alignment must be a power of two");
  static_assert(BytesPerStage_ % std::size_t(AlignmentBytes_) == 0,
                "B delivery stage stride must preserve shared alignment");

  using Layout = Layout_;
  using Element = Element_;
  static constexpr std::size_t bytes_per_stage = BytesPerStage_;
  static constexpr int n_atom = NAtom_;
  static constexpr int logical_k_atom = LogicalKAtom_;
  static constexpr int alignment_bytes = AlignmentBytes_;
};

// Gmem-to-smem tags.
struct LegacyAiuSwizzleG2S {};
struct AiuPlainG2S {};
struct CpAsyncG2S {};

// Smem-to-register tags.
struct TsmSwizzleS2R {};
struct UniversalS2R {};

template <class Tag>
struct G2STraits {
  static constexpr bool recognized = false;
  using Shared = SharedContract<SharedEncoding::Invalid>;
  static constexpr IssueScope issue_scope = IssueScope::Invalid;
  static constexpr bool opaque_instruction = false;
};

template <>
struct G2STraits<LegacyAiuSwizzleG2S> {
  static constexpr bool recognized = true;
  using Shared = SharedContract<SharedEncoding::AiuSwizzled>;
  static constexpr IssueScope issue_scope = IssueScope::OpaqueSingleIssuer;
  static constexpr bool opaque_instruction = true;
};

template <>
struct G2STraits<AiuPlainG2S> {
  static constexpr bool recognized = true;
  using Shared = SharedContract<SharedEncoding::Plain>;
  static constexpr IssueScope issue_scope = IssueScope::OpaqueSingleIssuer;
  static constexpr bool opaque_instruction = true;
};

template <>
struct G2STraits<CpAsyncG2S> {
  static constexpr bool recognized = true;
  using Shared = SharedContract<SharedEncoding::Plain>;
  static constexpr IssueScope issue_scope = IssueScope::ThreadPartitioned;
  static constexpr bool opaque_instruction = false;
};

template <class Tag>
struct S2RTraits {
  static constexpr bool recognized = false;
  using Shared = SharedContract<SharedEncoding::Invalid>;
  static constexpr bool opaque_instruction = false;
};

template <>
struct S2RTraits<TsmSwizzleS2R> {
  static constexpr bool recognized = true;
  using Shared = SharedContract<SharedEncoding::AiuSwizzled>;
  static constexpr bool opaque_instruction = true;
};

template <>
struct S2RTraits<UniversalS2R> {
  static constexpr bool recognized = true;
  using Shared = SharedContract<SharedEncoding::Plain>;
  static constexpr bool opaque_instruction = false;
};

template <class G2STag, class S2RTag>
struct IsCompatible
    : std::bool_constant<
          G2STraits<G2STag>::recognized &&
          S2RTraits<S2RTag>::recognized &&
          std::is_same_v<typename G2STraits<G2STag>::Shared,
                         typename S2RTraits<S2RTag>::Shared>> {};

template <class G2STag, class S2RTag>
inline constexpr bool is_compatible_v = IsCompatible<G2STag, S2RTag>::value;

// A legal delivery is only a composition and a set of compile-time facts.  It
// intentionally has no runtime state and no hardware copy type.  The builder
// can branch on issue_scope without adding a runtime branch to the hot path.
template <class G2STag, class S2RTag>
struct ComposedBDelivery {
  static_assert(G2STraits<G2STag>::recognized,
                "unknown quactlize B gmem-to-smem delivery tag");
  static_assert(S2RTraits<S2RTag>::recognized,
                "unknown quactlize B smem-to-register delivery tag");
  static_assert(is_compatible_v<G2STag, S2RTag>,
                "B writer and reader must name the same shared encoding");

  using G2S = G2STag;
  using S2R = S2RTag;
  using Shared = typename G2STraits<G2S>::Shared;

  static constexpr SharedEncoding shared_encoding = Shared::encoding;
  static constexpr IssueScope issue_scope = G2STraits<G2S>::issue_scope;
  static constexpr bool g2s_is_opaque = G2STraits<G2S>::opaque_instruction;
  static constexpr bool s2r_is_opaque = S2RTraits<S2R>::opaque_instruction;
  static constexpr bool single_issuer =
      issue_scope == IssueScope::OpaqueSingleIssuer;
  static constexpr bool thread_partitioned =
      issue_scope == IssueScope::ThreadPartitioned;
};

// Bind concrete writer and reader implementations to one physical shared
// contract.  Writer_ names `G2S` and `Shared`; Reader_ names `S2R` and
// `Shared`.  Hardware types such as GmemTiledCopy or SmemCopyAtom may be
// additional members of those implementations; the scaffold deliberately
// does not prescribe them.
//
// Check every physical field separately instead of only comparing the whole
// type.  Besides clearer diagnostics, this prevents two independently-spelled
// contracts with the same values from being rejected merely because their
// wrapper types differ.
template <class Writer_, class Reader_>
struct BoundBDelivery {
  using Writer = Writer_;
  using Reader = Reader_;
  using Tags = ComposedBDelivery<typename Writer::G2S, typename Reader::S2R>;
  using WriterShared = typename Writer::Shared;
  using ReaderShared = typename Reader::Shared;

  static_assert(std::is_same_v<typename WriterShared::Layout,
                               typename ReaderShared::Layout>,
                "QUACTLIZE_B_DELIVERY_LAYOUT_MISMATCH");
  static_assert(std::is_same_v<typename WriterShared::Element,
                               typename ReaderShared::Element>,
                "QUACTLIZE_B_DELIVERY_ELEMENT_MISMATCH");
  static_assert(WriterShared::bytes_per_stage ==
                    ReaderShared::bytes_per_stage,
                "QUACTLIZE_B_DELIVERY_STAGE_BYTES_MISMATCH");
  static_assert(WriterShared::n_atom == ReaderShared::n_atom,
                "QUACTLIZE_B_DELIVERY_N_ATOM_MISMATCH");
  static_assert(WriterShared::logical_k_atom ==
                    ReaderShared::logical_k_atom,
                "QUACTLIZE_B_DELIVERY_K_ATOM_MISMATCH");
  static_assert(WriterShared::alignment_bytes ==
                    ReaderShared::alignment_bytes,
                "QUACTLIZE_B_DELIVERY_ALIGNMENT_MISMATCH");

  using Shared = WriterShared;
  using G2S = typename Tags::G2S;
  using S2R = typename Tags::S2R;
  static constexpr IssueScope issue_scope = Tags::issue_scope;
  static constexpr bool single_issuer = Tags::single_issuer;
  static constexpr bool thread_partitioned = Tags::thread_partitioned;
};

// The three admitted physical chains.  Only the first is the production
// default; the other two are named counterfactual providers and are inert until
// a builder explicitly selects and supplies their concrete hardware types.
using LegacyAiuSwizzleDelivery =
    ComposedBDelivery<LegacyAiuSwizzleG2S, TsmSwizzleS2R>;
using AiuPlainUniversalDelivery =
    ComposedBDelivery<AiuPlainG2S, UniversalS2R>;
using CpAsyncUniversalDelivery =
    ComposedBDelivery<CpAsyncG2S, UniversalS2R>;

using ProductionDefaultG2S = LegacyAiuSwizzleG2S;
using ProductionDefaultS2R = TsmSwizzleS2R;
using ProductionBDelivery = LegacyAiuSwizzleDelivery;

// Local, dependency-free contract witnesses.  They protect the intended
// denominator if a future edit changes one side's shared encoding.
static_assert(is_compatible_v<LegacyAiuSwizzleG2S, TsmSwizzleS2R>);
static_assert(is_compatible_v<AiuPlainG2S, UniversalS2R>);
static_assert(is_compatible_v<CpAsyncG2S, UniversalS2R>);
static_assert(!is_compatible_v<LegacyAiuSwizzleG2S, UniversalS2R>);
static_assert(!is_compatible_v<AiuPlainG2S, TsmSwizzleS2R>);
static_assert(!is_compatible_v<CpAsyncG2S, TsmSwizzleS2R>);
static_assert(ProductionBDelivery::single_issuer);
static_assert(ProductionBDelivery::shared_encoding ==
              SharedEncoding::AiuSwizzled);

}  // namespace cutlass::gemm::collective::detail::quactlize_b_delivery
