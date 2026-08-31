// L238 -- compile-time oracle for the production B-delivery provider scaffold.
//
// This includes the real dependency-free policy header.  The hardware copy
// types stay mocked here, but the composition, shared ABI fields, admitted
// writer/reader pairs, and compile-fail diagnostics are the same templates the
// production builder consumes.

#include <cstdio>
#include <type_traits>

#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_b_delivery_policy.hpp"

namespace {

namespace bd =
    cutlass::gemm::collective::detail::quactlize_b_delivery;

template <int Id_>
struct LayoutTag {
  static constexpr int Id = Id_;
};

struct HalfTransport {};
struct OtherTransport {};
using LegacyLayout = LayoutTag<0x2381>;
using PlainLayout = LayoutTag<0x2382>;

using LegacyShared = bd::PhysicalSharedContract<
    LegacyLayout, HalfTransport, 8192, 16, 64, 16>;
using PlainShared = bd::PhysicalSharedContract<
    PlainLayout, HalfTransport, 8192, 16, 64, 16>;

template <class Tag_, class Shared_, char const* (*Name_)()>
struct MockWriter {
  using G2S = Tag_;
  using Shared = Shared_;
  static constexpr char const* name() { return Name_(); }
};

template <class Tag_, class Shared_, char const* (*Name_)()>
struct MockReader {
  using S2R = Tag_;
  using Shared = Shared_;
  static constexpr char const* name() { return Name_(); }
};

constexpr char const* legacy_name() { return "legacy-swzl"; }
constexpr char const* aiu_plain_name() { return "aiu-plain"; }
constexpr char const* cp_async_name() { return "cp-async"; }
constexpr char const* universal_name() { return "universal"; }

using LegacyWriter = MockWriter<bd::LegacyAiuSwizzleG2S,
                                LegacyShared, legacy_name>;
using LegacyReader = MockReader<bd::TsmSwizzleS2R,
                                LegacyShared, legacy_name>;
using AiuPlainWriter = MockWriter<bd::AiuPlainG2S,
                                  PlainShared, aiu_plain_name>;
using CpAsyncWriter = MockWriter<bd::CpAsyncG2S,
                                 PlainShared, cp_async_name>;
using UniversalReader = MockReader<bd::UniversalS2R,
                                   PlainShared, universal_name>;

#if defined(L238_PLANT_LAYOUT_MISMATCH)
using NegativeShared = bd::PhysicalSharedContract<
    LegacyLayout, HalfTransport, 8192, 16, 64, 16>;
#elif defined(L238_PLANT_ELEMENT_MISMATCH)
using NegativeShared = bd::PhysicalSharedContract<
    PlainLayout, OtherTransport, 8192, 16, 64, 16>;
#elif defined(L238_PLANT_SHARED_BYTES_MISMATCH)
using NegativeShared = bd::PhysicalSharedContract<
    PlainLayout, HalfTransport, 4096, 16, 64, 16>;
#elif defined(L238_PLANT_N_ATOM_MISMATCH)
using NegativeShared = bd::PhysicalSharedContract<
    PlainLayout, HalfTransport, 8192, 32, 64, 16>;
#elif defined(L238_PLANT_K_ATOM_MISMATCH)
using NegativeShared = bd::PhysicalSharedContract<
    PlainLayout, HalfTransport, 8192, 16, 128, 16>;
#else
using NegativeShared = PlainShared;
#endif

template <class Writer, class Reader>
void print_case() {
  using Case = bd::BoundBDelivery<Writer, Reader>;
  static_assert(Case::Shared::bytes_per_stage == 8192);
  static_assert(Case::Shared::n_atom == 16);
  static_assert(Case::Shared::logical_k_atom == 64);
  std::printf(
      "L238 provider-scaffold writer=%s reader=%s layout=0x%04x "
      "bytes=%zu natom=%d katom=%d issue=%s result=PASS\n",
      Writer::name(), Reader::name(), Case::Shared::Layout::Id,
      Case::Shared::bytes_per_stage, Case::Shared::n_atom,
      Case::Shared::logical_k_atom,
      Case::single_issuer ? "single" : "threads");
}

}  // namespace

int main() {
#if defined(L238_PLANT_TAG_MISMATCH)
  using NegativeReader = MockReader<bd::UniversalS2R,
                                    LegacyShared, universal_name>;
  using Negative = bd::BoundBDelivery<LegacyWriter, NegativeReader>;
  static_assert(Negative::Shared::bytes_per_stage > 0);
#elif defined(L238_PLANT_LAYOUT_MISMATCH) || \
      defined(L238_PLANT_ELEMENT_MISMATCH) || \
      defined(L238_PLANT_SHARED_BYTES_MISMATCH) || \
      defined(L238_PLANT_N_ATOM_MISMATCH) || \
      defined(L238_PLANT_K_ATOM_MISMATCH)
  using NegativeReader = MockReader<bd::UniversalS2R,
                                    NegativeShared, universal_name>;
  using Negative = bd::BoundBDelivery<AiuPlainWriter, NegativeReader>;
  static_assert(Negative::Shared::bytes_per_stage > 0);
#else
  using Legacy = bd::BoundBDelivery<LegacyWriter, LegacyReader>;
  using AiuPlain = bd::BoundBDelivery<AiuPlainWriter, UniversalReader>;
  using CpAsync = bd::BoundBDelivery<CpAsyncWriter, UniversalReader>;
  static_assert(Legacy::single_issuer && AiuPlain::single_issuer);
  static_assert(CpAsync::thread_partitioned);
  static_assert(std::is_same_v<typename AiuPlain::Shared,
                               typename CpAsync::Shared>);

  print_case<LegacyWriter, LegacyReader>();
  print_case<AiuPlainWriter, UniversalReader>();
  print_case<CpAsyncWriter, UniversalReader>();
  std::printf("L238 PROVIDER_SCAFFOLD_COMPILE_ORACLE PASS "
              "providers=3 shared-contracts=2 reds=6\n");
#endif
  return 0;
}
