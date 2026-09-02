// Representative type-only closure for exhaustive canonical K-pack
// FullyQuantized discovery.  The generated shards instantiate the same
// TcRowTypes template for every admitted parent.

#include <cstdio>
#include <type_traits>

#ifndef L252_QTYPE
#error "L252_QTYPE must be one of 10,11,12,13,14"
#endif
#define PPU_PACKED_SCALE 1
#include "fully_quantized_splitk_producer_bench.hpp"

template <int Q, int Layout, int TK, int AP = 0>
using Types = fq_internal_sweep::TcRowTypes<
    Q, 0, AP ? 8 : 64, 64, TK, AP ? 8 : 64, 32, 2, 0, AP, Layout, 32>;

template <class T>
constexpr bool closes() {
  using Descriptor = typename T::Shipping::MainloopPolicy::Descriptor;
  return Descriptor::quant_mode ==
             ppu_mixed_policy::QuantMode::FinegrainedScaleZero &&
         Descriptor::kpack_transpose && Descriptor::packed_metadata &&
         Descriptor::artifact_tile_k == 0 &&
         T::Shipping::MainloopPolicy::BChunkRequest == 0 &&
         dense_splitk_parallel_ppu::MainloopUsesPackedMetadata<
             typename T::Shipping::CollectiveMainloop>::value &&
         std::is_same_v<typename T::SplitKernel::CollectiveMainloop,
                        typename T::Shipping::CollectiveMainloop> &&
         T::Shipping::SharedStorageSize > 0 &&
         T::SplitKernel::SharedStorageSize > 0;
}

#if L252_QTYPE == 10
using Subject = Types<10, 2, 128>;
using SubjectAP1 = Types<10, 2, 128, 1>;
static_assert(closes<Subject>());
static_assert(closes<SubjectAP1>() &&
              SubjectAP1::a_provider_capacity_rows == 1);
#elif L252_QTYPE == 11
using Subject = Types<11, 2, 256>;
static_assert(closes<Subject>());
static_assert(Subject::Shipping::MainloopPolicy::Descriptor::high_bits == 1);
#elif L252_QTYPE == 12
using Subject = Types<12, 1, 64>;
using SubjectAP1 = Types<12, 1, 64, 1>;
static_assert(closes<Subject>());
static_assert(closes<SubjectAP1>() &&
              SubjectAP1::a_provider_capacity_rows == 1);
#elif L252_QTYPE == 13
using Subject = Types<13, 2, 256>;
static_assert(closes<Subject>());
static_assert(Subject::Shipping::MainloopPolicy::Descriptor::high_bits == 1);
#elif L252_QTYPE == 14
using Subject = Types<14, 2, 128>;
static_assert(closes<Subject>());
static_assert(Subject::Shipping::MainloopPolicy::Descriptor::high_bits == 2);
#else
#error "L252_QTYPE must be one of 10,11,12,13,14"
#endif

int main() {
  std::printf("L252 FullyQuantized K-pack dense type PASS "
              "qtype=%d AP1=Q2+Q4-only S=1/2/4/8\n", L252_QTYPE);
}
