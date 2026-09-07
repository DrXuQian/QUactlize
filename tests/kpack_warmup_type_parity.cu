// Compile-only equality to the types whose timings seeded the policy.
// Benchmark dependencies belong to this test, not to runtime modules.
#include "kernel_types.cuh"
#include "fully_quantized_splitk_producer_bench.hpp"
#include "scalefirst_internal_sweep_bench.hpp"
#include "fully_quantized_grouped_kpack_discovery.hpp"
#include "scalefirst_grouped_kpack_discovery.hpp"

template<int Q,int TK,int TM> constexpr bool same_types() {
  constexpr int L=Q==12?1:2;
  using Dense=quactlize::runtime::DenseTypes<Q,TM,64,TK,TM,16,2,0,16>;
#if PPU_PACKED_SCALE
  using FQ=fq_internal_sweep::TcRowTypes<Q,0,TM,64,TK,TM,16,2,0,0,L,16>;
  static_assert(std::is_same_v<typename Dense::Shipping,typename FQ::Shipping>);
#else
  using SF=scalefirst_internal_sweep::RowTypes<Q,0,TM,64,TK,TM,16,2,0,L,0,16>;
  static_assert(std::is_same_v<typename Dense::Shipping,typename SF::Shipping>);
  static_assert(std::is_same_v<typename Dense::PersistentKernel,typename SF::PersistentKernel>);
#endif
  using GP=quactlize::runtime::GroupedTypes<Q,TM,64,TK,TM,16,2,16,true>;
  using GN=quactlize::runtime::GroupedTypes<Q,TM,64,TK,TM,16,2,16,false>;
#if PPU_PACKED_SCALE
  using OldP=fully_quantized_grouped_kpack::RowTypes<Q,L,TM,64,TK,TM,16,2,16,true>;
  using OldN=fully_quantized_grouped_kpack::RowTypes<Q,L,TM,64,TK,TM,16,2,16,false>;
  static_assert(std::is_same_v<typename GP::Kernel,typename OldP::Kernel>);
  static_assert(std::is_same_v<typename GN::Kernel,typename OldN::Kernel>);
#else
  using Old=scalefirst_grouped_kpack::RowTypes<Q,L,TM,64,TK,TM,16,2,16>;
  static_assert(std::is_same_v<typename GP::Kernel,typename Old::PersistentKernel>);
  static_assert(std::is_same_v<typename GN::Kernel,typename Old::Kernel>);
#endif
  return true;
}
constexpr int tile_k=QK_PARITY_Q==12?64:((QK_PARITY_Q==10 || QK_PARITY_Q==14)?128:256);
static_assert(same_types<QK_PARITY_Q,tile_k,8>() && same_types<QK_PARITY_Q,tile_k,16>());
