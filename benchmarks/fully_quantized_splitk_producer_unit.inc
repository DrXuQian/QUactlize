// Generated units define FQ_TC_UNIT_ROWS(X), then include this file.
#include "fully_quantized_splitk_producer_bench.hpp"

#ifndef FQ_TC_GENERATED_NAMESPACE
#define FQ_TC_GENERATED_NAMESPACE fq_internal_sweep_generated
#endif
namespace FQ_TC_GENERATED_NAMESPACE {

#ifndef FQ_TC_WEIGHT_LAYOUT
#define FQ_TC_WEIGHT_LAYOUT 0
#endif

#if defined(FQ_TC_TYPE_ONLY)
#define FQ_TC_DEFINE_WRAPPER(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP)           \
  static_assert(fq_internal_sweep::admit_tc_row_type<                  \
      Q,A,TM,TN,TK,WM,WN,ST,BC,AP,FQ_TC_WEIGHT_LAYOUT>(),             \
      #FN " type admission failed");
#else
#define FQ_TC_DEFINE_WRAPPER(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP)           \
  bool FN(fq_internal_sweep::DeviceInputs const& inputs,               \
          fq_internal_sweep::Options const& options,                   \
          fq_internal_sweep::RowResult& result) {                      \
    return fq_internal_sweep::run_tc_row<                              \
        Q,A,TM,TN,TK,WM,WN,ST,BC,AP,FQ_TC_WEIGHT_LAYOUT>(             \
            inputs, options, result);                                  \
  }
#endif

FQ_TC_UNIT_ROWS(FQ_TC_DEFINE_WRAPPER)
#undef FQ_TC_DEFINE_WRAPPER

}  // namespace FQ_TC_GENERATED_NAMESPACE
#undef FQ_TC_GENERATED_NAMESPACE
