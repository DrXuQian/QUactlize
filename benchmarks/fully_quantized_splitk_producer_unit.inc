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
#define FQ_TC_DEFINE_12(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN)            \
  static_assert(fq_internal_sweep::admit_tc_row_type<                  \
      Q,A,TM,TN,TK,WM,WN,ST,BC,AP,FQ_TC_WEIGHT_LAYOUT,DN>(),          \
      #FN " type admission failed");
#else
#define FQ_TC_DEFINE_12(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN)            \
  bool FN(fq_internal_sweep::DeviceInputs const& inputs,               \
          fq_internal_sweep::Options const& options,                   \
          fq_internal_sweep::RowResult& result) {                      \
    return fq_internal_sweep::run_tc_row<                              \
        Q,A,TM,TN,TK,WM,WN,ST,BC,AP,FQ_TC_WEIGHT_LAYOUT,DN>(          \
            inputs, options, result);                                  \
  }
#endif

#define FQ_TC_DEFINE_11(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP)              \
  FQ_TC_DEFINE_12(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,                    \
                  FQ_TC_KPACK4_DELIVERY_N)
#define FQ_TC_SELECT(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,NAME,...) NAME
#define FQ_TC_DEFINE_WRAPPER(...)                                      \
  FQ_TC_SELECT(__VA_ARGS__,FQ_TC_DEFINE_12,FQ_TC_DEFINE_11)(__VA_ARGS__)

FQ_TC_UNIT_ROWS(FQ_TC_DEFINE_WRAPPER)
#undef FQ_TC_DEFINE_WRAPPER
#undef FQ_TC_SELECT
#undef FQ_TC_DEFINE_11
#undef FQ_TC_DEFINE_12

}  // namespace FQ_TC_GENERATED_NAMESPACE
#undef FQ_TC_GENERATED_NAMESPACE
