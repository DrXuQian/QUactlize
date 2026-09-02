// Generated units define SCALEFIRST_UNIT_ROWS(X), then include this file.
#include "scalefirst_internal_sweep_bench.hpp"

namespace scalefirst_internal_sweep_generated {

#if defined(SCALEFIRST_TYPE_ONLY)
#define SCALEFIRST_DEFINE(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN)           \
  static_assert(scalefirst_internal_sweep::admit_row_type<            \
      Q,A,TM,TN,TK,WM,WN,ST,BC,SCALEFIRST_SWEEP_WEIGHT_LAYOUT,AP,DN>(), #FN " type admission failed");
#else
#define SCALEFIRST_DEFINE(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN)           \
  bool FN(scalefirst_internal_sweep::DeviceInputs const& inputs,      \
          scalefirst_internal_sweep::Options const& options,          \
          scalefirst_internal_sweep::RowResult& result) {             \
    return scalefirst_internal_sweep::run_row<                        \
        Q,A,TM,TN,TK,WM,WN,ST,BC,SCALEFIRST_SWEEP_WEIGHT_LAYOUT,AP,DN>(inputs, options, result); \
  }
#endif

SCALEFIRST_UNIT_ROWS(SCALEFIRST_DEFINE)
#undef SCALEFIRST_DEFINE

}  // namespace scalefirst_internal_sweep_generated
