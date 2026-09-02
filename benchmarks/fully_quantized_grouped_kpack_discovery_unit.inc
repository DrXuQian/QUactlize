// Generated units define FQ_GROUPED_KPACK_UNIT_ROWS(X).
#include "fully_quantized_grouped_kpack_discovery.hpp"

namespace fully_quantized_grouped_kpack_generated {

#define FQ_GROUPED_KPACK_DEFINE(FN,Q,L,TM,TN,TK,WM,WN,ST,DN,PERSIST)  \
  bool FN(fully_quantized_grouped_kpack::Inputs const& inputs,        \
          fully_quantized_grouped_kpack::Options const& options,      \
          fully_quantized_grouped_kpack::Result& result) {            \
    return fully_quantized_grouped_kpack::run_row<                    \
        Q,L,TM,TN,TK,WM,WN,ST,DN,(PERSIST != 0)>(inputs, options, result); \
  }

FQ_GROUPED_KPACK_UNIT_ROWS(FQ_GROUPED_KPACK_DEFINE)
#undef FQ_GROUPED_KPACK_DEFINE

}  // namespace fully_quantized_grouped_kpack_generated
