// Generated units define SCALEFIRST_GROUPED_UNIT_ROWS(X).
#include "scalefirst_grouped_kpack_discovery.hpp"

namespace scalefirst_grouped_kpack_generated {

#define SCALEFIRST_GROUPED_DEFINE(FN,Q,L,TM,TN,TK,WM,WN,ST,DN)        \
  bool FN(scalefirst_grouped_kpack::Inputs const& inputs,             \
          scalefirst_grouped_kpack::Options const& options,           \
          scalefirst_grouped_kpack::Result& result) {                 \
    return scalefirst_grouped_kpack::run_row<                         \
        Q,L,TM,TN,TK,WM,WN,ST,DN>(inputs, options, result);           \
  }

SCALEFIRST_GROUPED_UNIT_ROWS(SCALEFIRST_GROUPED_DEFINE)
#undef SCALEFIRST_GROUPED_DEFINE

}  // namespace scalefirst_grouped_kpack_generated
