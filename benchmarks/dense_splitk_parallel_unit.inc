// Generated units define DENSE_SPLITK_UNIT_CONFIGS(X), then include this file.
// One unit carries only rows with one PPU_B_CHUNK value; the generated wrapper
// statically binds that value back to the committed table row.
#include "dense_splitk_parallel_bench.hpp"

namespace dense_splitk_sweep_generated {

#define DENSE_SPLITK_DEFINE_WRAPPER(FN,TM,TN,TK,WM,WN,ST,BC)             \
  bool FN(dense_splitk_sweep::DeviceInputs const& inputs,                \
          dense_splitk_sweep::Options const& options,                    \
          dense_splitk_sweep::RowResult& result) {                       \
    return dense_splitk_sweep::run_row<TM,TN,TK,WM,WN,ST,BC>(            \
        inputs, options, result);                                         \
  }

DENSE_SPLITK_UNIT_CONFIGS(DENSE_SPLITK_DEFINE_WRAPPER)
#undef DENSE_SPLITK_DEFINE_WRAPPER

}  // namespace dense_splitk_sweep_generated
