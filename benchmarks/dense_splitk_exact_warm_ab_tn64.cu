/***************************************************************************************************
 * Exact warm-resident reshape A/B for the committed TN64 row.
 * Kept in its own TU so the diagnostic does not multiply the 201-row generated sweep bodies.
 **************************************************************************************************/

#include "dense_splitk_parallel_bench.hpp"

namespace dense_splitk_sweep_exact {

bool tn64(dense_splitk_sweep::DeviceInputs const& reshape,
          dense_splitk_sweep::DeviceInputs const& internal, int iterations,
          dense_splitk_sweep::ExactWarmAbResult& result) {
  return dense_splitk_sweep::run_exact_warm_ab<8,64,128,8,16,2,0>(
      reshape, internal, iterations, result);
}

}  // namespace dense_splitk_sweep_exact
