#pragma once

#include "cutlass/cutlass.h"

namespace quactlize::grouped_schedule {

struct ExpertSlice {
  int expert;
  int slice;
};

// Uniform grouped launches pack z = expert*S + slice.  Keep the arithmetic in
// one host/device seam so the exhaustive L125 oracle and the shipping kernel
// cannot silently assign a different weight to an expert-id bit.
CUTLASS_HOST_DEVICE constexpr ExpertSlice decode_uniform_z(int z, int splits) {
  return {z / splits, z % splits};
}

}  // namespace quactlize::grouped_schedule
