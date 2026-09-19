// Compile-only coverage of parameterized bodies, not a device correctness gate.
#include <hggc_runtime.h>
#include "quactlize/execution/simt_q8_vector.cuh"

namespace quactlize::execution::simt {
template __global__ void register_reuse_model<13,1,3,4,2,8,1,3,3072,512>(qkg_call_v1);
namespace q8_vector {
template __global__ void kernel_model<1,1,1,8,4,4,true,3072,4096,4>(qkg_call_v1);
template __global__ void kernel_s1<1,1,1,4,8,4,true>(qkg_call_v1);
}
}
