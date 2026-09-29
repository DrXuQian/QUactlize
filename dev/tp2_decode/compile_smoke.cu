// Compile-only coverage of parameterized bodies, not a device correctness gate.
#include <hggc_runtime.h>
#include "quactlize/execution/simt_q8_vector.cuh"

namespace quactlize::execution::simt {
// Instantiate real generic launchers, not just a proposed reader body. These
// cover scalar/aligned row reducers and the precision/shape fallback branches.
template int launch_v2<10,3,4,4,4>(qkg_simt_call_v2 const&,int);
template int launch_v2<11,3,4,4,4>(qkg_simt_call_v2 const&,int);
template int launch_v2<12,3,4,4,4>(qkg_simt_call_v2 const&,int);
template int launch_v2<13,3,4,2,8>(qkg_simt_call_v2 const&,int);
template int launch_v2<14,3,4,4,4>(qkg_simt_call_v2 const&,int);
template __global__ void register_reuse_model<13,1,3,4,2,8,1,3,3072,512>(qkg_call_v1);
namespace q8_vector {
template int launch_v2<8,5,8,4,4>(qkg_simt_call_v2 const&,int);
template int launch_v2<8,5,4,2,4>(qkg_simt_call_v2 const&,int);
template int launch_v2<8,5,4,8,4>(qkg_simt_call_v2 const&,int);
template __global__ void kernel_model<1,1,1,8,4,4,true,3072,4096,4>(qkg_call_v1);
template __global__ void kernel_s1<1,1,1,4,8,4,true>(qkg_call_v1);
}
}
