// Compile-only fixture. No qkg/execution header is included: this must instantiate
// both collectives using only the public include tree and pinned actlize headers.
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_exact_code_staged.hpp"

namespace ec=cutlass::gemm::collective;

template<class Collective>
__global__ void __launch_bounds__(Collective::Threads)
extension_only(typename Collective::Arguments args,float* out) {
    extern __shared__ __align__(128) unsigned char shared[];
    Collective::run(args,shared,[&](int token,int n,float value) {
        out[int64_t(token)*args.planes.n+n]=value;
    });
}

using F16=cutlass::half_t;
using BF16=cutlass::bfloat16_t;
template<int Q,class E,int M,class Delivery>
using C=ec::ExactCodeCollectivePPU<Q,E,4,4,1,M,Delivery>;

// Explicit instantiation really compiles the device body (not just its declaration).
#define PROBE(Q,E,M,D) template __global__ void extension_only<C<Q,E,M,D>>(ec::ExactCodeTileArgumentsPPU,float*);
using Direct=ec::ExactCodeRegisterDirectPPU<2>;
using Staged=ec::ExactCodeCpAsyncPPU<2>;
#if defined(EXACT_CODE_STAGED)
PROBE(12,F16,1,Staged)
PROBE(12,F16,2,Staged)
PROBE(13,F16,1,Staged)
PROBE(14,F16,1,Staged)
PROBE(8,F16,1,Staged)
PROBE(12,BF16,1,Staged)
PROBE(13,BF16,1,Staged)
using StagedFoldQ4=ec::ExactCodeCollectivePPU<12,F16,2,8,2,1,ec::ExactCodeCpAsyncPPU<3>>;
using StagedFoldQ6=ec::ExactCodeCollectivePPU<14,F16,8,2,2,2,ec::ExactCodeCpAsyncPPU<4>>;
template __global__ void extension_only<StagedFoldQ4>(ec::ExactCodeTileArgumentsPPU,float*);
template __global__ void extension_only<StagedFoldQ6>(ec::ExactCodeTileArgumentsPPU,float*);
#else
PROBE(12,F16,1,Direct)
PROBE(12,F16,2,Direct)
PROBE(12,F16,4,Direct)
PROBE(12,F16,8,Direct)
PROBE(13,F16,1,Direct)
PROBE(14,F16,1,Direct)
PROBE(8,F16,1,Direct)
PROBE(12,BF16,1,Direct)
PROBE(13,BF16,1,Direct)
using DirectFoldQ4=ec::ExactCodeCollectivePPU<12,F16,2,2,4,4,ec::ExactCodeRegisterDirectPPU<4>>;
using DirectFoldQ6=ec::ExactCodeCollectivePPU<14,F16,8,2,2,1,ec::ExactCodeRegisterDirectPPU<2>>;
template __global__ void extension_only<DirectFoldQ4>(ec::ExactCodeTileArgumentsPPU,float*);
template __global__ void extension_only<DirectFoldQ6>(ec::ExactCodeTileArgumentsPPU,float*);
#endif
