#include <hggc_runtime.h>
#include "cutlass/numeric_types.h"
namespace q4_decode_io {
using Half=cutlass::half_t;
template<bool Input> __global__ void convert(void const* source,void* destination,int rows,int columns,int stride) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*columns;i+=gridDim.x*blockDim.x) {
        int row=i/columns,col=i%columns;
        if constexpr(Input) static_cast<Half*>(destination)[i]=Half(static_cast<float const*>(source)[row*stride+col]);
        else static_cast<float*>(destination)[row*stride+col]=float(static_cast<Half const*>(source)[i]);
    }
}
}
extern "C" int q4_decode_dense_cast(int input,void const* source,void* destination,int rows,int columns,int stride,void* stream) {
    if((input!=0 && input!=1) || !source || !destination || rows<1 || rows>8 || columns<1 ||
       stride<columns || int64_t(rows)*columns>INT32_MAX) return 2;
    if(hggcGetLastError()!=hggcSuccess) return 3;
    int grid=(rows*columns+255)/256;
    if(input) q4_decode_io::convert<true><<<grid,256,0,static_cast<hggcStream_t>(stream)>>>(source,destination,rows,columns,stride);
    else q4_decode_io::convert<false><<<grid,256,0,static_cast<hggcStream_t>(stream)>>>(source,destination,rows,columns,stride);
    return hggcGetLastError()==hggcSuccess ? 0 : 3;
}
