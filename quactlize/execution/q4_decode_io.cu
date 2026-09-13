#include <hggc_runtime.h>
#include "cutlass/numeric_types.h"
#include "q4_decode.h"
#include "validation.hpp"
#include "../runtime/indexed_rows.hpp"
#include "../integrations/llama/indexed.h"

namespace quactlize::execution::decode_io {
using Half=cutlass::half_t;
template<bool Input> __global__ void convert(void const* source,void* destination,int rows,int columns,int stride) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*columns;i+=gridDim.x*blockDim.x) {
        int row=i/columns,col=i%columns;
        if constexpr(Input) static_cast<Half*>(destination)[i]=Half(static_cast<float const*>(source)[row*stride+col]);
        else static_cast<float*>(destination)[row*stride+col]=float(static_cast<Half const*>(source)[i]);
    }
}
constexpr int Capacity=64;
__global__ void prepare(qk_llama_indexed_v1 io,Half* a,int* offsets,int k,int experts) {
    __shared__ int ids[Capacity], ranks[Capacity];
    int tid=threadIdx.x, m=io.tokens*io.topk;
    if(tid<m) ids[tid]=io.ids[int64_t(tid/io.topk)*io.ids_stride+tid%io.topk];
    __syncthreads();
    bool invalid=false;
    if(tid<m) {
        invalid=ids[tid]<0 || ids[tid]>=experts;
        for(int j=tid-tid%io.topk;j<tid;++j) invalid|=ids[j]==ids[tid];
        ranks[tid]=quactlize::runtime::ranked_row(ids,m,tid);
    }
    bool valid=__syncthreads_or(invalid)==0;
    if(blockIdx.x==0) {
        for(int e=tid;e<=experts;e+=blockDim.x)
            offsets[e]=valid ? quactlize::runtime::expert_begin(ids,m,e) : 0;
        if(tid<m) io.row_ids[valid?ranks[tid]:tid]=valid?tid:-1;
    }
    if(!valid) return;
    int r=blockIdx.x;
    int64_t from=int64_t(r/io.topk)*io.a_token_stride+(r%io.topk%io.channels)*io.a_row_stride;
    for(int col=tid;col<k;col+=blockDim.x) a[int64_t(ranks[r])*k+col]=Half(io.a[from+col]);
}
__global__ void finish(qk_llama_indexed_v1 io,Half const* compact,int n) {
    int row=blockIdx.y, to=io.row_ids[row];
    for(int col=blockIdx.x*blockDim.x+threadIdx.x;col<n;col+=gridDim.x*blockDim.x) {
        float value=to>=0 ? float(compact[int64_t(row)*n+col]) : __int_as_float(0x7fc00000);
        io.output[int64_t(to>=0?to:row)*io.out_row_stride+col]=value;
    }
}
bool disjoint(uintptr_t const* p,uint64_t const* b,int count,int first_write) {
    for(int i=0;i<count;++i) {
        if(!span(p[i],b[i])) return false;
        for(int j=0;j<i;++j) if(i>=first_write && overlap(p[i],b[i],p[j],b[j])) return false;
    }
    return true;
}
int indexed(qkg_call_v1 const* c,qk_llama_indexed_v1& io,int32_t* rows) {
    if(!c) return QKG_INVALID;
    auto arr=ppu_arrangements::q4_kpack4_transpose_v1();
    qkg_config_v1 f{1,sizeof(f),16,4,1};qkg_sizes_v1 sizes{};
    int rc=query(*c,f,&arr,sizes);
    if(rc) return rc;
    if(c->qtype!=12 || c->mode!=QKG_INDEXED || c->input_type!=QKG_F32 || c->experts!=256 ||
       c->topk!=8 || c->rows<=32 || c->rows>64 || (c->channels!=1 && c->channels!=8)) return QKG_SHAPE;
    if(!c->ids || !c->a || !c->output || !rows ||
       ((uintptr_t(c->ids)|uintptr_t(c->a)|uintptr_t(c->output)|uintptr_t(rows))&3)) return QKG_INVALID;
    io={1,sizeof(io),c->rows/8,8,c->channels,0,c->ids_stride,c->a_row_stride,c->a_token_stride,
        c->out_row_stride,c->ids,static_cast<float const*>(c->a),c->output,rows};
    return QKG_OK;
}
}

extern "C" int quactlize_kpack_q4_decode_cast_v1(int input,void const* source,void* destination,
    int rows,int columns,int stride,void* stream) {
    using namespace quactlize::execution::decode_io;
    if((input!=0 && input!=1) || rows<1 || rows>8 || columns<1 || stride<columns ||
       int64_t(rows)*stride>INT32_MAX || int64_t(rows)*columns>INT32_MAX-255) return QKG_INVALID;
    uint64_t compact=uint64_t(rows)*columns*2, wide=(uint64_t(rows-1)*stride+columns)*4;
    uintptr_t p[]={uintptr_t(source),uintptr_t(destination)};
    uint64_t b[]={input?wide:compact,input?compact:wide};
    if((p[0]&(input?3:1)) || (p[1]&(input?1:3)) || !disjoint(p,b,2,1)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    int grid=(rows*columns+255)/256;
    if(input) convert<true><<<grid,256,0,static_cast<hggcStream_t>(stream)>>>(source,destination,rows,columns,stride);
    else convert<false><<<grid,256,0,static_cast<hggcStream_t>(stream)>>>(source,destination,rows,columns,stride);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
extern "C" int quactlize_kpack_q4_decode_indexed_prepare_v1(qkg_call_v1 const* c,
    void* compact,int32_t* offsets,int32_t* rows) {
    using namespace quactlize::execution::decode_io;
    qk_llama_indexed_v1 io{};int rc=indexed(c,io,rows);if(rc) return rc;
    uintptr_t p[]={uintptr_t(c->a),uintptr_t(c->ids),uintptr_t(compact),uintptr_t(offsets),uintptr_t(rows)};
    uint64_t b[]={(uint64_t(io.tokens-1)*c->a_token_stride+uint64_t(c->channels-1)*c->a_row_stride+c->k)*4,
        (uint64_t(io.tokens-1)*c->ids_stride+8)*4,uint64_t(c->rows)*c->k*2,257*4,uint64_t(c->rows)*4};
    if((p[2]&1) || (p[3]&3) || !disjoint(p,b,5,2)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    prepare<<<c->rows,256,0,static_cast<hggcStream_t>(c->stream)>>>(io,static_cast<Half*>(compact),offsets,c->k,256);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
extern "C" int quactlize_kpack_q4_decode_indexed_finish_v1(qkg_call_v1 const* c,
    void const* compact,int32_t const* rows) {
    using namespace quactlize::execution::decode_io;
    qk_llama_indexed_v1 io{};int rc=indexed(c,io,const_cast<int32_t*>(rows));if(rc) return rc;
    uintptr_t p[]={uintptr_t(compact),uintptr_t(rows),uintptr_t(c->output)};
    uint64_t b[]={uint64_t(c->rows)*c->n*2,uint64_t(c->rows)*4,(uint64_t(c->rows-1)*c->out_row_stride+c->n)*4};
    if((p[0]&1) || c->n>INT32_MAX-255 || !disjoint(p,b,3,2)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    finish<<<dim3((c->n+255)/256,c->rows),256,0,static_cast<hggcStream_t>(c->stream)>>>(io,static_cast<Half const*>(compact),c->n);
    return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
