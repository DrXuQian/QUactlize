#include "api.h"
#ifdef QZR_CUDA_TEST
#include <cuda_runtime.h>
using RouteStream=cudaStream_t;
static int launch_error() { return cudaGetLastError()==cudaSuccess?QZR_OK:QZR_RUNTIME; }
#else
#include <hggc_runtime.h>
using RouteStream=hggcStream_t;
#endif
#include "kernels.cuh"

#ifndef QZR_CUDA_TEST
static int launch_error() { return hggcGetLastError()==hggcSuccess?QZR_OK:QZR_RUNTIME; }
#endif

namespace quactlize::routing {
// Non-template entry definitions belong to one TU. Template producers in
// kernels.cuh are also instantiated by the optional fused prepare library.
__global__ void empty_offsets(int* out,int experts) {
    int e=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(e<=experts) out[e]=0;
}
// Prefix chunks for each expert, then experts in two warp scans. Storage
// grows with chunk count, not shared memory. All chunks are consumed, even
// beyond the native helper's per-CTA shared-memory/token limit.
__global__ void prefix_chunks(int const* histogram,int* prefix_chunks,int* offsets,int experts,int tiles) {
    __shared__ int warp_totals[32];
    int e=int(threadIdx.x),lane=e&31,warp=e/32,total=0;
    if(e<experts) for(int t=0;t<tiles;++t) {
        int64_t index=int64_t(t)*experts+e;
        int value=histogram[index];prefix_chunks[index]=total;total+=value;
    }
    int inclusive=warp_scan(total);
    if(lane==31) warp_totals[warp]=inclusive;
    __syncthreads();
    if(warp==0) {
        int value=lane<int(blockDim.x)/32?warp_totals[lane]:0;
        int prefix=warp_scan(value)-value;
        warp_totals[lane]=prefix;
    }
    __syncthreads();
    if(e<experts) {
        int begin=warp_totals[warp]+inclusive-total;
        offsets[e]=begin;
        if(e==experts-1) offsets[experts]=begin+total;
    }
}
template<int TopK,int Groups> void scalar_launch_body(qzr_call_v1 const& c,int rows,RouteStream stream) {
    warp_scalar<TopK,Groups><<<c.experts,32,0,stream>>>(c.ids,c.src_rows,c.dst_rows,c.offsets,
        c.ids_stride,c.source_stride,c.experts,c.topk,c.channels,c.inverse,rows);
}
template<int TopK> void scalar_launch(qzr_call_v1 const& c,int rows,RouteStream stream) {
    if(rows<=32) scalar_launch_body<TopK,1>(c,rows,stream);
    else if(rows<=64) scalar_launch_body<TopK,2>(c,rows,stream);
    else if(rows<=128) scalar_launch_body<TopK,4>(c,rows,stream);
    else if(rows<=256) scalar_launch_body<TopK,8>(c,rows,stream);
    else scalar_launch_body<TopK,16>(c,rows,stream);
}
template<int TopK,int Tile>
int launch(qzr_call_v1 const& c,Layout const& l,int stage) {
    auto stream=static_cast<RouteStream>(c.stream);
    if(l.rows==0) {
        empty_offsets<<<(c.experts+256)/256,256,0,stream>>>(c.offsets,c.experts);
    } else if(l.algorithm==QZR_WARP) {
        warp_compact<TopK><<<c.experts,32,size_t(c.tokens)*4,stream>>>(c,l.rows);
    } else if(l.algorithm==QZR_SCALAR) {
        scalar_launch<TopK>(c,l.rows,stream);
    } else {
        auto histogram=static_cast<int*>(c.workspace);
        auto prefix=reinterpret_cast<int*>(static_cast<uint8_t*>(c.workspace)+l.prefix);
        auto ranks=reinterpret_cast<int*>(static_cast<uint8_t*>(c.workspace)+l.ranks);
        if(stage<0 || stage==0) {
            histogram_rank<TopK,Tile><<<l.tiles,128,size_t(4*c.experts+2*Tile)*4,stream>>>(c,l.rows,histogram,ranks);
            int rc=launch_error();if(rc) return rc;
        }
        if(stage<0 || stage==1) {
            prefix_chunks<<<1,((c.experts+31)/32)*32,0,stream>>>(histogram,prefix,c.offsets,c.experts,l.tiles);
            int rc=launch_error();if(rc) return rc;
        }
        if(stage<0 || stage==2)
            scatter_maps<TopK,Tile><<<(int64_t(l.rows)+255)/256,256,0,stream>>>(c,l.rows,prefix,ranks);
    }
    return launch_error();
}
template<int TopK> int tile_launch(qzr_call_v1 const& c,Layout const& l,int stage) {
    switch(l.tile) {
        case 256:return launch<TopK,256>(c,l,stage);
        case 512:return launch<TopK,512>(c,l,stage);
        default:return launch<TopK,1024>(c,l,stage);
    }
}
int run(qzr_call_v1 const* c,qzr_config_v1 const* p,int stage) {
    if(!c || !p) return QZR_INVALID;
    Layout l;int rc=layout(*c,*p,l);if(rc) return rc;
    if(stage>=l.stages || stage<-1) return QZR_INVALID;
    rc=buffers(*c,l);if(rc) return rc;
    switch(c->topk) {
        case 1:return tile_launch<1>(*c,l,stage);
        case 2:return tile_launch<2>(*c,l,stage);
        case 4:return tile_launch<4>(*c,l,stage);
        case 6:return tile_launch<6>(*c,l,stage);
        case 8:return tile_launch<8>(*c,l,stage);
        case 16:return tile_launch<16>(*c,l,stage);
        case 32:return tile_launch<32>(*c,l,stage);
        default:return tile_launch<0>(*c,l,stage);
    }
}
}
extern "C" int quactlize_ppu_moe_route_query_v1(qzr_call_v1 const* c,qzr_config_v1 const* p,qzr_sizes_v1* out) {
    if(!c || !p || !out) return QZR_INVALID;
    quactlize::routing::Layout l;int rc=quactlize::routing::layout(*c,*p,l);
    if(!rc) *out={l.bytes,l.algorithm,l.tile,l.tiles,l.stages};
    return rc;
}
extern "C" int quactlize_ppu_moe_route_run_v1(qzr_call_v1 const* c,qzr_config_v1 const* p) {
    return quactlize::routing::run(c,p,-1);
}
#ifdef QZR_TESTING
namespace quactlize::routing {
template<int TopK> int register_launch(qzr_call_v1 const& c,int rows) {
    auto stream=static_cast<RouteStream>(c.stream);
    if(rows<=32) warp_register<TopK,1><<<c.experts,32,0,stream>>>(c,rows);
    else if(rows<=64) warp_register<TopK,2><<<c.experts,32,0,stream>>>(c,rows);
    else if(rows<=128) warp_register<TopK,4><<<c.experts,32,0,stream>>>(c,rows);
    else warp_register<TopK,8><<<c.experts,32,0,stream>>>(c,rows);
    return launch_error();
}
template<int TopK,int Groups> void ballot_launch_body(qzr_call_v1 const& c,int rows,int scalar) {
    auto stream=static_cast<RouteStream>(c.stream);
    if(scalar)
        warp_scalar<TopK,Groups><<<c.experts,32,0,stream>>>(c.ids,c.src_rows,c.dst_rows,c.offsets,
            c.ids_stride,c.source_stride,c.experts,c.topk,c.channels,c.inverse,rows);
    else warp_ballot<TopK,Groups><<<c.experts,32,0,stream>>>(c,rows);
}
template<int TopK> int ballot_launch(qzr_call_v1 const& c,int rows,int scalar) {
    if(rows<=32) ballot_launch_body<TopK,1>(c,rows,scalar);
    else if(rows<=64) ballot_launch_body<TopK,2>(c,rows,scalar);
    else if(rows<=128) ballot_launch_body<TopK,4>(c,rows,scalar);
    else if(rows<=256) ballot_launch_body<TopK,8>(c,rows,scalar);
    else ballot_launch_body<TopK,16>(c,rows,scalar);
    return launch_error();
}
}
extern "C" int qzr_test_register_v1(qzr_call_v1 const* c) {
    using namespace quactlize::routing;
    if(!c) return QZR_INVALID;
    qzr_config_v1 config{1,sizeof(config),QZR_WARP,512};
    Layout l;int rc=layout(*c,config,l);if(rc) return rc;
    if(l.rows<1 || l.rows>256) return QZR_CAPACITY;
    rc=buffers(*c,l);if(rc) return rc;
    switch(c->topk) {
        case 1:return register_launch<1>(*c,l.rows);
        case 2:return register_launch<2>(*c,l.rows);
        case 4:return register_launch<4>(*c,l.rows);
        case 6:return register_launch<6>(*c,l.rows);
        case 8:return register_launch<8>(*c,l.rows);
        case 16:return register_launch<16>(*c,l.rows);
        case 32:return register_launch<32>(*c,l.rows);
        default:return register_launch<0>(*c,l.rows);
    }
}
extern "C" int qzr_test_ballot_v1(qzr_call_v1 const* c,int scalar) {
    using namespace quactlize::routing;
    if(!c || scalar<0 || scalar>1) return QZR_INVALID;
    qzr_config_v1 config{1,sizeof(config),QZR_WARP,512};
    Layout l;int rc=layout(*c,config,l);if(rc) return rc;
    if(l.rows<1 || l.rows>512) return QZR_CAPACITY;
    rc=buffers(*c,l);if(rc) return rc;
    switch(c->topk) {
        case 1:return ballot_launch<1>(*c,l.rows,scalar);
        case 2:return ballot_launch<2>(*c,l.rows,scalar);
        case 4:return ballot_launch<4>(*c,l.rows,scalar);
        case 6:return ballot_launch<6>(*c,l.rows,scalar);
        case 8:return ballot_launch<8>(*c,l.rows,scalar);
        case 16:return ballot_launch<16>(*c,l.rows,scalar);
        case 32:return ballot_launch<32>(*c,l.rows,scalar);
        default:return ballot_launch<0>(*c,l.rows,scalar);
    }
}
extern "C" int qzr_test_device_v1(char* name,int* ordinal,int* cu,int* warp) {
#ifdef QZR_CUDA_TEST
    cudaDeviceProp p{};int rc=int(cudaGetDevice(ordinal));
    if(!rc) rc=int(cudaGetDeviceProperties(&p,*ordinal));
#else
    hggcDeviceProp p{};int rc=int(hggcGetDevice(ordinal));
    if(!rc) rc=int(hggcGetDeviceProperties(&p,*ordinal));
#endif
    if(!rc) {for(int i=0;i<256;++i) name[i]=p.name[i];*cu=p.multiProcessorCount;*warp=p.warpSize;}
    return rc;
}
// Test-only isolated launches. Every stage has disjoint input/output planes,
// so repeated isolated replays do not corrupt another stage's input.
extern "C" int qzr_test_stage_v1(qzr_call_v1 const* c,qzr_config_v1 const* p,int stage) {
    return quactlize::routing::run(c,p,stage);
}
#endif
