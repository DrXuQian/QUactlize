#pragma once
#include "layout.hpp"

namespace quactlize::routing {
template<int TopK> __device__ __forceinline__ int topk(qzr_call_v1 const& c) {
    return TopK?TopK:c.topk;
}
template<int TopK> __device__ __forceinline__ int read_id(qzr_call_v1 const& c,int r) {
    int k=topk<TopK>(c);
    return c.ids[int64_t(r/k)*c.ids_stride+r%k];
}
template<int TopK> __device__ __forceinline__ void store_map(qzr_call_v1 const& c,int r,int to) {
    int k=topk<TopK>(c);
    c.dst_rows[to]=r;
    if(c.inverse) c.src_rows[r]=to;
    else {
        int channel=c.channels==1?0:c.channels==k?r%k:(r%k)%c.channels;
        c.src_rows[to]=int(int64_t(r/k)*c.source_stride+channel);
    }
}
__device__ __forceinline__ int warp_sum(int v) {
#if __CUDA_ARCH__ >= 800
    return __reduce_add_sync(0xffffffff,v);
#else
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_xor_sync(0xffffffff,v,d);
    return v;
#endif
}
__device__ __forceinline__ int warp_scan(int v) {
    int lane=int(threadIdx.x)&31;
    #pragma unroll
    for(int d=1;d<32;d<<=1) {
        int other=__shfl_up_sync(0xffffffff,v,d);
        if(lane>=d) v+=other;
    }
    return v;
}
__global__ void empty_offsets(int* out,int experts) {
    int e=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(e<=experts) out[e]=0;
}

// Flatten token/slot lanes. One ballot replaces the per-token subgroup OR
// and four-token shuffle scan in the incumbent top8 helper. The stored item
// is a flat row, not packed token/slot bitfields. WARP is bounded by layout.
template<int TopK>
__global__ __launch_bounds__(32) void warp_compact(qzr_call_v1 c,int rows) {
    extern __shared__ int selected[];
    int lane=int(threadIdx.x),expert=int(blockIdx.x),count=0,lower=0;
    for(int64_t base=0;base<rows;base+=32) {
        int64_t r=base+lane;int id=r<rows?read_id<TopK>(c,int(r)):INT_MAX;
        lower+=id<expert;
        unsigned hit=__ballot_sync(0xffffffff,id==expert);
        if(id==expert) selected[count+__popc(hit&((1u<<lane)-1))]=int(r);
        count+=__popc(hit);
    }
    int begin=warp_sum(lower);
    __syncwarp();
    for(int i=lane;i<count;i+=32) store_map<TopK>(c,selected[i],begin+i);
    if(!lane) {
        c.offsets[expert]=begin;
        if(expert==c.experts-1) c.offsets[c.experts]=begin+count;
    }
}

// Stable per-chunk ranks. Warp-private histograms avoid global atomics and
// contention between tokens choosing the same expert. Rows in each warp are
// contiguous, so both ID reads and temporary rank stores are coalesced.
template<int TopK,int Tile>
__global__ __launch_bounds__(128) void histogram_rank(qzr_call_v1 c,int rows,int* histogram,int* ranks) {
    extern __shared__ int shared[];
    int* counts=shared;int* ids=counts+4*c.experts;int* local=ids+Tile;
    int tid=int(threadIdx.x),lane=tid&31,warp=tid/32;
    for(int i=tid;i<4*c.experts;i+=128) counts[i]=0;
    __syncthreads();
    int64_t first=int64_t(blockIdx.x)*Tile;
    #pragma unroll
    for(int i=0;i<Tile/128;++i) {
        int pos=warp*(Tile/4)+i*32+lane;int64_t r=first+pos;
        int id=r<rows?read_id<TopK>(c,int(r)):-1;
        // Invalid IDs are outside the contract. Guard indexing nevertheless.
        bool valid=id>=0 && id<c.experts;
        unsigned same=__match_any_sync(0xffffffff,id);
        int leader=__ffs(same)-1,base=0;
        if(valid && lane==leader) {
            base=counts[warp*c.experts+id];
            counts[warp*c.experts+id]=base+__popc(same);
        }
        base=__shfl_sync(0xffffffff,base,leader);
        ids[pos]=valid?id:-1;
        local[pos]=base+__popc(same&((1u<<lane)-1));
        __syncwarp();
    }
    __syncthreads();
    for(int e=tid;e<c.experts;e+=128) {
        int total=0;
        #pragma unroll
        for(int w=0;w<4;++w) total+=counts[w*c.experts+e];
        histogram[int64_t(blockIdx.x)*c.experts+e]=total;
    }
    #pragma unroll
    for(int i=0;i<Tile/128;++i) {
        int pos=warp*(Tile/4)+i*32+lane;int id=ids[pos];int rank=local[pos];
        #pragma unroll
        for(int w=0;w<4;++w) if(w<warp && id>=0) rank+=counts[w*c.experts+id];
        if(first+pos<rows) ranks[first+pos]=rank;
    }
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

template<int TopK,int Tile>
__global__ void scatter_maps(qzr_call_v1 c,int rows,int const* histogram,int const* ranks) {
    int64_t r=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(r>=rows) return;
    int id=read_id<TopK>(c,int(r));
    if(id<0 || id>=c.experts) return;
    int to=c.offsets[id]+histogram[(r/Tile)*c.experts+id]+ranks[r];
    store_map<TopK>(c,int(r),to);
}
} // namespace quactlize::routing
