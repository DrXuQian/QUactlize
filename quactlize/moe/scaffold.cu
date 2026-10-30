#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include "selection.hpp"
#include "../routing/kernels.cuh"

namespace quactlize::routing {
// Internal stage launcher defined by route.cu; production helpers must not
// depend on a QZR_TESTING entry to publish their maps.
int run(qzr_call_v1 const*,qzr_config_v1 const*,int);
}

namespace quactlize::moe {
static int launch_status() { return hggcGetLastError()==hggcSuccess?QZR_OK:QZR_RUNTIME; }
__device__ float load(void const* ptr,int64_t index,int type) {
    if(type==QZM_F32) return static_cast<float const*>(ptr)[index];
    if(type==QZM_F16) return __half2float(static_cast<__half const*>(ptr)[index]);
    return __bfloat162float(static_cast<__ppu_bfloat16 const*>(ptr)[index]);
}
__device__ void store(void* ptr,int64_t index,int type,float v) {
    if(type==QZM_F32) static_cast<float*>(ptr)[index]=v;
    else if(type==QZM_F16) static_cast<__half*>(ptr)[index]=__float2half_rn(v);
    else static_cast<__ppu_bfloat16*>(ptr)[index]=__float2bfloat16_rn(v);
}
__device__ float rounded(float v,int type) {
    if(type==QZM_F16) return __half2float(__float2half_rn(v));
    if(type==QZM_BF16) return __bfloat162float(__float2bfloat16_rn(v));
    return v;
}
__device__ float projection(qzm_value_v1 const& v,int row,int col) {
    int64_t at=int64_t(row)*v.row_stride+col;
    float result;
    if(v.state==QZM_PARTIALS) {
        result=0.f;
        for(int s=0;s<v.partial_count;++s)
            result=__fadd_rn(result,static_cast<float const*>(v.data)[int64_t(s)*v.partial_stride+at]);
    } else result=load(v.data,at,v.dtype);
    return v.state==QZM_ACTIVATED?result:rounded(result,v.round_type);
}

__global__ void prepare_empty(qzm_prepare_v1 c) {
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<=c.route.experts) c.route.offsets[i]=0;
    if(i<c.route.experts) c.expert_rows[i]=0;
    if(!i) *c.active_count=0;
}

// Counts and offsets require completion of every histogram CTA. Keep this
// separate launch; no inter-CTA spin/wait or assumed launch order is used.
template<bool Histogram>
__global__ void prepare_prefix(qzm_prepare_v1 c,int tiles,int const* histogram,int* chunk_prefix) {
    __shared__ int warp_totals[32];
    int e=int(threadIdx.x),lane=e&31,warp=e/32,total=0;
    if(e<c.route.experts) {
        if constexpr(Histogram) {
            for(int t=0;t<tiles;++t) {
                int64_t at=int64_t(t)*c.route.experts+e;
                int count=histogram[at];chunk_prefix[at]=total;total+=count;
            }
        } else total=c.route.offsets[e+1]-c.route.offsets[e];
        c.expert_rows[e]=total;
    }
    if constexpr(Histogram) {
        int inclusive=routing::warp_scan(total);
        if(lane==31) warp_totals[warp]=inclusive;
        __syncthreads();
        if(!warp) {
            int v=lane<int(blockDim.x)/32?warp_totals[lane]:0;
            warp_totals[lane]=routing::warp_scan(v)-v;
        }
        __syncthreads();
        if(e<c.route.experts) {
            int begin=warp_totals[warp]+inclusive-total;
            c.route.offsets[e]=begin;
            if(e==c.route.experts-1) c.route.offsets[c.route.experts]=begin+total;
        }
        __syncthreads();
    }
    int active=total>0,inclusive=routing::warp_scan(active);
    if(lane==31) warp_totals[warp]=inclusive;
    __syncthreads();
    if(!warp) {
        int v=lane<int(blockDim.x)/32?warp_totals[lane]:0;
        warp_totals[lane]=routing::warp_scan(v)-v;
    }
    __syncthreads();
    int begin=warp_totals[warp]+inclusive-active;
    if(active) c.active_experts[begin]=e;
    if(e==c.route.experts-1) *c.active_count=begin+active;
}

// Each column CTA derives its destination from the completed prefix/ranks;
// it never reads a map being written by another CTA in the same launch.
template<int TopK,int Tile,bool Tiled>
__global__ void prepare_gather(qzm_prepare_v1 c,int nr,int const* prefix,int const* ranks) {
    int64_t column_tiles=(int64_t(c.columns)+255)/256;
    int k=TopK?TopK:c.route.topk;
    for(int64_t tile=blockIdx.x;tile<int64_t(nr)*column_tiles;tile+=gridDim.x) {
        int row=int(tile/column_tiles),column_tile=int(tile%column_tiles),r,to,source;
        if constexpr(Tiled) {
            r=row;
            int id=c.route.ids[int64_t(r/k)*c.route.ids_stride+r%k];
            if(id<0 || id>=c.route.experts) continue; // caller's valid-ID contract
            to=c.route.offsets[id]+prefix[int64_t(r/Tile)*c.route.experts+id]+ranks[r];
            source=int(int64_t(r/k)*c.route.source_stride+(r%k)%c.route.channels);
        } else {
            to=row;r=c.route.dst_rows[to];source=c.route.src_rows[to];
        }
        if(!threadIdx.x && !column_tile) {
            if constexpr(Tiled) {c.route.src_rows[to]=source;c.route.dst_rows[to]=r;}
            c.inverse_rows[r]=to;
        }
        int64_t col=int64_t(column_tile)*256+threadIdx.x;
        if(col<c.columns)
            store(c.output,int64_t(to)*c.columns+col,c.dtype,c.input[int64_t(source)*c.input_row_stride+col]);
    }
}

template<int TopK,int Tile>
int prepare_launch(qzm_prepare_v1 const& c,routing::Layout const& l,int stage) {
    auto stream=static_cast<hggcStream_t>(c.route.stream);
    if(!l.rows) {
        if(stage>0) return QZR_INVALID;
        prepare_empty<<<(c.route.experts+256)/256,256,0,stream>>>(c);
        return launch_status();
    }
    int threads=((c.route.experts+31)/32)*32;
    if(l.algorithm!=QZR_TILED) {
        if(stage<0 || stage==0) {
            int rc=quactlize_ppu_moe_route_run_v1(&c.route,&c.config);if(rc) return rc;
        }
        if(stage<0 || stage==1) {
            prepare_prefix<false><<<1,threads,0,stream>>>(c,0,nullptr,nullptr);
            int rc=launch_status();if(rc) return rc;
        }
        if(stage<0 || stage==2) {
            int64_t count=int64_t(l.rows)*((int64_t(c.columns)+255)/256);
            int blocks=int(count>65535?65535:count);
            prepare_gather<TopK,Tile,false><<<blocks,256,0,stream>>>(c,l.rows,nullptr,nullptr);
        }
    } else {
        auto histogram=static_cast<int*>(c.route.workspace);
        auto prefix=reinterpret_cast<int*>(static_cast<uint8_t*>(c.route.workspace)+l.prefix);
        auto ranks=reinterpret_cast<int*>(static_cast<uint8_t*>(c.route.workspace)+l.ranks);
        if(stage<0 || stage==0) {
            routing::histogram_rank<TopK,Tile><<<l.tiles,128,size_t(4*c.route.experts+2*Tile)*4,stream>>>(
                c.route,l.rows,histogram,ranks);
            int rc=launch_status();if(rc) return rc;
        }
        if(stage<0 || stage==1) {
            prepare_prefix<true><<<1,threads,0,stream>>>(c,l.tiles,histogram,prefix);
            int rc=launch_status();if(rc) return rc;
        }
        if(stage<0 || stage==2) {
            int64_t count=int64_t(l.rows)*((int64_t(c.columns)+255)/256);
            int blocks=int(count>65535?65535:count);
            prepare_gather<TopK,Tile,true><<<blocks,256,0,stream>>>(c,l.rows,prefix,ranks);
        }
    }
    return launch_status();
}
template<int TopK>
int prepare_tiles(qzm_prepare_v1 const& c,routing::Layout const& l,int stage) {
    if(l.tile==256) return prepare_launch<TopK,256>(c,l,stage);
    if(l.tile==1024) return prepare_launch<TopK,1024>(c,l,stage);
    return prepare_launch<TopK,512>(c,l,stage);
}
int run_prepare(qzm_prepare_v1 const* c,int stage) {
    if(!c || stage<-1 || stage>2) return QZR_INVALID;
    routing::Layout l;int rc=prepare(*c,l,true);if(rc) return rc;
    return c->route.topk==8?prepare_tiles<8>(*c,l,stage):prepare_tiles<0>(*c,l,stage);
}

__device__ int input_row(qzm_value_v1 const& v,int row,int order,int const* row_ids,int const* inverse) {
    return v.order==order?row:order==QZM_GROUPED?row_ids[row]:inverse[row];
}
__global__ void transition_kernel(qzm_transition_v1 c,int nr) {
    int64_t tiles=(int64_t(c.hidden)+255)/256;
    for(int64_t tile=blockIdx.x;tile<int64_t(nr)*tiles;tile+=gridDim.x) {
        int row=int(tile/tiles);
        int64_t col=(tile%tiles)*256+threadIdx.x;
        if(col>=c.hidden) continue;
        int grow=input_row(c.gate,row,c.output_order,c.row_ids,c.inverse_rows);
        float g=projection(c.gate,grow,int(col)),v=g;
        if(c.gate.state!=QZM_ACTIVATED) {
            float u=c.merged?projection(c.gate,grow,int(col)+c.hidden):
                projection(c.up,input_row(c.up,row,c.output_order,c.row_ids,c.inverse_rows),int(col));
            v=(g/(1.f+expf(-g)))*u;
        }
        store(c.output,int64_t(row)*c.output_stride+col,c.output_dtype,v);
    }
}
__global__ void finish_kernel(qzm_finish_v1 c) {
    int64_t tiles=(int64_t(c.columns)+255)/256;
    for(int64_t tile=blockIdx.x;tile<int64_t(c.tokens)*tiles;tile+=gridDim.x) {
        int token=int(tile/tiles);
        int64_t col=(tile%tiles)*256+threadIdx.x;
        if(col>=c.columns) continue;
        float sum=0.f;
        for(int slot=0;slot<c.topk;++slot) {
            int original=token*c.topk+slot;
            int row=c.down.order==QZM_GROUPED?c.inverse_rows[original]:original;
            float v=projection(c.down,row,int(col));
            float w=__fmul_rn(v,c.weights[int64_t(token)*c.weights_stride+slot]);
            sum=slot?__fadd_rn(sum,w):w;
        }
        c.output[int64_t(token)*c.output_stride+col]=sum;
    }
}
} // namespace quactlize::moe

extern "C" int quactlize_ppu_moe_prepare_query_v1(qzm_prepare_v1 const* c,qzr_sizes_v1* out) {
    if(!c || !out) return QZR_INVALID;
    quactlize::routing::Layout l;int rc=quactlize::moe::prepare(*c,l,false);
    if(!rc) *out={l.bytes,l.algorithm,l.tile,l.tiles,l.rows?3:1};
    return rc;
}
extern "C" int quactlize_ppu_moe_prepare_run_v1(qzm_prepare_v1 const* c) {
    return quactlize::moe::run_prepare(c,-1);
}
extern "C" int quactlize_ppu_moe_transition_query_v1(qzm_transition_v1 const* c,uint32_t* ops) {
    if(!c || !ops) return QZR_INVALID;
    return quactlize::moe::transition(*c,*ops,false);
}
extern "C" int quactlize_ppu_moe_transition_run_v1(qzm_transition_v1 const* c) {
    using namespace quactlize::moe;
    if(!c) return QZR_INVALID;
    uint32_t ops=0;int rc=transition(*c,ops,true);if(rc || ops==QZM_NOOP) return rc;
    int64_t nr=int64_t(c->tokens)*c->topk,count=nr*((int64_t(c->hidden)+255)/256);
    transition_kernel<<<int(count>65535?65535:count),256,0,static_cast<hggcStream_t>(c->stream)>>>(*c,int(nr));
    return launch_status();
}
extern "C" int quactlize_ppu_moe_finish_run_v1(qzm_finish_v1 const* c) {
    using namespace quactlize::moe;
    if(!c) return QZR_INVALID;
    int rc=finish(*c,true);if(rc || !c->tokens) return rc;
    int64_t count=int64_t(c->tokens)*((int64_t(c->columns)+255)/256);
    finish_kernel<<<int(count>65535?65535:count),256,0,static_cast<hggcStream_t>(c->stream)>>>(*c);
    return launch_status();
}
namespace quactlize::moe {
int prepare_separate(qzm_prepare_v1 const* c,int stage) {
    using namespace quactlize;
    if(!c) return QZR_INVALID;
    routing::Layout l;int rc=moe::prepare(*c,l,true);if(rc) return rc;
    if(!l.rows) return moe::run_prepare(c,stage);
    int route_stages=l.stages;
    if(stage<0 || stage>=route_stages+2) return QZR_INVALID;
    if(stage<route_stages) return routing::run(&c->route,&c->config,stage);
    auto stream=static_cast<hggcStream_t>(c->route.stream);
    if(stage==route_stages)
        moe::prepare_prefix<false><<<1,((c->route.experts+31)/32)*32,0,stream>>>(*c,0,nullptr,nullptr);
    else {
        int64_t count=int64_t(l.rows)*((int64_t(c->columns)+255)/256);
        moe::prepare_gather<0,512,false><<<int(count>65535?65535:count),256,0,stream>>>(*c,l.rows,nullptr,nullptr);
    }
    return moe::launch_status();
}
// Keep routing publication separate from grouped-row traversal. This candidate
// saves only the standalone expert-metadata launch, not the map dependency.
int prepare_hybrid(qzm_prepare_v1 const* c,int stage) {
    using namespace quactlize;
    if(!c || stage<-1 || stage>3 || c->config.algorithm!=QZR_TILED) return QZR_INVALID;
    routing::Layout l;int rc=moe::prepare(*c,l,true);if(rc) return rc;
    if(!l.rows) return stage<=0?moe::run_prepare(c,0):QZR_INVALID;
    for(int s=0;s<4;++s) {
        if(stage>=0 && stage!=s) continue;
        if(s<2) rc=moe::run_prepare(c,s);
        else if(s==2) rc=routing::run(&c->route,&c->config,2);
        else rc=prepare_separate(c,4);
        if(rc) return rc;
    }
    return QZR_OK;
}
// A warp owns 32 columns of one slot. Each slot keeps the original ordered
// Split-K reduction and projection rounding; only independent slots run in
// parallel. The final warp still sums slots 0..topk-1 without reassociation.
template<int Warps>
__global__ void finish_slots(qzm_finish_v1 c) {
    __shared__ float weighted[32][32];
    int lane=int(threadIdx.x)&31,warp=int(threadIdx.x)/32;
    int64_t tiles=(int64_t(c.columns)+31)/32;
    for(int64_t tile=blockIdx.x;tile<int64_t(c.tokens)*tiles;tile+=gridDim.x) {
        int token=int(tile/tiles);
        int64_t col=(tile%tiles)*32+lane;
        for(int slot=warp;slot<c.topk;slot+=Warps) {
            float value=0.f;
            if(col<c.columns) {
                int original=token*c.topk+slot;
                int row=c.down.order==QZM_GROUPED?c.inverse_rows[original]:original;
                value=__fmul_rn(projection(c.down,row,int(col)),
                                c.weights[int64_t(token)*c.weights_stride+slot]);
            }
            weighted[slot][lane]=value;
        }
        __syncthreads();
        if(!warp && col<c.columns) {
            float sum=weighted[0][lane];
            for(int slot=1;slot<c.topk;++slot) sum=__fadd_rn(sum,weighted[slot][lane]);
            c.output[int64_t(token)*c.output_stride+col]=sum;
        }
        // All consumers finish before any producer overwrites shared storage
        // on the next grid-stride iteration, including invalid column lanes.
        __syncthreads();
    }
}
__global__ void materialize(qzm_value_v1 v,int nr,int nc,float* out) {
    for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<int64_t(nr)*nc;i+=int64_t(gridDim.x)*256)
        out[i]=projection(v,int(i/nc),int(i%nc));
}
#ifdef QZM_TESTING
__global__ void cast_materialized(float const* in,void* out,int nr,int nc,int type,int64_t stride) {
    for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<int64_t(nr)*nc;i+=int64_t(gridDim.x)*256)
        store(out,(i/nc)*stride+i%nc,type,in[i]);
}
#endif
__global__ void weight_slots(qzm_finish_v1 c,float const* projected,float* weighted) {
    int64_t nr=int64_t(c.tokens)*c.topk;
    for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<nr*c.columns;i+=int64_t(gridDim.x)*256) {
        int row=int(i/c.columns),col=int(i%c.columns);
        int from=c.down.order==QZM_GROUPED?c.inverse_rows[row]:row;
        weighted[i]=__fmul_rn(projected[int64_t(from)*c.columns+col],
            c.weights[int64_t(row/c.topk)*c.weights_stride+row%c.topk]);
    }
}
__global__ void sum_slots(qzm_finish_v1 c,float const* weighted) {
    for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<int64_t(c.tokens)*c.columns;i+=int64_t(gridDim.x)*256) {
        int token=int(i/c.columns),col=int(i%c.columns);float sum=0.f;
        for(int s=0;s<c.topk;++s) {
            float v=weighted[(int64_t(token)*c.topk+s)*c.columns+col];
            sum=s?__fadd_rn(sum,v):v;
        }
        c.output[int64_t(token)*c.output_stride+col]=sum;
    }
}
inline int flat_blocks(int rows,int columns) {
    int64_t n=(int64_t(rows)*columns+255)/256;
    return int(n>65535?65535:n);
}
int finish_parallel(qzm_finish_v1 const* c,int warps) {
    using namespace quactlize::moe;
    if(!c || (warps!=4 && warps!=8)) return QZR_INVALID;
    int rc=finish(*c,true);if(rc || !c->tokens) return rc;
    int64_t count=int64_t(c->tokens)*((int64_t(c->columns)+31)/32);
    int blocks=int(count>65535?65535:count);
    auto stream=static_cast<hggcStream_t>(c->stream);
    if(warps==4) finish_slots<4><<<blocks,128,0,stream>>>(*c);
    else finish_slots<8><<<blocks,256,0,stream>>>(*c);
    return launch_status();
}
} // namespace quactlize::moe

extern "C" int quactlize_ppu_moe_prepare_execute_v1(qzm_prepare_v1 const* c,qzm_plan_v1 const* plan) {
    using namespace quactlize::moe;
    if(!c || !plan) return QZR_INVALID;
    qzm_plan_v1 expected{};int rc=prepare_plan(*c,plan->implementation,expected);if(rc) return rc;
    if(!same_plan(*plan,expected)) return QZR_INVALID;
    if(plan->implementation==QZM_PREPARE_FUSED) return run_prepare(c,-1);
    if(plan->implementation==QZM_PREPARE_HYBRID) return prepare_hybrid(c,-1);
    for(int stage=0;stage<plan->stages;++stage) {
        rc=prepare_separate(c,stage);if(rc) return rc;
    }
    return QZR_OK;
}
extern "C" int quactlize_ppu_moe_prepare_execute_v2(qzm_prepare_v1 const* c,qzm_prepare_plan_v2 const* plan) {
    using namespace quactlize::moe;
    if(!c || !plan) return QZR_INVALID;
    qzm_prepare_plan_v2 expected{};int rc=prepare_plan_v2(*c,expected);if(rc) return rc;
    if(plan->version!=expected.version || plan->size!=sizeof(*plan) ||
       plan->config.version!=expected.config.version || plan->config.size!=expected.config.size ||
       plan->config.algorithm!=expected.config.algorithm || plan->config.tile_rows!=expected.config.tile_rows ||
       !same_plan(plan->helper,expected.helper)) return QZR_INVALID;
    qzm_prepare_v1 selected=*c;selected.config=plan->config;
    return quactlize_ppu_moe_prepare_execute_v1(&selected,&plan->helper);
}
extern "C" int quactlize_ppu_moe_finish_execute_v1(qzm_finish_v1 const* c,qzm_plan_v1 const* plan,
                                                  void* workspace,uint64_t bytes) {
    using namespace quactlize::moe;
    if(!c || !plan) return QZR_INVALID;
    qzm_plan_v1 expected{};int rc=finish_plan(*c,plan->implementation,expected);if(rc) return rc;
    if(!same_plan(*plan,expected)) return QZR_INVALID;
    rc=finish(*c,true);if(rc) return rc;
    rc=finish_workspace(*c,plan->workspace_bytes,workspace,bytes);if(rc || !c->tokens) return rc;
    if(plan->implementation==QZM_FINISH_SERIAL) return quactlize_ppu_moe_finish_run_v1(c);
    if(plan->implementation==QZM_FINISH_SLOTS4 || plan->implementation==QZM_FINISH_SLOTS8)
        return finish_parallel(c,plan->implementation==QZM_FINISH_SLOTS4?4:8);
    auto projected=static_cast<float*>(workspace);
    auto weighted=reinterpret_cast<float*>(static_cast<uint8_t*>(workspace)+plan->workspace_bytes/2);
    auto stream=static_cast<hggcStream_t>(c->stream);int nr=c->tokens*c->topk;
    materialize<<<flat_blocks(nr,c->columns),256,0,stream>>>(c->down,nr,c->columns,projected);
    rc=launch_status();if(rc) return rc;
    weight_slots<<<flat_blocks(nr,c->columns),256,0,stream>>>(*c,projected,weighted);
    rc=launch_status();if(rc) return rc;
    sum_slots<<<flat_blocks(c->tokens,c->columns),256,0,stream>>>(*c,weighted);
    return launch_status();
}

#ifdef QZM_TESTING
extern "C" int qzm_test_prepare_stage_v1(qzm_prepare_v1 const* c,int stage) {
    return quactlize::moe::run_prepare(c,stage);
}
extern "C" int qzm_test_prepare_unfused_v1(qzm_prepare_v1 const* c,int stage) {
    return quactlize::moe::prepare_separate(c,stage);
}
extern "C" int qzm_test_prepare_hybrid_v1(qzm_prepare_v1 const* c,int stage) {
    return quactlize::moe::prepare_hybrid(c,stage);
}
extern "C" int qzm_test_finish_parallel_v1(qzm_finish_v1 const* c,int warps) {
    return quactlize::moe::finish_parallel(c,warps);
}
extern "C" int qzm_test_materialize_v1(qzm_value_v1 const* v,int rows,int columns,float* out,void* stream) {
    using namespace quactlize::moe;
    uint64_t bytes=0;
    if(!v || value(*v,rows,columns,bytes) || !out) return QZR_INVALID;
    if(!rows) return QZR_OK;
    materialize<<<flat_blocks(rows,columns),256,0,static_cast<hggcStream_t>(stream)>>>(*v,rows,columns,out);
    return launch_status();
}
extern "C" int qzm_test_cast_v1(float const* in,void* out,int rows,int columns,int type,int64_t stride,void* stream) {
    using namespace quactlize::moe;
    if(rows<0 || columns<1 || !dtype(type) || stride<columns || !in || !out) return QZR_INVALID;
    if(!rows) return QZR_OK;
    cast_materialized<<<flat_blocks(rows,columns),256,0,static_cast<hggcStream_t>(stream)>>>(in,out,rows,columns,type,stride);
    return launch_status();
}
extern "C" int qzm_test_weight_v1(qzm_finish_v1 const* c,float const* projected,float* weighted) {
    using namespace quactlize::moe;
    if(!c || finish(*c,true) || !projected || !weighted) return QZR_INVALID;
    if(!c->tokens) return QZR_OK;
    weight_slots<<<flat_blocks(c->tokens*c->topk,c->columns),256,0,static_cast<hggcStream_t>(c->stream)>>>(*c,projected,weighted);
    return launch_status();
}
extern "C" int qzm_test_sum_v1(qzm_finish_v1 const* c,float const* weighted) {
    using namespace quactlize::moe;
    if(!c || finish(*c,true) || !weighted) return QZR_INVALID;
    if(!c->tokens) return QZR_OK;
    sum_slots<<<flat_blocks(c->tokens,c->columns),256,0,static_cast<hggcStream_t>(c->stream)>>>(*c,weighted);
    return launch_status();
}
#endif
