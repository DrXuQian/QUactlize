// Runs production SIMT stages with independent completed-projection fixtures.
// Does not replace or emulate a PPU GEMM.
#include <cuda_runtime.h>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"
namespace quactlize::runtime { using Half=cutlass::half_t; }
#include "quactlize/runtime/indexed.cuh"
#include "quactlize/runtime/moe_chain.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

void check(cudaError_t rc);
static bool benchmark_prepare=false;
static bool generic_prepare=false;
static bool multi_token=false;
static bool mixed_stages=false;
template<class F> static void time_prepare(F prepare,int merged,int tokens,int topk,int experts,int router,int k) {
  constexpr int batch=64,samples=15;
  cudaGraph_t graph; cudaGraphExec_t instance;
  cudaEvent_t start,stop;
  check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
  for (int i=0;i<batch;++i) prepare();
  check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
  check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
  check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
  std::vector<float> times;
  for (int i=-5;i<samples;++i) {
    check(cudaEventRecord(start,cudaStreamPerThread));
    check(cudaGraphLaunch(instance,cudaStreamPerThread));
    check(cudaEventRecord(stop,cudaStreamPerThread)); check(cudaEventSynchronize(stop));
    float ms=0; check(cudaEventElapsedTime(&ms,start,stop));
    if (i>=0) times.push_back(ms*1000.f/batch);
  }
  auto sorted=times; std::sort(sorted.begin(),sorted.end());
  std::printf("KPACK_MOE_PREPARE_PERF merged=%d tokens=%d topk=%d experts=%d router=%d k=%d median_us=%.6f batch=%d warmups=5 samples=[",
      merged,tokens,topk,experts,router,k,sorted[samples/2],batch);
  for (int i=0;i<samples;++i) std::printf("%s%.6f",i?",":"",times[i]);
  std::puts("]");
  check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
  check(cudaGraphExecDestroy(instance)); check(cudaGraphDestroy(graph));
}

void check(cudaError_t rc) { if (rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
template<class T> struct Buffer {
  T* ptr=nullptr; size_t count;
  explicit Buffer(size_t n):count(n) { check(cudaMalloc(&ptr,sizeof(T)*n)); }
  ~Buffer() { cudaFree(ptr); }
  void put(std::vector<T> const& v) {
    if (v.size()!=count) throw std::runtime_error("upload size");
    check(cudaMemcpy(ptr,v.data(),sizeof(T)*count,cudaMemcpyHostToDevice));
  }
  std::vector<T> get() const { std::vector<T> v(count);
    check(cudaMemcpy(v.data(),ptr,sizeof(T)*count,cudaMemcpyDeviceToHost)); return v; }
};
using namespace quactlize::runtime;
using Shape=cute::Shape<int,int,int>;
using Stride=cute::Stride<int64_t,cute::_1,cute::_0>;

__global__ void router_probe(qk_llama_router_v1 router,qk_llama_indexed_v1 io,bool fast) {
  int fixture=int(blockIdx.x);
  router.logits+=fixture*256;
  if (router.bias) router.bias+=fixture*256;
  router.weights+=fixture*16;
  io.ids+=fixture*32;
  __shared__ int ids[32];
  if (fast) {
    if (router.bias) quactlize::llama::router_256_top8<true>(router,io,ids);
    else quactlize::llama::router_256_top8<false>(router,io,ids);
  } else quactlize::llama::router_256(router,io,ids,true);
  __syncthreads();
  if (threadIdx.x<8) const_cast<int32_t*>(io.ids)[16+threadIdx.x]=ids[threadIdx.x];
}

static void router_equivalence() {
  constexpr int fixtures=256;
  Buffer<float> logits(fixtures*256),bias(fixtures*256),weights(fixtures*16);
  Buffer<int> ids(fixtures*32);
  std::vector<float> hl(logits.count),hb(bias.count);
  uint32_t state=0x93457719;
  for (int f=0;f<fixtures;++f) for (int e=0;e<256;++e) {
    state^=state<<13; state^=state>>17; state^=state<<5;
    float value=float(int(state%65537)-32768)*.00317f;
    // Equal scores, signed zero, sparse extremes and nonfinite inputs all
    // exercise the same exact tie/sanitization rules as the generic router.
    if (f==0) value=0.f;
    if (f==1) value=e%2?-0.f:0.f;
    if (f==2) value=float(e%7);
    if (f==3) value=e%3?NAN:float(e);
    if (f==4) value=-INFINITY;
    if (f==5) value=e==37?INFINITY:-20.f;
    if (f==6) value=e<8?10000.f:-10000.f;
    hl[f*256+e]=value;
    hb[f*256+e]=f<3?0.f:float(int(state%131)-65)*.01937f;
  }
  logits.put(hl); bias.put(hb);
  qk_llama_indexed_v1 io{};
  io.tokens=1; io.topk=8; io.ids_stride=8; io.ids=ids.ptr;
  size_t bad=0;
  for (int mode=0;mode<16;++mode) {
    qk_llama_router_v1 router{1,sizeof(router),mode&1,(mode>>1)&1,(mode>>2)&1,0,
        6.103515625e-5f,1.25f,logits.ptr,mode&8?bias.ptr:nullptr,weights.ptr};
    std::vector<int> reference_ids;
    std::vector<float> reference_weights;
    for (bool fast:{false,true}) {
      ids.put(std::vector<int>(ids.count,-123));
      weights.put(std::vector<float>(weights.count,-123.f));
      router_probe<<<fixtures,256>>>(router,io,fast);
      check(cudaGetLastError()); check(cudaDeviceSynchronize());
      auto got_ids=ids.get(); auto got_weights=weights.get();
      if (!fast) { reference_ids=got_ids; reference_weights=got_weights; }
      else {
        for (size_t i=0;i<got_ids.size();++i) bad+=got_ids[i]!=reference_ids[i];
        for (size_t i=0;i<got_weights.size();++i)
          bad+=std::memcmp(&got_weights[i],&reference_weights[i],sizeof(float))!=0;
      }
      for (int f=0;f<fixtures;++f) for (int j=0;j<8;++j) {
        bad+=got_ids[f*32+j]!=got_ids[f*32+16+j];
        bad+=got_ids[f*32+8+j]!=-123 || got_ids[f*32+24+j]!=-123;
        bad+=got_weights[f*16+8+j]!=-123.f;
      }
    }
  }
  std::printf("KPACK_MOE_ROUTER_EQUIVALENCE modes=16 fixtures=256 ids_and_weights=RAW_BITS bad=%zu\n",bad);
  if (bad) throw std::runtime_error("router equivalence failed");
}

struct Projection {
  qk_moe_projection_v1 p{};
  Buffer<Half> a,completed;
  Buffer<float> partial;
  Buffer<float> float_a,float_completed;
  Buffer<int> offsets,rows,row_ids;
  Buffer<Shape> shapes;
  Buffer<Stride> strides;
  Buffer<void*> outputs;
  Buffer<moe::Header> header;
  Buffer<moe::BlockEntry> entries;
  bool simt;
  Projection(int m,int n,int k,int e,int tm,int s,bool simt=false):a(size_t(m)*k+2),completed(size_t(m)*n),
    partial(size_t(s)*m*n),float_a(size_t(m)*k+2),float_completed(size_t(m)*n+2),
    offsets(e+1),rows(e),row_ids(m),shapes(e),strides(e*s),outputs(e*s),header(1),entries(m+2),simt(simt) {
    p.version=1; p.size=sizeof(p); p.m=m; p.n=n; p.k=k; p.experts=e; p.tile_m=tm; p.splits=s;
    p.a=a.ptr+1; p.output=completed.ptr; p.partials=partial.ptr;
    p.offsets=offsets.ptr; p.rows=rows.ptr; p.shapes=shapes.ptr; p.strides=strides.ptr;
    p.outputs=outputs.ptr; p.directory_header=header.ptr; p.directory_entries=entries.ptr+1; p.directory_capacity=m;
    p.io.row_ids=row_ids.ptr;
    if (simt) {p.a=float_a.ptr+1;p.output=float_completed.ptr+1;}
  }
  std::vector<float> seed(int replay) {
    std::vector<float> raw(partial.count), sums(size_t(p.m)*p.n);
    std::vector<Half> half(completed.count);
    for (size_t i=0;i<raw.size();++i) raw[i]=float(int((i*17+replay*13)%113)-56)*.10371f;
    for (size_t i=0;i<half.size();++i) {
      float sum=0;
      for (int s=0;s<p.splits;++s) sum+=raw[size_t(s)*half.size()+i];
      half[i]=Half(sum); sums[i]=simt?sum:float(half[i]);
    }
    partial.put(raw); completed.put(half);
    a.put(std::vector<Half>(a.count,Half(-77.f)));
    float_a.put(std::vector<float>(float_a.count,-77.f));
    std::vector<float> values(float_completed.count,-77.f);
    std::copy(sums.begin(),sums.end(),values.begin()+1);float_completed.put(values);
    entries.put(std::vector<moe::BlockEntry>(entries.count,{-123,-123,-123,-123}));
    offsets.put(std::vector<int>(offsets.count,-123));rows.put(std::vector<int>(rows.count,-123));
    shapes.put(std::vector<Shape>(shapes.count,cute::make_shape(-123,-123,-123)));
    strides.put(std::vector<Stride>(strides.count,cute::make_stride(int64_t(-123),cute::_1{},cute::_0{})));
    outputs.put(std::vector<void*>(outputs.count,nullptr));
    return sums;
  }
};

static void run(bool merged,int tokens,int topk,int experts,int sg,int su,int sd,int router_mode=-1,int k=2048,uint32_t simt_mask=0) {
  int m=tokens*topk,n=512,out_n=1024,ids_stride=topk+5;
  if (!fused_indexed_rows(m,tokens)) throw std::runtime_error("outside bounded fused decode chain");
  Buffer<int> ids(tokens*ids_stride);
  Buffer<float> source(size_t(tokens)*(k+17)), output(size_t(m)*(out_n+3)+2);
  Buffer<float> logits(tokens*256), bias(256), weights(m);
  if (simt_mask&1) sg=1;
  if (simt_mask&2) su=1;
  if (simt_mask&4) sd=1;
  Projection gate(m,n*(merged?2:1),k,experts,8,sg,simt_mask&1), up(m,n,k,experts,16,su,simt_mask&2), down(m,out_n,n,experts,32,sd,simt_mask&4);
  auto io=[&](Projection& p) {
    p.p.io={1,sizeof(qk_llama_indexed_v1),tokens,topk,1,0,ids_stride,k+17,k+17,out_n+3,
        ids.ptr,source.ptr,output.ptr+1,p.row_ids.ptr};
  };
  io(gate); io(up); io(down);
  qk_moe_plan_v1 plan{1,sizeof(plan),uint32_t(merged),0,gate.p,up.p,down.p};
  if (router_mode>=0) plan.router={1,sizeof(plan.router),int(router_mode==1),int(router_mode!=2),
      int(router_mode==2),0,6.103515625e-5f,1.25f,logits.ptr,router_mode==1?bias.ptr:nullptr,weights.ptr};
  MixedMoePlan mixed;static_cast<qk_moe_plan_v1&>(mixed)=plan;mixed.simt_mask=simt_mask;
  auto prepare=[&] {
    if (simt_mask) {
      if (moe_prepare_m1_supported(mixed)) moe_chain_prepare_m1<Shape,Stride><<<1,256,0,cudaStreamPerThread>>>(mixed);
      else if (m>32) moe_chain_prepare<Shape,Stride,64><<<moe_prepare_blocks(experts,m),256,0,cudaStreamPerThread>>>(mixed);
      else moe_chain_prepare<Shape,Stride><<<moe_prepare_blocks(experts,m),256,0,cudaStreamPerThread>>>(mixed);
      return;
    }
#ifdef KPACK_MOE_PREPARE_BASELINE
    moe_chain_prepare<Shape,Stride><<<dim3(8,m),256,0,cudaStreamPerThread>>>(plan);
#else
    if (!generic_prepare && moe_prepare_m1_supported(plan))
      moe_chain_prepare_m1<Shape,Stride><<<1,256,0,cudaStreamPerThread>>>(plan);
    else if (m>32) moe_chain_prepare<Shape,Stride,64><<<moe_prepare_blocks(experts,m),256,0,cudaStreamPerThread>>>(plan);
    else moe_chain_prepare<Shape,Stride><<<moe_prepare_blocks(experts,m),256,0,cudaStreamPerThread>>>(plan);
#endif
  };
  auto activate=[&] {
    if (simt_mask) moe_chain_swiglu_mixed<<<dim3(2,m),256,0,cudaStreamPerThread>>>(mixed);
    else moe_chain_swiglu<<<dim3(2,m),256,0,cudaStreamPerThread>>>(plan);
  };
  auto finish=[&] {
    if (simt_mask&4) return; // Stage-only fixture: no SIMT producer to finish.
#define FINISH(S) case S: indexed_finish<S><<<dim3(4,m),256,0,cudaStreamPerThread>>>( \
    down.partial.ptr,down.completed.ptr,output.ptr+1,down.row_ids.ptr,m,out_n,out_n+3,down.header.ptr); break
    switch(sd) { FINISH(1); FINISH(2); FINISH(4); FINISH(8); }
#undef FINISH
  };
  cudaGraph_t graph; cudaGraphExec_t instance;
  check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
  prepare(); activate(); finish(); check(cudaGetLastError());
  check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
  check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
  size_t bad=0,activation_half_bad=0,rounding_red=0,gate_up_red=0;
  for (int replay=0;replay<7;++replay) {
    std::vector<int> hi(ids.count,-1), logical(m), ordered(m);
    for (int t=0;t<tokens;++t) for (int s=0;s<topk;++s)
      hi[t*ids_stride+s]=logical[t*topk+s]=(t*3+s*5+replay*7)%experts;
    std::vector<float> expected_weights(m), hl(logits.count), hb(bias.count);
    if (router_mode>=0) {
      for (size_t j=0;j<hl.size();++j) hl[j]=replay==0?0.f:float(int((j*17+replay*29)%263)-131)*.0625f;
      for (size_t j=0;j<hb.size();++j) hb[j]=replay==0?0.f:float(int((j*31+replay*17)%257)-128)*.00437f;
      for (int t=0;t<tokens;++t) {
        std::vector<double> values(256), selection(256);
        std::vector<int> experts_order(256); std::iota(experts_order.begin(),experts_order.end(),0);
        double maximum=*std::max_element(hl.begin()+t*256,hl.begin()+(t+1)*256),total=0;
        for (int e=0;e<256;++e) {
          double x=hl[t*256+e];
          values[e]=router_mode==2?x:router_mode==1?1./(1.+std::exp(-x)):std::exp(x-maximum);
          total+=values[e];
        }
        if (router_mode==0) for (double& v:values) v/=total;
        for (int e=0;e<256;++e) selection[e]=values[e]+(router_mode==1?hb[e]:0.f);
        std::stable_sort(experts_order.begin(),experts_order.end(),[&](int x,int y){return selection[x]>selection[y];});
        total=0; maximum=values[experts_order[0]];
        for (int s=0;s<topk;++s) total+=router_mode==2?std::exp(values[experts_order[s]]-maximum):values[experts_order[s]];
        for (int s=0;s<topk;++s) {
          logical[t*topk+s]=experts_order[s];
          double v=values[experts_order[s]];
          expected_weights[t*topk+s]=float((router_mode==2?std::exp(v-maximum)/total:v/std::max(total,6.103515625e-5))*1.25);
        }
      }
      logits.put(hl); bias.put(hb); std::fill(hi.begin(),hi.end(),-123);
      weights.put(std::vector<float>(m,-123.f));
    }
    std::iota(ordered.begin(),ordered.end(),0);
    std::stable_sort(ordered.begin(),ordered.end(),[&](int x,int y){return logical[x]<logical[y];});
    ids.put(hi);
    std::vector<float> ha(source.count);
    for (size_t i=0;i<ha.size();++i) ha[i]=float(int((i*3+replay*7)%127)-63)*.07131f;
    source.put(ha); output.put(std::vector<float>(output.count,-123.f));
    auto gv=gate.seed(replay),uv=up.seed(replay+3),dv=down.seed(replay+1);
    check(cudaGraphLaunch(instance,cudaStreamPerThread)); check(cudaDeviceSynchronize());
    if (router_mode>=0) {
      auto got_ids=ids.get(); auto got_weights=weights.get();
      for (int t=0;t<tokens;++t) for (int s=0;s<ids_stride;++s)
        bad+=got_ids[t*ids_stride+s]!=(s<topk?logical[t*topk+s]:-123);
      for (int j=0;j<m;++j) bad+=std::abs(got_weights[j]-expected_weights[j])>2.e-6f;
    }
    for (auto * p:{&gate,merged?nullptr:&up,&down}) {
      if (!p) continue;
      auto offsets=p->offsets.get(), rows=p->rows.get(), map=p->row_ids.get();
      auto shapes=p->shapes.get(); auto strides=p->strides.get(); auto pointers=p->outputs.get();
      auto entries=p->entries.get(); auto h=p->header.get()[0];
      int begin=0,tiles=0;
      bool sparse_simt=p->simt && moe_prepare_m1_supported(plan);
      for (int e=0;e<experts;++e) {
        int count=std::count(logical.begin(),logical.end(),e);
        bad+=offsets[e]!=(sparse_simt?-123:begin) || rows[e]!=(sparse_simt?-123:count);
        bad+=cute::get<0>(shapes[e])!=(sparse_simt?-123:count) ||
             cute::get<1>(shapes[e])!=(sparse_simt?-123:p->p.n) || cute::get<2>(shapes[e])!=(sparse_simt?-123:p->p.k);
        for (int s=0;s<p->p.splits;++s) {
          void* want=p->p.splits==1 ? (void*)(static_cast<Half*>(p->p.output)+size_t(begin)*p->p.n) :
            (void*)(p->partial.ptr+(size_t(s)*m+begin)*p->p.n);
          bad+=pointers[s*experts+e]!=(sparse_simt?nullptr:want) ||
               cute::get<0>(strides[s*experts+e])!=(sparse_simt?-123:p->p.n);
        }
        for (int j=0;j<(count+p->p.tile_m-1)/p->p.tile_m;++j) {
          auto entry=entries[1+tiles+j];
          bad+=entry.expert!=e || entry.row_begin!=begin || entry.expert_rows!=count || entry.expert_block_begin!=tiles;
        }
        begin+=count; tiles+=(count+p->p.tile_m-1)/p->p.tile_m;
      }
      bad+=offsets.back()!=(sparse_simt?-123:m) || h.num_m_blocks!=tiles || h.status || h.tile_m!=p->p.tile_m;
      bad+=map!=ordered;
      bad+=entries.front().expert!=-123;
      for (int i=tiles+1;i<int(entries.size());++i) bad+=entries[i].expert!=-123;
    }
    auto ga=gate.a.get(), ua=up.a.get(), da=down.a.get();
    auto gfa=gate.float_a.get(), ufa=up.float_a.get(), dfa=down.float_a.get();
    for (auto p:{&gate,&up,&down}) {
      auto fa=p->float_a.get(),fc=p->float_completed.get();
      bad+=fa.front()!=-77.f || fa.back()!=-77.f || fc.front()!=-77.f || fc.back()!=-77.f;
    }
    auto got=output.get();
    bad+=ga.front().raw()!=Half(-77.f).raw() || ga.back().raw()!=Half(-77.f).raw();
    bad+=da.front().raw()!=Half(-77.f).raw() || da.back().raw()!=Half(-77.f).raw();
    for (int r=0;r<m;++r) {
      for (int c=0;c<k;++c) {
        Half want(ha[size_t(ordered[r]/topk)*(k+17)+c]);
        if (gate.simt) bad+=gfa[1+size_t(r)*k+c]!=-77.f;
        else bad+=ga[1+size_t(r)*k+c].raw()!=want.raw();
        if (!merged) {
          if (up.simt) bad+=ufa[1+size_t(r)*k+c]!=-77.f;
          else bad+=ua[1+size_t(r)*k+c].raw()!=want.raw();
        }
      }
      for (int c=0;c<n;++c) {
        size_t gr=gate.simt?ordered[r]:r,ur=up.simt?ordered[r]:r;
        float g=gv[gr*gate.p.n+c],u=merged?gv[gr*gate.p.n+c+n]:uv[ur*n+c];
        float value=(g/(1.f+std::exp(-g)))*u;
        Half want(value), swapped((u/(1.f+std::exp(-u)))*g);
        rounding_red+=float(want)!=value; gate_up_red+=want.raw()!=swapped.raw();
        // Host libm/device exp may straddle a half midpoint. Require tight
        // F32 error as well as counting exact half equality explicitly.
        if (down.simt) {
          float actual=dfa[1+size_t(ordered[r])*n+c];
          bad+=!std::isfinite(actual) || std::abs(actual-value)>0.00001f*std::max(1.f,std::abs(value));
        } else {
          auto actual=da[1+size_t(r)*n+c];
          activation_half_bad+=actual.raw()!=want.raw();
          bad+=std::abs(float(actual)-float(want))>0.001f*std::max(1.f,std::abs(float(want)));
        }
      }
      for (int c=0;c<out_n;++c) bad+=got[1+size_t(ordered[r])*(out_n+3)+c]!=(down.simt?-123.f:dv[size_t(r)*out_n+c]);
      for (int c=out_n;c<out_n+3;++c) bad+=got[1+size_t(r)*(out_n+3)+c]!=-123.f;
    }
    bad+=got.front()!=-123.f || got.back()!=-123.f;
    if (replay==6 && benchmark_prepare && bad==0) time_prepare(prepare,int(merged),tokens,topk,experts,router_mode,k);
    if (replay==6 && topk>1 && router_mode<0) {
      hi[1]=hi[0]; ids.put(hi);
      check(cudaGraphLaunch(instance,cudaStreamPerThread)); check(cudaDeviceSynchronize());
      bad+=!gate.header.get()[0].status || !down.header.get()[0].status;
      auto invalid=down.simt?down.float_a.get():output.get();
      for (int r=0;r<m;++r) for (int c=0;c<(down.simt?n:out_n);++c)
        bad+=std::isfinite(invalid[1+size_t(r)*(down.simt?n:out_n+3)+c]);
    }
  }
  check(cudaGraphExecDestroy(instance)); check(cudaGraphDestroy(graph));
  std::printf("KPACK_MOE_CHAIN_CUDA merged=%d tokens=%d topk=%d experts=%d splits=%d,%d,%d router=%d k=%d prepare=%s replays=7 bad=%zu activation_half_bad=%zu rounding_red=%zu gate_up_red=%zu simt_mask=%u\n",
      int(merged),tokens,topk,experts,sg,su,sd,router_mode,k,
      !generic_prepare&&moe_prepare_m1_supported(plan)?"M1_FAST":"GENERAL",bad,activation_half_bad,rounding_red,gate_up_red,simt_mask);
  if (bad || !rounding_red || !gate_up_red) throw std::runtime_error("MoE chain oracle failed");
}

__global__ void weighted_rows_reference(float const* input,float const* weights,float* output,
    int n,int64_t input_stride,int64_t weights_stride) {
  int row=int(blockIdx.y),col=int(blockIdx.x)*256+int(threadIdx.x);
  if (col<n) output[int64_t(row)*n+col]=__fmul_rn(input[int64_t(row)*input_stride+col],
      weights[int64_t(row/8)*weights_stride+row%8]);
}
__global__ void sum_slots_reference(float const* input,float* output,int n,int64_t stride) {
  int token=int(blockIdx.y),col=int(blockIdx.x)*128+int(threadIdx.x);
  if (col>=n) return;
  float sum=input[int64_t(token*8)*n+col];
  for (int slot=1;slot<8;++slot) sum=__fadd_rn(sum,input[int64_t(token*8+slot)*n+col]);
  output[int64_t(token)*stride+col]=sum;
}

template<class F,class G>
static void time_finish(F baseline,G candidate,bool simt,int tokens,int n,int splits) {
  constexpr int batch=64,samples=15;
  cudaGraph_t graphs[2];cudaGraphExec_t instances[2];
  for (int arm=0;arm<2;++arm) {
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
    for (int i=0;i<batch;++i) { if (arm) candidate();else baseline(); }
    check(cudaStreamEndCapture(cudaStreamPerThread,&graphs[arm]));
    check(cudaGraphInstantiate(&instances[arm],graphs[arm],nullptr,nullptr,0));
  }
  cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
  std::vector<float> times[2];
  for (int round=-5;round<samples;++round) for (int turn=0;turn<2;++turn) {
    int arm=(round+5+turn)%2;
    check(cudaEventRecord(begin,cudaStreamPerThread));
    check(cudaGraphLaunch(instances[arm],cudaStreamPerThread));check(cudaEventRecord(end,cudaStreamPerThread));
    check(cudaEventSynchronize(end));float ms=0;check(cudaEventElapsedTime(&ms,begin,end));
    if (round>=0) times[arm].push_back(ms*1000.f/batch);
  }
  auto a=times[0],b=times[1];std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
  std::printf("KPACK_MOE_FINISH_PERF tokens=%d n=%d split=%d simt=%d reference_us=%.6f fused_us=%.6f baseline_kernels=%d fused_kernels=1 batch=64 samples=15 scope=RESIDENT_HELPERS_NOT_MODEL baseline=UNFUSED_STAGE_REFERENCE_NOT_LLAMA_BINARY\n",
      tokens,n,splits,int(simt),a[7],b[7],simt?2:3);
  for (int arm=0;arm<2;++arm) {
    std::printf("KPACK_MOE_FINISH_SAMPLES arm=%d values=[",arm);
    for (int i=0;i<samples;++i) std::printf("%s%.6f",i?",":"",times[arm][i]);
    std::puts("]");check(cudaGraphExecDestroy(instances[arm]));check(cudaGraphDestroy(graphs[arm]));
  }
  check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
}

static void weighted_finish_equivalence() {
  size_t bad=0,order_red=0;
  for (bool simt:{false,true}) for (int tokens:{1,2,8}) for (int splits:{1,2,4,8}) {
    if (simt && splits!=1) continue;
    int m=tokens*8,n=2048;
    Projection down(m,n,512,256,8,splits,simt);
    Buffer<float> original(size_t(m)*(n+3)),weights(tokens*11),result(size_t(tokens)*(n+5)+2),weighted(size_t(m)*n);
    down.p.io={1,sizeof(qk_llama_indexed_v1),tokens,8,8,0,8,512,4096,n+3,nullptr,nullptr,original.ptr,down.row_ids.ptr};
    qk_llama_moe_finish_v1 finish{1,sizeof(finish),11,n+5,weights.ptr,result.ptr+1};
    auto launch=[&] {
      if (simt) moe_weighted_finish<true><<<dim3((n+127)/128,tokens),128,0,cudaStreamPerThread>>>(down.p,finish);
      else moe_weighted_finish<false><<<dim3((n+127)/128,tokens),128,0,cudaStreamPerThread>>>(down.p,finish);
    };
    auto reference=[&] {
      if (!simt) {
#define REF_FINISH(S) case S: indexed_finish<S><<<dim3((n+255)/256,m),256,0,cudaStreamPerThread>>>( \
    down.partial.ptr,down.completed.ptr,original.ptr,down.row_ids.ptr,m,n,n+3,down.header.ptr);break
        switch(splits) { REF_FINISH(1);REF_FINISH(2);REF_FINISH(4);REF_FINISH(8); }
#undef REF_FINISH
      }
      weighted_rows_reference<<<dim3((n+255)/256,m),256,0,cudaStreamPerThread>>>(original.ptr,weights.ptr,weighted.ptr,n,n+3,11);
      sum_slots_reference<<<dim3((n+127)/128,tokens),128,0,cudaStreamPerThread>>>(weighted.ptr,result.ptr+1,n,n+5);
    };
    cudaGraph_t graph;cudaGraphExec_t instance;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));launch();
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
    for (int replay=0;replay<7;++replay) {
      auto done=down.seed(replay);
      std::vector<int> map(m),inverse(m);
      for (int r=0;r<m;++r) {map[r]=(r*13+replay)%m;inverse[map[r]]=r;}
      down.row_ids.put(map);down.header.put({{m,0,8,256}});
      std::vector<float> source(original.count,-123.f),w(weights.count,-123.f);
      for (int r=0;r<m;++r) for (int col=0;col<n;++col) source[size_t(r)*(n+3)+col]=done[size_t(r)*n+col];
      for (int t=0;t<tokens;++t) for (int slot=0;slot<8;++slot) w[t*11+slot]=float((slot*31+replay*7)%67-33)*.13719f;
      original.put(source);weights.put(w);result.put(std::vector<float>(result.count,-123.f));
      check(cudaGraphLaunch(instance,cudaStreamPerThread));check(cudaDeviceSynchronize());
      auto got=result.get();
      for (int t=0;t<tokens;++t) for (int col=0;col<n;++col) {
        float want=0.f,reverse=0.f;
        for (int slot=0;slot<8;++slot) {
          int r=simt?t*8+slot:inverse[t*8+slot];
          volatile float product=done[size_t(r)*n+col]*w[t*11+slot];
          want=slot?want+product:product;
          int rev=7-slot,rr=simt?t*8+rev:inverse[t*8+rev];
          volatile float other=done[size_t(rr)*n+col]*w[t*11+rev];
          reverse=slot?reverse+other:other;
        }
        bad+=std::memcmp(&want,&got[1+size_t(t)*(n+5)+col],sizeof(float))!=0;
        order_red+=std::memcmp(&want,&reverse,sizeof(float))!=0;
      }
      bad+=got.front()!=-123.f || got.back()!=-123.f;
      for (int t=0;t<tokens;++t) for (int col=n;col<n+5;++col) bad+=got[1+size_t(t)*(n+5)+col]!=-123.f;
      reference();check(cudaGetLastError());check(cudaDeviceSynchronize());
      auto baseline=result.get();bad+=std::memcmp(baseline.data(),got.data(),baseline.size()*sizeof(float))!=0;
      if (replay==6 && benchmark_prepare && bad==0) time_finish(reference,launch,simt,tokens,n,splits);
      if (replay==6) {
        down.header.put({{0,1,8,256}});check(cudaGraphLaunch(instance,cudaStreamPerThread));check(cudaDeviceSynchronize());
        got=result.get();for (int t=0;t<tokens;++t) for (int col=0;col<n;++col)
          bad+=!std::isnan(got[1+size_t(t)*(n+5)+col]);
      }
    }
    check(cudaGraphExecDestroy(instance));check(cudaGraphDestroy(graph));
  }
  std::printf("KPACK_MOE_WEIGHTED_FINISH cells=15 replays=7 tokens=1,2,8 completion=F16,F32 split=1,2,4,8 raw_bad=%zu order_red=%zu\n",bad,order_red);
  if (bad || !order_red) throw std::runtime_error("weighted finish oracle failed");
}
int main(int argc,char** argv) {
  try {
    for (int i=1;i<argc;++i) {
      if (std::strcmp(argv[i],"--benchmark")==0) benchmark_prepare=true;
      else if (std::strcmp(argv[i],"--generic")==0) generic_prepare=true;
      else if (std::strcmp(argv[i],"--multi-token")==0) multi_token=true;
      else if (std::strcmp(argv[i],"--mixed")==0) mixed_stages=true;
      else throw std::runtime_error("usage: moe-chain [--benchmark] [--generic] [--multi-token]");
    }
    router_equivalence();
    weighted_finish_equivalence();
    if (mixed_stages) {
      for (bool merged:{false,true}) for (uint32_t mask=1;mask<8;++mask) {
        if (merged && (mask&2)) continue;
        for (int tokens:{1,2,4,8}) for (int router:{-1,0}) run(merged,tokens,8,256,4,2,8,router,2048,mask);
      }
      std::puts("KPACK_MOE_MIXED_STAGES PASS cells=80 PPU_GEMM_ADMISSION=NOT_TESTED");return 0;
    }
    if (multi_token) {
      for (int k:{512,2048,3072}) for (int tokens:{1,2,3,4,5,6,7,8})
        for (bool merged:{false,true}) for (int router:{-1,0,1,2})
          run(merged,tokens,8,256,4,2,8,router,k);
      std::puts("KPACK_MOE_CHAIN_CUDA PASS cells=192 PPU_GEMM_ADMISSION=NOT_TESTED");
      return 0;
    }
    run(false,1,8,256,4,2,1); run(false,4,8,256,1,8,4);
    run(true,1,8,256,4,1,2); run(true,9,1,2,2,1,8);
    run(false,1,8,256,4,2,1,0); run(false,4,8,256,4,2,4,1); run(true,1,8,256,2,1,8,2);
    run(true,1,8,256,2,1,4,0); run(false,1,8,256,8,4,2,1);
    run(false,17,1,1024,8,4,2); run(false,1,1,1,1,1,1); run(true,31,1,33,2,1,8);
    run(false,5,8,256,4,2,1); run(true,8,8,256,8,1,4);
    run(false,7,8,256,4,2,8,0); run(true,8,8,256,2,1,8,1);
    run(false,8,8,256,4,2,1,2);
    std::puts("KPACK_MOE_CHAIN_CUDA PASS cells=17 PPU_GEMM_ADMISSION=NOT_TESTED"); return 0;
  } catch (std::exception const& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
