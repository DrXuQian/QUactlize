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
#include <numeric>
#include <stdexcept>
#include <vector>

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
struct Projection {
  qk_moe_projection_v1 p{};
  Buffer<Half> a,completed;
  Buffer<float> partial;
  Buffer<int> offsets,rows,row_ids;
  Buffer<Shape> shapes;
  Buffer<Stride> strides;
  Buffer<void*> outputs;
  Buffer<moe::Header> header;
  Buffer<moe::BlockEntry> entries;
  Projection(int m,int n,int k,int e,int tm,int s):a(size_t(m)*k+2),completed(size_t(m)*n),
    partial(size_t(s)*m*n),offsets(e+1),rows(e),row_ids(m),shapes(e),strides(e*s),outputs(e*s),header(1),entries(m+2) {
    p.version=1; p.size=sizeof(p); p.m=m; p.n=n; p.k=k; p.experts=e; p.tile_m=tm; p.splits=s;
    p.a=a.ptr+1; p.output=completed.ptr; p.partials=partial.ptr;
    p.offsets=offsets.ptr; p.rows=rows.ptr; p.shapes=shapes.ptr; p.strides=strides.ptr;
    p.outputs=outputs.ptr; p.directory_header=header.ptr; p.directory_entries=entries.ptr+1; p.directory_capacity=m;
    p.io.row_ids=row_ids.ptr;
  }
  std::vector<float> seed(int replay) {
    std::vector<float> raw(partial.count), sums(size_t(p.m)*p.n);
    std::vector<Half> half(completed.count);
    for (size_t i=0;i<raw.size();++i) raw[i]=float(int((i*17+replay*13)%113)-56)*.10371f;
    for (size_t i=0;i<half.size();++i) {
      float sum=0;
      for (int s=0;s<p.splits;++s) sum+=raw[size_t(s)*half.size()+i];
      half[i]=Half(sum); sums[i]=float(half[i]);
    }
    partial.put(raw); completed.put(half);
    a.put(std::vector<Half>(a.count,Half(-77.f)));
    entries.put(std::vector<moe::BlockEntry>(entries.count,{-123,-123,-123,-123}));
    return sums;
  }
};

static void run(bool merged,int tokens,int topk,int experts,int sg,int su,int sd,int router_mode=-1) {
  int m=tokens*topk,k=2048,n=512,out_n=1024,ids_stride=topk+5;
  Buffer<int> ids(tokens*ids_stride);
  Buffer<float> source(size_t(tokens)*(k+17)), output(size_t(m)*(out_n+3)+2);
  Buffer<float> logits(tokens*256), bias(256), weights(m);
  Projection gate(m,n*(merged?2:1),k,experts,8,sg), up(m,n,k,experts,16,su), down(m,out_n,n,experts,32,sd);
  auto io=[&](Projection& p) {
    p.p.io={1,sizeof(qk_llama_indexed_v1),tokens,topk,1,0,ids_stride,k+17,k+17,out_n+3,
        ids.ptr,source.ptr,output.ptr+1,p.row_ids.ptr};
  };
  io(gate); io(up); io(down);
  qk_moe_plan_v1 plan{1,sizeof(plan),uint32_t(merged),0,gate.p,up.p,down.p};
  if (router_mode>=0) plan.router={1,sizeof(plan.router),int(router_mode==1),int(router_mode!=2),
      int(router_mode==2),0,6.103515625e-5f,1.25f,logits.ptr,router_mode==1?bias.ptr:nullptr,weights.ptr};
  auto prepare=[&] { moe_chain_prepare<Shape,Stride><<<dim3(8,m),256,0,cudaStreamPerThread>>>(plan); };
  auto activate=[&] { moe_chain_swiglu<<<dim3(2,m),256,0,cudaStreamPerThread>>>(plan); };
  auto finish=[&] {
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
      for (int e=0;e<experts;++e) {
        int count=std::count(logical.begin(),logical.end(),e);
        bad+=offsets[e]!=begin || rows[e]!=count;
        bad+=cute::get<0>(shapes[e])!=count || cute::get<1>(shapes[e])!=p->p.n || cute::get<2>(shapes[e])!=p->p.k;
        for (int s=0;s<p->p.splits;++s) {
          void* want=p->p.splits==1 ? (void*)(p->completed.ptr+size_t(begin)*p->p.n) :
            (void*)(p->partial.ptr+(size_t(s)*m+begin)*p->p.n);
          bad+=pointers[s*experts+e]!=want || cute::get<0>(strides[s*experts+e])!=p->p.n;
        }
        for (int j=0;j<(count+p->p.tile_m-1)/p->p.tile_m;++j) {
          auto entry=entries[1+tiles+j];
          bad+=entry.expert!=e || entry.row_begin!=begin || entry.expert_rows!=count || entry.expert_block_begin!=tiles;
        }
        begin+=count; tiles+=(count+p->p.tile_m-1)/p->p.tile_m;
      }
      bad+=offsets.back()!=m || h.num_m_blocks!=tiles || h.status || h.tile_m!=p->p.tile_m;
      bad+=map!=ordered;
      bad+=entries.front().expert!=-123;
      for (int i=tiles+1;i<int(entries.size());++i) bad+=entries[i].expert!=-123;
    }
    auto ga=gate.a.get(), ua=up.a.get(), da=down.a.get();
    auto got=output.get();
    bad+=ga.front().raw()!=Half(-77.f).raw() || ga.back().raw()!=Half(-77.f).raw();
    bad+=da.front().raw()!=Half(-77.f).raw() || da.back().raw()!=Half(-77.f).raw();
    for (int r=0;r<m;++r) {
      for (int c=0;c<k;++c) {
        Half want(ha[size_t(ordered[r]/topk)*(k+17)+c]);
        bad+=ga[1+size_t(r)*k+c].raw()!=want.raw();
        if (!merged) bad+=ua[1+size_t(r)*k+c].raw()!=want.raw();
      }
      for (int c=0;c<n;++c) {
        float g=gv[size_t(r)*gate.p.n+c],u=merged?gv[size_t(r)*gate.p.n+c+n]:uv[size_t(r)*n+c];
        float value=(g/(1.f+std::exp(-g)))*u;
        Half want(value), swapped((u/(1.f+std::exp(-u)))*g);
        rounding_red+=float(want)!=value; gate_up_red+=want.raw()!=swapped.raw();
        // Host libm/device exp may straddle a half midpoint. Require tight
        // F32 error as well as counting exact half equality explicitly.
        auto actual=da[1+size_t(r)*n+c];
        activation_half_bad+=actual.raw()!=want.raw();
        bad+=std::abs(float(actual)-float(want))>0.001f*std::max(1.f,std::abs(float(want)));
      }
      for (int c=0;c<out_n;++c) bad+=got[1+size_t(ordered[r])*(out_n+3)+c]!=dv[size_t(r)*out_n+c];
      for (int c=out_n;c<out_n+3;++c) bad+=got[1+size_t(r)*(out_n+3)+c]!=-123.f;
    }
    bad+=got.front()!=-123.f || got.back()!=-123.f;
    if (replay==6 && topk>1 && router_mode<0) {
      hi[1]=hi[0]; ids.put(hi);
      check(cudaGraphLaunch(instance,cudaStreamPerThread)); check(cudaDeviceSynchronize());
      bad+=!gate.header.get()[0].status || !down.header.get()[0].status;
      auto invalid=output.get();
      for (int r=0;r<m;++r) for (int c=0;c<out_n;++c) bad+=std::isfinite(invalid[1+size_t(r)*(out_n+3)+c]);
    }
  }
  check(cudaGraphExecDestroy(instance)); check(cudaGraphDestroy(graph));
  std::printf("KPACK_MOE_CHAIN_CUDA merged=%d tokens=%d topk=%d experts=%d splits=%d,%d,%d router=%d replays=7 bad=%zu activation_half_bad=%zu rounding_red=%zu gate_up_red=%zu\n",
      int(merged),tokens,topk,experts,sg,su,sd,router_mode,bad,activation_half_bad,rounding_red,gate_up_red);
  if (bad || !rounding_red || !gate_up_red) throw std::runtime_error("MoE chain oracle failed");
}
int main() {
  try {
    run(false,1,8,256,4,2,1); run(false,4,8,256,1,8,4);
    run(true,1,8,256,4,1,2); run(true,9,1,2,2,1,8);
    run(false,1,8,256,4,2,1,0); run(false,4,8,256,4,2,4,1); run(true,1,8,256,2,1,8,2);
    std::puts("KPACK_MOE_CHAIN_CUDA PASS cells=7 PPU_GEMM_ADMISSION=NOT_TESTED"); return 0;
  } catch (std::exception const& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
