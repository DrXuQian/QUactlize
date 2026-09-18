#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"
namespace quactlize::runtime { using Half=cutlass::half_t; }
#include "quactlize/runtime/indexed.cuh"
#include "quactlize/runtime/moe_chain.cuh"
// INSERT_WARP_ROUTER
#include "dev/moe_prepare/fast.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace quactlize::runtime;
using Shape=cute::Shape<int,int,int>;
using Stride=cute::Stride<int64_t,cute::_1,cute::_0>;
// Exact shipping router, without gather or tuple-writing code. Oracle only.
__global__ void router_control(qk_moe_plan_v1 plan) {
  __shared__ int ids[8];
  int token=int(blockIdx.x);
  auto router=plan.router;auto io=plan.gate.io;
  router.logits+=int64_t(token)*256;router.weights+=token*8;
  io.ids+=int64_t(token)*io.ids_stride;
  if(router.bias) quactlize::llama::router_256_top8<true>(router,io,ids);
  else quactlize::llama::router_256_top8<false>(router,io,ids);
}
template<bool Coalesced>
__global__ void router_warp_control(qk_moe_plan_v1 plan) {
  __shared__ int ids[8];
  quactlize::llama::router_256_top8_warp<false,Coalesced>(plan.router,plan.gate.io,ids);
}
static std::string context;
static void ck(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(context+": "+cudaGetErrorString(s)); }
static void require(bool ok,char const* why) { if(!ok) throw std::runtime_error(context+": "+why); }

template<class T> struct Buffer {
  T* base=nullptr;T* ptr=nullptr;size_t count,head;
  explicit Buffer(size_t n,bool weak=false):count(n),head(16+int(weak)) {
    ck(cudaMalloc(&base,(head+count+16)*sizeof(T)));ptr=base+head;reset();
  }
  ~Buffer() { if(base) cudaFree(base); }
  Buffer(Buffer const&)=delete;
  void reset() { ck(cudaMemset(base,0xa5,(head+count+16)*sizeof(T))); }
  void put(std::vector<T> const& v) {
    require(v.size()==count,"upload size");ck(cudaMemcpy(ptr,v.data(),count*sizeof(T),cudaMemcpyHostToDevice));
  }
  void put_on_stream(std::vector<T> const& v,cudaStream_t stream) {
    require(v.size()==count,"upload size");
    ck(cudaMemcpyAsync(ptr,v.data(),count*sizeof(T),cudaMemcpyHostToDevice,stream));
    // Untimed fixture update: order publication on the consumer stream and
    // keep pageable source storage alive through DMA completion.
    ck(cudaStreamSynchronize(stream));
  }
  std::vector<T> get() const {
    std::vector<T> v(count);ck(cudaMemcpy(v.data(),ptr,count*sizeof(T),cudaMemcpyDeviceToHost));return v;
  }
  void guards() const {
    std::vector<unsigned char> a(head*sizeof(T)),b(16*sizeof(T));
    ck(cudaMemcpy(a.data(),base,a.size(),cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(),ptr+count,b.size(),cudaMemcpyDeviceToHost));
    require(std::all_of(a.begin(),a.end(),[](auto x){return x==0xa5;}) &&
            std::all_of(b.begin(),b.end(),[](auto x){return x==0xa5;}),"buffer guard");
  }
};

template<class C> struct Projection {
  qk_moe_projection_v1 p{};
  Buffer<C> a,out;
  Buffer<float> partial;
  Buffer<int> offsets,rows,map;
  Buffer<Shape> shapes;
  Buffer<Stride> strides;
  Buffer<void*> outputs;
  Buffer<moe::Header> header;
  Buffer<moe::BlockEntry> entries;
  Projection(int m,int n,int k,int splits,bool weak):a(size_t(m)*k,weak),out(size_t(m)*n),
      partial(size_t(splits)*m*n),offsets(257),rows(256),map(m),shapes(256),
      strides(256*splits),outputs(256*splits),header(1),entries(m) {
    p.version=1;p.size=sizeof(p);p.m=m;p.n=n;p.k=k;p.experts=256;p.tile_m=8;p.splits=splits;
    p.a=a.ptr;p.output=out.ptr;p.partials=partial.ptr;p.offsets=offsets.ptr;p.rows=rows.ptr;
    p.io.row_ids=map.ptr;p.shapes=shapes.ptr;p.strides=strides.ptr;p.outputs=outputs.ptr;
    p.directory_header=header.ptr;p.directory_entries=entries.ptr;p.directory_capacity=m;
  }
  void reset() {a.reset();offsets.reset();rows.reset();map.reset();shapes.reset();strides.reset();
    outputs.reset();header.reset();entries.reset();}
  void guards() const {a.guards();offsets.guards();rows.guards();map.guards();shapes.guards();
    strides.guards();outputs.guards();header.guards();entries.guards();}
};

template<class C> struct Case {
  int tokens,k,mask,merged,router_mode;bool weak;
  Projection<C> gate,up,down;
  Buffer<float> source,logits,bias,weights;
  Buffer<int> ids;
  ComputeMoePlan<C> plan{};
  Case(int t,int kk,int mm,int merge,int router,bool w):tokens(t),k(kk),mask(mm),merged(merge),
      router_mode(router),weak(w),gate(t*8,merge?1024:512,kk,mm&1?1:2,w),
      up(t*8,512,kk,mm&2?1:4,w),down(t*8,2048,512,mm&4?1:8,w),
      source(size_t(t)*(kk+(w?17:0)),w),logits(t*256),bias(256),weights(t*8),ids(t*13) {
    static_cast<qk_moe_plan_v1&>(plan)={1,sizeof(qk_moe_plan_v1),uint32_t(merge),0,gate.p,up.p,down.p};
    plan.simt_mask=mask;
    for(auto* p:{&plan.gate,&plan.up,&plan.down}) {
      p->io.version=1;p->io.size=sizeof(p->io);p->io.tokens=t;p->io.topk=8;p->io.channels=1;
      p->io.ids_stride=13;p->io.ids=ids.ptr;p->io.a=source.ptr;
      p->io.a_row_stride=kk;p->io.a_token_stride=kk+(w?17:0);
    }
    if(router>=0) plan.router={1,sizeof(qk_llama_router_v1),int(router==1),int(router!=2),
        int(router==2),0,6.103515625e-5f,1.25f,logits.ptr,router==1?bias.ptr:nullptr,weights.ptr};
  }
  void launch(int arm,cudaStream_t stream) {
#ifdef QK_PREPARE_BASELINE
    if(arm==0 && prepare_incumbent::admitted(plan)) {
      prepare_incumbent::launch<Shape,Stride>(plan,stream);
      ck(cudaGetLastError());return;
    }
#endif
    if(arm==1 && prepare_detail::supported(plan)) {
      bool all=(mask&(merged?5:7))==(merged?5:7);
      int capacity=tokens==1?8:tokens<=4?32:64;
#define FAST(CAP,ALL) prepare_detail::once<Shape,Stride,CAP,ALL><<<1,ALL?tokens*32:256,0,stream>>>(plan)
      if(all) { if(capacity==8) {FAST(8,true);} else if(capacity==32) {FAST(32,true);} else {FAST(64,true);} }
      else { if(capacity==8) {FAST(8,false);} else if(capacity==32) {FAST(32,false);} else {FAST(64,false);} }
#undef FAST
    } else if(moe_prepare_m1_supported(plan)) moe_chain_prepare_m1<Shape,Stride><<<1,256,0,stream>>>(plan);
    else if(tokens>4) moe_chain_prepare<Shape,Stride,64><<<moe_prepare_blocks(256,tokens*8),256,0,stream>>>(plan);
    else moe_chain_prepare<Shape,Stride><<<moe_prepare_blocks(256,tokens*8),256,0,stream>>>(plan);
    ck(cudaGetLastError());
  }
  void check_projection(Projection<C>& v,qk_moe_projection_v1 const& p,
                        std::vector<int> const& order,std::vector<int> const& logical,
                        std::vector<float> const& input,bool simt,bool gathered,bool direct=false) {
    auto h=v.header.get()[0];auto map=v.map.get();auto entries=v.entries.get();
    require(map==order,"expert sorted row map");
    require(!h.status && h.tile_m==8 && h.experts==256,"directory header");
    if(direct) {
      require(!h.num_m_blocks,"SIMT-only must not publish unused TC directory");
      v.guards();return;
    }
    auto offsets=v.offsets.get(),counts=v.rows.get();auto shapes=v.shapes.get();
    auto strides=v.strides.get();auto pointers=v.outputs.get();
    int begin=0,tile=0;
    for(int e=0;e<256;++e) {
      int count=std::count(logical.begin(),logical.end(),e);
      if(!simt) {
        require(offsets[e]==begin && counts[e]==count,"expert offsets/counts");
        require(cute::get<0>(shapes[e])==count && cute::get<1>(shapes[e])==p.n && cute::get<2>(shapes[e])==p.k,"shape");
        for(int s=0;s<p.splits;++s) {
          int64_t off=(int64_t(s)*p.m+begin)*p.n;
          void* expected=p.splits==1?static_cast<void*>(static_cast<C*>(p.output)+off):static_cast<void*>(static_cast<float*>(p.partials)+off);
          require(pointers[s*256+e]==expected && cute::get<0>(strides[s*256+e])==p.n,"output pointer/stride");
        }
      }
      if(count) {
        auto entry=entries[tile];
        require(entry.expert==e && entry.row_begin==begin && entry.expert_rows==count && entry.expert_block_begin==tile,"directory entry");
        ++tile;
      }
      begin+=count;
    }
    require(h.num_m_blocks==tile,"directory length");
    if(!simt) require(offsets[256]==p.m,"terminal offset");
    if(gathered && !simt) {
      auto a=v.a.get();
      for(int r=0;r<p.m;++r) for(int col=0;col<p.k;++col) {
        C expected=C(input[size_t(order[r]/8)*plan.gate.io.a_token_stride+col]);
        require(a[size_t(r)*p.k+col].raw()==expected.raw(),"gather compute conversion");
      }
    }
    v.guards();
  }
  void activation_proof(cudaStream_t stream) {
    if((mask&(merged?5:7))!=(merged?5:7)) return;
    Buffer<float> g(size_t(tokens*8)*plan.gate.n),u(size_t(tokens*8)*plan.up.n),d(size_t(tokens*8)*plan.down.k);
    std::vector<float> gv(g.count),uv(u.count),expected(d.count);
    int n=plan.down.k;
    for(int r=0;r<tokens*8;++r) for(int col=0;col<n;++col) {
      float gate_value=float((r*13+col*3)%31-15)*.13f;
      float up_value=float((r*7+col*11)%37-18)*.17f;
      if(std::is_same<C,cutlass::bfloat16_t>::value && !r && !col) {gate_value=2.f;up_value=243383.484375f;}
      gv[size_t(r)*plan.gate.n+col]=gate_value;
      if(merged) gv[size_t(r)*plan.gate.n+n+col]=up_value;
      else uv[size_t(r)*n+col]=up_value;
      if(std::is_same<C,cutlass::bfloat16_t>::value) {gate_value=float(C(gate_value));up_value=float(C(up_value));}
      expected[size_t(r)*n+col]=(gate_value/(1.f+std::exp(-gate_value)))*up_value;
    }
    g.put(gv);u.put(uv);ck(cudaDeviceSynchronize());
    auto p=plan;p.gate.output=g.ptr;p.up.output=u.ptr;p.down.a=d.ptr;
    moe_chain_swiglu_compute<<<dim3((n+255)/256,tokens*8),256,0,stream>>>(p);
    ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));
    auto out=d.get();
    for(size_t i=0;i<out.size();++i) require(std::isfinite(out[i]) && std::abs(out[i]-expected[i])<=1e-5f*std::max(1.f,std::abs(expected[i])),"composed SIMT row-map/SwiGLU output");
    g.guards();u.guards();d.guards();
    auto positive=out;
    // Changing only the down row map must be observable at the consumer.
    auto map=down.map.get(),bad=map;std::swap(bad[0],bad[1]);
    auto expected_negative=positive;
    for(int col=0;col<n;++col)
      std::swap(expected_negative[size_t(map[0])*n+col],expected_negative[size_t(map[1])*n+col]);
    require(expected_negative!=positive,"row-map negative fixture is not observable");
    down.map.put_on_stream(bad,stream);
    require(down.map.get()==bad,"row-map negative publication differs");
    moe_chain_swiglu_compute<<<dim3((n+255)/256,tokens*8),256,0,stream>>>(p);
    ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));out=d.get();
    require(out!=positive,"wrong row-map negative accepted");
    require(out==expected_negative,"row-map negative has unexpected output placement");
    d.guards();down.map.put_on_stream(map,stream);
  }
  void correctness(bool candidate_only=false,int alias_offset=-1) {
    require(alias_offset<0 || (tokens==1 && router_mode>=0 && alias_offset+8<=256),"alias scope");
    std::vector<int> reference_ids;std::vector<float> reference_weights;
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaGraph_t graphs[2];cudaGraphExec_t exec[2];
    if(alias_offset>=0) plan.router.weights=logits.ptr+alias_offset;
    for(int arm=0;arm<2;++arm) {ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
      launch(arm,stream);ck(cudaStreamEndCapture(stream,&graphs[arm]));ck(cudaGraphInstantiate(&exec[arm],graphs[arm],nullptr,nullptr,0));}
    plan.router.weights=weights.ptr; // Separate output for the router control.
    for(int repeat=0;repeat<4;++repeat) {
      std::vector<int> in(ids.count,-123);std::vector<float> input(source.count),l(logits.count),b(bias.count);
      for(int t=0;t<tokens;++t) for(int s=0;s<8;++s) in[t*13+s]=((repeat==1?0:t*13)+s*17+repeat*7)%256;
      for(size_t i=0;i<input.size();++i) input[i]=float(int((i*19+repeat*41)%193)-96)*.07131f;
      if(repeat==3) input[0]=243383.484375f;
      for(size_t i=0;i<l.size();++i) l[i]=repeat==0?0.f:float(int((i*17+repeat*29)%263)-131)*.0625f;
      for(size_t i=0;i<b.size();++i) b[i]=repeat==0?0.f:float(int((i*31+repeat*17)%257)-128)*.00437f;
      source.put(input);logits.put(l);bias.put(b);
      if(candidate_only || alias_offset>=0) {
        ids.put(in);weights.reset();
        ck(cudaDeviceSynchronize());
        if(plan.router.version) {router_control<<<tokens,32,0,stream>>>(plan);ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));}
        reference_ids=ids.get();reference_weights=weights.get();
        if(alias_offset>=0) {
          auto aliased=plan;aliased.router.weights=logits.ptr+alias_offset;
          router_control<<<1,32,0,stream>>>(aliased);
          ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));
          auto after=logits.get();
          require(ids.get()==reference_ids && !std::memcmp(after.data()+alias_offset,
              reference_weights.data(),8*sizeof(float)),"original router snapshot alias differs");
          for(int i=0;i<256;++i) if(i<alias_offset || i>=alias_offset+8)
            require(std::memcmp(&after[i],&l[i],sizeof(float))==0,"original router alias guard");
        }
      }
      for(int arm=int(candidate_only);arm<2;++arm) {
        context="tokens="+std::to_string(tokens)+" k="+std::to_string(k)+" mask="+std::to_string(mask)+
            " merged="+std::to_string(merged)+" router="+std::to_string(router_mode)+
            " arm="+std::to_string(arm)+" repeat="+std::to_string(repeat)+
            " compute="+(std::is_same<C,cutlass::bfloat16_t>::value?"bf16":"f16")+
            " weak="+std::to_string(int(weak))+" alias="+std::to_string(alias_offset);
        gate.reset();up.reset();down.reset();ids.put(in);weights.reset();
        // As in a graph replay, the preceding projection rewrites logits.
        // Never reuse a previous router's aliased output as its next input.
        if(alias_offset>=0) logits.put(l);
        // Poison/upload use the default stream; the measured stream is
        // nonblocking. Establish the edge outside graph capture/timing.
        ck(cudaDeviceSynchronize());
        ck(cudaGraphLaunch(exec[arm],stream));ck(cudaStreamSynchronize(stream));
        auto got_ids=ids.get();auto got_weights=weights.get();
        if(alias_offset>=0) {
          auto after=logits.get();
          got_weights.assign(after.begin()+alias_offset,after.begin()+alias_offset+8);
          for(int i=0;i<256;++i) if(i<alias_offset || i>=alias_offset+8)
            require(std::memcmp(&after[i],&l[i],sizeof(float))==0,"alias wrote outside weights");
        }
        if(arm==0 && alias_offset<0) {reference_ids=got_ids;reference_weights=got_weights;}
        else {require(got_ids==reference_ids,"router IDs differ");
          if(std::memcmp(got_weights.data(),reference_weights.data(),weights.count*sizeof(float))) {
            for(size_t i=0;i<weights.count;++i) if(std::memcmp(&got_weights[i],&reference_weights[i],sizeof(float))) {
              printf("ROUTER_WEIGHT_DIFF index=%zu want=%.9g got=%.9g\n",i,reference_weights[i],got_weights[i]);break;
            }
          }
          require(std::memcmp(got_weights.data(),reference_weights.data(),weights.count*sizeof(float))==0,"router weights raw bits differ");}
        std::vector<int> logical(tokens*8),order(tokens*8);std::iota(order.begin(),order.end(),0);
        for(int t=0;t<tokens;++t) for(int s=0;s<13;++s) {
          if(s<8) {logical[t*8+s]=got_ids[t*13+s];require(logical[t*8+s]>=0 && logical[t*8+s]<256,"invalid fixture route");}
          else require(got_ids[t*13+s]==-123,"ID stride padding");
        }
        bool direct=arm==1 && (mask&(merged?5:7))==(merged?5:7);
#ifdef QK_PREPARE_BASELINE
        direct|=arm==0 && prepare_incumbent::admitted(plan);
#endif
        if(!direct) std::stable_sort(order.begin(),order.end(),[&](int x,int y){return logical[x]<logical[y];});
        check_projection(gate,plan.gate,order,logical,input,mask&1,true,direct);
        if(!merged) check_projection(up,plan.up,order,logical,input,mask&2,true,direct);
        check_projection(down,plan.down,order,logical,input,mask&4,false,direct);
        if(repeat==0 || repeat==3) activation_proof(stream);
        source.guards();ids.guards();logits.guards();bias.guards();weights.guards();
      }
      if(router_mode<0 && repeat==3) {
        for(int invalid:{-1,256,in[0]}) {
          auto bad=in;bad[1]=invalid;ids.put_on_stream(bad,stream);
          for(int arm=int(candidate_only);arm<2;++arm) {ck(cudaGraphLaunch(exec[arm],stream));ck(cudaStreamSynchronize(stream));
            require(gate.header.get()[0].status && down.header.get()[0].status,"invalid IDs not rejected");gate.guards();up.guards();down.guards();}
        }
        ids.put_on_stream(in,stream);launch(int(candidate_only),stream);ck(cudaStreamSynchronize(stream));
      }
    }
    for(int a=0;a<2;++a) {ck(cudaGraphExecDestroy(exec[a]));ck(cudaGraphDestroy(graphs[a]));}
    ck(cudaStreamDestroy(stream));
  }
  void benchmark(int profile=-1) {
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    if(profile>=0) {
      launch(profile,stream);ck(cudaStreamSynchronize(stream));
      ck(cudaProfilerStart());
      for(int i=0;i<8;++i) launch(profile,stream);
      ck(cudaStreamSynchronize(stream));ck(cudaProfilerStop());
      ck(cudaStreamDestroy(stream));return;
    }
    cudaGraph_t graph[2];cudaGraphExec_t exec[2];cudaEvent_t begin,end;
    ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    for(int arm=0;arm<2;++arm) {ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
      for(int j=0;j<64;++j) launch(arm,stream);
      ck(cudaStreamEndCapture(stream,&graph[arm]));ck(cudaGraphInstantiate(&exec[arm],graph[arm],nullptr,nullptr,0));
      for(int j=0;j<5;++j) ck(cudaGraphLaunch(exec[arm],stream));}
    ck(cudaStreamSynchronize(stream));
    std::vector<float> samples[2];
    for(int round=0;round<4;++round) for(int sample=0;sample<15;++sample) for(int turn=0;turn<2;++turn) {
      int arm=(round+turn)%2;
      ck(cudaEventRecord(begin,stream));ck(cudaGraphLaunch(exec[arm],stream));ck(cudaEventRecord(end,stream));
      ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,begin,end));samples[arm].push_back(ms*1000.f/64);
    }
    for(int arm=0;arm<2;++arm) {
      auto sorted=samples[arm];std::sort(sorted.begin(),sorted.end());
      printf("MOE_PREPARE_TIME arm=%d compute=%s tokens=%d k=%d mask=%d merged=%d router=%d weak=%d median_us=%.6f samples=[",
          arm,(std::is_same<C,Half>::value ? "f16" : "bf16"),tokens,k,mask,merged,router_mode,int(weak),(sorted[29]+sorted[30])/2);
      for(size_t i=0;i<samples[arm].size();++i) printf("%s%.6f",i?",":"",samples[arm][i]);puts("]");
      ck(cudaGraphExecDestroy(exec[arm]));ck(cudaGraphDestroy(graph[arm]));
    }
    ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));ck(cudaStreamDestroy(stream));
  }
};

static void router_edges() {
  Buffer<float> logits(256),weights(8);
  Buffer<int> ids(13);
  qk_moe_plan_v1 plan{};
  plan.router={1,sizeof(qk_llama_router_v1),0,1,0,0,6.103515625e-5f,1.25f,logits.ptr,nullptr,weights.ptr};
  plan.gate.io.ids=ids.ptr;plan.gate.io.ids_stride=13;
  int cases=0;
  for(int fixture=0;fixture<14;++fixture) for(int alias:{-1,0,8,248}) {
    context="router-edge fixture="+std::to_string(fixture)+" alias="+std::to_string(alias);
    std::vector<float> input(256);
    for(int i=0;i<256;++i) {
      if(fixture==1) input[i]=i%2 ? -0.f : 0.f;
      if(fixture==2) input[i]=float(i%7);
      if(fixture==3) input[i]=i%32==0 ? float(100-i/32) : -100.f;
      if(fixture==4) input[i]=i%13==0 ? NAN : float(i%11);
      if(fixture==5) input[i]=i==0 ? INFINITY : float(i%11);
      if(fixture==6) input[i]=-INFINITY;
      if(fixture==7) input[i]=i==127 ? 1000.f : -1000.f;
      if(fixture>=8) input[i]=float((i*17+fixture*29)%263-131)*.0625f;
    }
    std::vector<int> expected_ids;
    std::vector<float> expected_weights;
    plan.router.weights=alias<0?weights.ptr:logits.ptr+alias;
    for(int arm=0;arm<3;++arm) {
      logits.put(input);weights.reset();ids.put(std::vector<int>(13,-123));
      if(arm==0) router_control<<<1,32>>>(plan);
      else if(arm==1) router_warp_control<false><<<1,32>>>(plan);
      else router_warp_control<true><<<1,32>>>(plan);
      ck(cudaGetLastError());ck(cudaDeviceSynchronize());
      auto got_ids=ids.get();auto after=logits.get();auto got_weights=weights.get();
      if(alias>=0) got_weights.assign(after.begin()+alias,after.begin()+alias+8);
      for(int i=0;i<256;++i) if(alias<0 || i<alias || i>=alias+8)
        require(!std::memcmp(&after[i],&input[i],sizeof(float)),"router input/alias guard");
      for(int s=0;s<8;++s) {
        require(got_ids[s]>=0 && got_ids[s]<256,"router ID range");
        for(int j=0;j<s;++j) require(got_ids[s]!=got_ids[j],"router ID uniqueness");
      }
      for(int s=8;s<13;++s) require(got_ids[s]==-123,"router ID stride guard");
      if(!arm) {expected_ids=got_ids;expected_weights=got_weights;}
      else require(got_ids==expected_ids && !std::memcmp(got_weights.data(),expected_weights.data(),8*sizeof(float)),"router edge raw bits");
      logits.guards();weights.guards();ids.guards();
    }
    ++cases;
  }
  printf("MOE_ROUTER_EDGE PASS cases=%d arms=3 scope=ORDINARY_SOFTMAX_NO_BIAS\n",cases);
}

int main(int argc,char** argv) {
  try {
    if(argc==2 && !std::strcmp(argv[1],"--router-edge-check")) {router_edges();return 0;}
    bool bench=argc>1 && !std::strcmp(argv[1],"--benchmark");
    bool candidate_only=argc>1 && !std::strcmp(argv[1],"--candidate-check");
    bool alias_check=argc>1 && !std::strcmp(argv[1],"--router-alias-check");
    bool single=argc>1 && (!std::strcmp(argv[1],"--case") || !std::strcmp(argv[1],"--profile"));
    bool profile=argc>1 && !std::strcmp(argv[1],"--profile");
    if(alias_check) {
      bool alias_candidate=argc==3 && !std::strcmp(argv[2],"--candidate-only");
      require(argc==2 || alias_candidate,"--router-alias-check [--candidate-only]");
      int cases=0;
      for(int compute=0;compute<2;++compute) for(int k:{512,2048})
      for(bool merged:{false,true}) for(int mask:{0,1,3,4,5,7})
      for(int router:{0,1,2}) for(int offset:{0,8,248}) {
        if(merged&&(mask&2)) continue;
        auto run=[&](auto tag) {Case<decltype(tag)> c(1,k,mask,merged,router,false);c.correctness(alias_candidate,offset);};
        if(compute)run(cutlass::bfloat16_t{});else run(Half{});
        if(++cases%36==0) {printf("MOE_ROUTER_ALIAS_PROGRESS cases=%d\n",cases);fflush(stdout);}
      }
      printf("MOE_ROUTER_ALIAS PASS cases=%d replays=4 arms=%d scope=M1_TOP8_LOGITS_REUSE\n",cases,alias_candidate?1:2);
    } else if(single) {
      require(argc==9,"--case/--profile tokens K mask merged router bf16 weak_or_arm");
      int t=std::atoi(argv[2]),k=std::atoi(argv[3]),mask=std::atoi(argv[4]),merged=std::atoi(argv[5]),router=std::atoi(argv[6]),bf=std::atoi(argv[7]);
      int option=std::atoi(argv[8]);require(t>=1&&t<=8 && k>0 && mask>=0&&mask<=7 && (!merged||!(mask&2)),"case arguments");
      auto run=[&](auto tag) {using C=decltype(tag);Case<C> c(t,k,mask,merged,router,profile?false:bool(option));c.correctness();c.benchmark(profile?option:-1);};
      if(bf) run(cutlass::bfloat16_t{});else run(Half{});
    } else {
      int cases=0;
      for(int compute=0;compute<2;++compute) for(int t=1;t<=8;++t) for(int k:{512,2048,3072})
      for(bool merged:{false,true}) for(int mask:{0,1,3,4,5,7}) for(int router:{-1,0,1,2}) for(bool weak:{false,true}) {
        if(merged&&(mask&2))continue;
        auto run=[&](auto tag) {using C=decltype(tag);Case<C> c(t,k,mask,merged,router,weak);c.correctness(candidate_only);if(bench)c.benchmark();};
        if(compute)run(cutlass::bfloat16_t{});else run(Half{});
        if(++cases%32==0) {printf("MOE_PREPARE_PROGRESS cases=%d\n",cases);fflush(stdout);}
      }
      printf("MOE_PREPARE_CORRECTNESS PASS cases=%d replays=4 arms=%s scope=PREPARE_ONLY_NOT_GEMM\n",cases,candidate_only?"CANDIDATE_WITH_ROUTER_ORACLE":"BASELINE_AND_CANDIDATE");
    }
    return 0;
  } catch(std::exception const& e) {fprintf(stderr,"MOE_PREPARE_FAIL %s\n",e.what());return 1;}
}
