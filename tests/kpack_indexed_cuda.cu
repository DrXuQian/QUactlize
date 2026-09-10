// SIMT adapter proof only. Does not emulate the PPU GEMM or admit its timing.
#include <cuda_runtime.h>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#if !defined(__HGGCCC__)
using hggcStream_t = cudaStream_t;
#endif
#include "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"
namespace quactlize::runtime { using Half=cutlass::half_t; }
#include "quactlize/runtime/indexed.cuh"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

void check(cudaError_t rc) { if (rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
template<class T> struct Buffer {
  T* ptr=nullptr;
  size_t count;
  explicit Buffer(size_t n) : count(n) { check(cudaMalloc(&ptr,sizeof(T)*n)); }
  ~Buffer() { cudaFree(ptr); }
  void put(std::vector<T> const& v) { if (v.size()!=count) throw std::runtime_error("upload size");
    check(cudaMemcpy(ptr,v.data(),sizeof(T)*count,cudaMemcpyHostToDevice)); }
  std::vector<T> get() const { std::vector<T> v(count);
    check(cudaMemcpy(v.data(),ptr,sizeof(T)*count,cudaMemcpyDeviceToHost)); return v; }
};
template<class F> float time_us(F fn) {
  cudaEvent_t a,b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
  cudaGraph_t graph; cudaGraphExec_t instance;
  check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
  for (int i=0;i<101;++i) fn();
  check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
  check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
  for (int i=0;i<5;++i) check(cudaGraphLaunch(instance,cudaStreamPerThread));
  check(cudaStreamSynchronize(cudaStreamPerThread));
  check(cudaEventRecord(a,cudaStreamPerThread));
  check(cudaGraphLaunch(instance,cudaStreamPerThread));
  check(cudaEventRecord(b,cudaStreamPerThread)); check(cudaEventSynchronize(b));
  float ms; check(cudaEventElapsedTime(&ms,a,b));
  check(cudaGraphExecDestroy(instance)); check(cudaGraphDestroy(graph));
  check(cudaEventDestroy(a)); check(cudaEventDestroy(b)); return ms*1000/101;
}

template<int TM,int S> void run(int tokens,int topk,int experts,int channels,int n,int k) {
  using namespace quactlize::runtime;
  using Shape=cute::Shape<int,int,int>;
  using Stride=cute::Stride<int64_t,cute::_1,cute::_0>;
  using Output=std::conditional_t<S==1,Half,float>;
  int m=tokens*topk, ids_stride=topk+5, a_row=k+7, a_token=channels*a_row+13, out_row=n+3;
  Buffer<int> ids(tokens*ids_stride), offsets(experts+1), rows(experts), row_ids(m);
  Buffer<Half> gathered(size_t(m)*k), completed(size_t(m)*n);
  Buffer<float> a(size_t(tokens)*a_token), out(size_t(m)*out_row+2), partials(size_t(S)*m*n);
  Buffer<Shape> shapes(experts);
  Buffer<Stride> strides(experts*S);
  Buffer<Output*> outputs(experts*S);
  Buffer<quactlize::moe_directory::Header> header(1);
  Buffer<quactlize::moe_directory::BlockEntry> entries(m);
  quactlize::moe_directory::View view{header.ptr,entries.ptr,m};
  qk_llama_indexed_v1 io{1,sizeof(io),tokens,topk,channels,0,ids_stride,a_row,a_token,out_row,
      ids.ptr,a.ptr,out.ptr+1,row_ids.ptr};
  Output* destination;
  if constexpr (S==1) destination=completed.ptr;
  else destination=partials.ptr;
  auto prepare=[&] { indexed_prepare<TM><<<dim3(std::min((k+255)/256,32),m),256,0,cudaStreamPerThread>>>(io,
      gathered.ptr,offsets.ptr,rows.ptr,shapes.ptr,outputs.ptr,strides.ptr,destination,
      m,n,k,experts,S,view); check(cudaGetLastError()); };
  auto finish=[&] { indexed_finish<S><<<dim3(std::min((n+255)/256,32),m),256,0,cudaStreamPerThread>>>(
      partials.ptr,completed.ptr,out.ptr+1,row_ids.ptr,m,n,out_row,header.ptr); check(cudaGetLastError()); };
  std::vector<float> hp(partials.count);
  std::vector<Half> hc(completed.count);
  for (size_t i=0;i<hp.size();++i) hp[i]=float(int(i*17%113)-56)*0.00371f;
  for (size_t i=0;i<hc.size();++i) hc[i]=Half(float(int(i*13%59)-29)*0.00811f);
  partials.put(hp); completed.put(hc);
  std::vector<int> initial(ids.count,0);
  for (int t=0;t<tokens;++t) for (int s=0;s<topk;++s) initial[t*ids_stride+s]=s;
  ids.put(initial); a.put(std::vector<float>(a.count,1.f));
  cudaGraph_t graph; cudaGraphExec_t instance;
  prepare(); finish(); check(cudaDeviceSynchronize()); // Fully initialized before capture.
  check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
  // Capture the production SIMT functions, not a recomputed test kernel.
  indexed_prepare<TM><<<dim3(std::min((k+255)/256,32),m),256,0,cudaStreamPerThread>>>(io,
      gathered.ptr,offsets.ptr,rows.ptr,shapes.ptr,outputs.ptr,strides.ptr,destination,m,n,k,experts,S,view);
  indexed_finish<S><<<dim3(std::min((n+255)/256,32),m),256,0,cudaStreamPerThread>>>(
      partials.ptr,completed.ptr,out.ptr+1,row_ids.ptr,m,n,out_row,header.ptr);
  check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
  check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
  size_t bad=0, negative=0;
  std::vector<float> expected(out.count,-123.f);
  for (int replay=0;replay<7;++replay) {
    std::vector<int> hi(ids.count,-1), logical(m), ordered(m);
    for (int t=0;t<tokens;++t) for (int slot=0;slot<topk;++slot) {
      int e=(t*3+slot*5+replay*7)%experts;
      hi[t*ids_stride+slot]=logical[t*topk+slot]=e;
    }
    std::iota(ordered.begin(),ordered.end(),0);
    std::stable_sort(ordered.begin(),ordered.end(),[&](int x,int y){return logical[x]<logical[y];});
    std::vector<float> ha(a.count);
    for (size_t i=0;i<ha.size();++i) ha[i]=float(int((i*7+replay*31)%127)-63)*0.02013f;
    ids.put(hi); a.put(ha); out.put(std::vector<float>(out.count,-123.f));
    check(cudaMemset(gathered.ptr,0xA5,gathered.count*sizeof(Half)));
    check(cudaGraphLaunch(instance,cudaStreamPerThread)); check(cudaDeviceSynchronize());
    auto got_a=gathered.get(); auto got_out=out.get(); auto map=row_ids.get();
    auto got_offsets=offsets.get(); auto got_rows=rows.get(); auto got_shapes=shapes.get();
    auto got_pointers=outputs.get(); auto got_strides=strides.get();
    auto got_header=header.get()[0]; auto got_entries=entries.get();
    int prefix=0, tile_prefix=0;
    for (int e=0;e<experts;++e) {
      int count=int(std::count(logical.begin(),logical.end(),e));
      bad+=got_offsets[e]!=prefix; bad+=got_rows[e]!=count;
      bad+=cute::get<0>(got_shapes[e])!=count || cute::get<1>(got_shapes[e])!=n || cute::get<2>(got_shapes[e])!=k;
      for (int s=0;s<S;++s) {
        bad+=got_pointers[s*experts+e]!=destination+(int64_t(s)*m+prefix)*n;
        bad+=cute::get<0>(got_strides[s*experts+e])!=n;
      }
      for (int j=0;j<(count+TM-1)/TM;++j) {
        auto got=got_entries[tile_prefix+j];
        bad+=got.expert!=e || got.expert_rows!=count || got.row_begin!=prefix || got.expert_block_begin!=tile_prefix;
      }
      prefix+=count; tile_prefix+=(count+TM-1)/TM;
    }
    bad+=got_offsets[experts]!=m || got_header.status!=0 || got_header.num_m_blocks!=tile_prefix || got_header.tile_m!=TM;
    std::fill(expected.begin(),expected.end(),-123.f);
    for (int r=0;r<m;++r) {
      int original=ordered[r]; bad+=map[r]!=original;
      size_t from=size_t(original/topk)*a_token+(original%topk%channels)*a_row;
      for (int col=0;col<k;++col) bad+=got_a[size_t(r)*k+col].raw()!=Half(ha[from+col]).raw();
      for (int col=0;col<n;++col) {
        float value;
        if constexpr (S==1) value=float(hc[size_t(r)*n+col]);
        else {
          float sum=0;
          for (int s=0;s<S;++s) sum+=hp[(size_t(s)*m+r)*n+col];
          value=float(Half(sum));
          // The negative proves that dropping FP16 rounding changes this fixture.
          negative+=value!=sum;
        }
        expected[1+size_t(original)*out_row+col]=value;
      }
    }
    for (size_t i=0;i<expected.size();++i) bad+=got_out[i]!=expected[i];
  }
  if (S>1 && !negative) throw std::runtime_error("rounding negative not discriminating");
  auto correct_map=row_ids.get(), rotated=correct_map;
  if (m>1) {
    std::rotate(rotated.begin(),rotated.begin()+1,rotated.end());
    row_ids.put(rotated); finish(); check(cudaDeviceSynchronize());
    if (out.get()==expected) throw std::runtime_error("scatter permutation negative missed");
    row_ids.put(correct_map);
  }
  if (topk>1) {
    auto original=ids.get(), duplicate=original;
    duplicate[1]=duplicate[0]; ids.put(duplicate);
    check(cudaGraphLaunch(instance,cudaStreamPerThread)); check(cudaDeviceSynchronize());
    if (!header.get()[0].status) throw std::runtime_error("duplicate router accepted");
    auto invalid=out.get();
    for (int row=0;row<m;++row) for (int col=0;col<n;++col) {
      float value=invalid[1+size_t(row)*out_row+col];
      if (value==value) throw std::runtime_error("invalid router output remained finite");
    }
    ids.put(original); prepare(); finish(); check(cudaDeviceSynchronize());
  }
  float prep_us=time_us(prepare), post_us=time_us(finish);
  std::printf("KPACK_INDEXED_CUDA tm=%d splits=%d tokens=%d topk=%d experts=%d channels=%d n=%d k=%d bad=%zu rounding_red=%zu replays=7 prep_us=%.3f finish_us=%.3f scope=SIMT_ONLY\n",
      TM,S,tokens,topk,experts,channels,n,k,bad,negative,prep_us,post_us);
  check(cudaGraphExecDestroy(instance)); check(cudaGraphDestroy(graph));
  if (bad) throw std::runtime_error("independent indexed oracle failed");
}
int main() {
  try {
    run<8,1>(1,8,256,1,512,2048); run<8,2>(1,8,256,1,512,2048);
    run<8,4>(1,8,256,1,512,2048); run<8,8>(1,8,256,8,2048,512);
    run<8,4>(4,8,256,8,512,2048); run<8,2>(9,1,2,1,512,512);
    run<16,4>(1,8,256,1,512,2048); run<16,2>(4,8,256,8,512,2048);
    std::puts("KPACK_INDEXED_CUDA verdict=PASS cells=8 PPU_GEMM_ADMISSION=NOT_TESTED");
    return 0;
  } catch (std::exception const& e) { std::fprintf(stderr,"KPACK_INDEXED_CUDA FAIL %s\n",e.what()); return 1; }
}
