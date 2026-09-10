// Production Q8 SIMT kernel, independent logical weights and graph timing.
// This NVIDIA development test does not admit PPU timing or a model policy.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "quactlize/execution/api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

extern "C" int qkg_launch_8(qkg_call_v1 const&,qkg_config_v1 const&);
static void checked(cudaError_t s) {
  if (s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct Device {
  void* p=nullptr;
  explicit Device(size_t bytes) { checked(cudaMalloc(&p,bytes)); }
  ~Device() { cudaFree(p); }
  void put(void const* source,size_t bytes) { checked(cudaMemcpy(p,source,bytes,cudaMemcpyHostToDevice)); }
};
static void run(int n,int k,int m,int mode=0,int channels=1) {
  int e=mode?4:1,topk=mode==2?2:1;
  std::vector<int> ids(m),offsets{0,2,2,5,6},owner(m);
  for (int r=0;r<m;++r) {
    ids[r]=(r/2*3+r%2)%4;
    owner[r]=mode==0?0:mode==2?ids[r]:r<2?0:r<5?2:3;
  }
  int a_rows=mode==2?m/topk*channels:m,as=k+5,os=n+8;
  std::vector<float> a(size_t(a_rows)*as),gold(size_t(m)*n),denom(gold.size()),out(size_t(m)*os+8);
  std::vector<unsigned char> low(size_t(e)*n*k);
  std::vector<__half> d(size_t(e)*n*k/32);
  auto code=[](int expert,int col,int kk) {return (expert*17+col*31+kk*7)%256-128;};
  for (size_t i=0;i<a.size();++i) a[i]=float(int(i*13%127)-63)*.017319f;
  for (int ex=0;ex<e;++ex) for (int col=0;col<n;++col) for (int kk=0;kk<k;++kk) {
    size_t index=size_t(ex)*n*k+size_t((kk/16)*8+kk%8)*n*2+col*2+kk%16/8;
    low[index]=unsigned(code(ex,col,kk)+128);
    if (kk%32==0) d[(size_t(ex)*(k/32)+kk/32)*n+col]=__float2half(
        float((col+ex*3+kk/32)%9-4)*.00001379f);
  }
  for (int r=0;r<m;++r) for (int col=0;col<n;++col) {
    double sum=0,condition=0; int ex=owner[r],ar=mode==2?r/topk*channels+r%topk%channels:r;
    for (int kk=0;kk<k;++kk) {
      float weight=__half2float(__float2half(float(code(ex,col,kk))*__half2float(d[(size_t(ex)*(k/32)+kk/32)*n+col])));
      double term=double(__half2float(__float2half(a[size_t(ar)*as+kk])))*weight;
      sum+=term; condition+=std::abs(term);
    }
    gold[size_t(r)*n+col]=float(sum); denom[size_t(r)*n+col]=float(condition);
  }
  Device da(a.size()*4),dl(low.size()),dd(d.size()*2),di(ids.size()*4),doff(offsets.size()*4),dy(out.size()*4),dw(size_t(m)*n*4*4+32);
  da.put(a.data(),a.size()*4); dl.put(low.data(),low.size()); dd.put(d.data(),d.size()*2);
  di.put(ids.data(),ids.size()*4); doff.put(offsets.data(),offsets.size()*4);
  cudaStream_t stream; checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  qkg_call_v1 c{}; c.version=1; c.size=sizeof(c); c.qtype=8; c.n=n; c.k=k;
  c.experts=e; c.rows=m; c.mode=mode; c.input_type=QKG_F32; c.channels=channels; c.topk=topk;
  c.a_row_stride=as; c.a_token_stride=channels*as; c.ids_stride=topk; c.out_row_stride=os;
  c.a=da.p; c.low=(unsigned char*)dl.p; c.units=(unsigned char*)dd.p; c.ids=(int*)di.p; c.offsets=(int*)doff.p;
  c.output=(float*)dy.p+4; c.workspace=(float*)dw.p+4; c.workspace_bytes=size_t(m)*n*4*4; c.stream=stream;
  for (int columns:{16,32}) for (int warps:{4,8}) for (int split:{1,4}) {
    qkg_config_v1 cfg{1,sizeof(cfg),columns,warps,split};
    auto launch=[&] {if(qkg_launch_8(c,cfg)) throw std::runtime_error("Q8 launch failed");};
    checked(cudaMemsetAsync(dy.p,0xff,out.size()*4,stream));
    checked(cudaMemsetAsync(dw.p,0xff,c.workspace_bytes+32,stream)); launch();
    checked(cudaStreamSynchronize(stream));
    checked(cudaMemcpy(out.data(),dy.p,out.size()*4,cudaMemcpyDeviceToHost));
    double err=0;
    for (int r=0;r<m;++r) for (int col=0;col<n;++col) {
      float got=out[4+size_t(r)*os+col];
      if (!std::isfinite(got)) throw std::runtime_error("Q8 nonfinite output");
      err=std::max(err,std::abs(double(got)-gold[size_t(r)*n+col])/std::max(double(denom[size_t(r)*n+col]),1e-30));
    }
    if (err>=.005) throw std::runtime_error("Q8 independent dot failed");
    for (int i=0;i<4;++i) if(!std::isnan(out[i])||!std::isnan(out[out.size()-1-i])) throw std::runtime_error("Q8 guard");
    for (int r=0;r<m;++r) for (int col=n;col<os;++col)
      if(!std::isnan(out[4+size_t(r)*os+col])) throw std::runtime_error("Q8 row guard");
    // Missing codes must change the actual kernel result, not just the packer.
    checked(cudaMemsetAsync(dl.p,128,low.size(),stream)); launch(); checked(cudaStreamSynchronize(stream));
    checked(cudaMemcpy(out.data(),dy.p,out.size()*4,cudaMemcpyDeviceToHost));
    int red=0;
    for (int r=0;r<m;++r) for (int col=0;col<n;++col) {
      if (out[4+size_t(r)*os+col]!=0.f) throw std::runtime_error("Q8 zero-code product");
      red+=std::abs(gold[size_t(r)*n+col])>.005*denom[size_t(r)*n+col];
    }
    if(!red) throw std::runtime_error("Q8 missing-code oracle insensitive");
    dl.put(low.data(),low.size());
    cudaGraph_t graph; cudaGraphExec_t instance; cudaEvent_t begin,end;
    checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for(int i=0;i<32;++i) launch();
    checked(cudaStreamEndCapture(stream,&graph)); checked(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
    checked(cudaEventCreate(&begin)); checked(cudaEventCreate(&end));
    std::vector<float> times;
    for(int i=-5;i<11;++i) {
      checked(cudaEventRecord(begin,stream)); checked(cudaGraphLaunch(instance,stream));
      checked(cudaEventRecord(end,stream)); checked(cudaEventSynchronize(end));
      float ms; checked(cudaEventElapsedTime(&ms,begin,end)); if(i>=0) times.push_back(ms*1000/32);
    }
    auto sorted=times; std::sort(sorted.begin(),sorted.end());
    std::printf("Q8_SIMT_CUDA n=%d k=%d m=%d mode=%d channels=%d config=%d-%d-%d error=%.8g negative=RED median_us=%.6f samples=[",
        n,k,m,mode,channels,columns,warps,split,err,sorted[5]);
    for(size_t i=0;i<times.size();++i) std::printf("%s%.6f",i?",":"",times[i]); std::puts("]");
    checked(cudaEventDestroy(begin)); checked(cudaEventDestroy(end));
    checked(cudaGraphExecDestroy(instance)); checked(cudaGraphDestroy(graph));
  }
  checked(cudaStreamDestroy(stream));
}
int main() {
  try {
    for(int m:{1,4}) {
      run(512,2048,m); run(2048,512,m); run(2048,4096,m);
      run(4096,2048,m); run(8192,2048,m); run(1024,5120,m);
    }
    run(256,512,6,1); run(256,512,8,2,1); run(256,512,8,2,2);
    std::puts("Q8_SIMT_CUDA PASS contexts=15 configs=8 activation=FP16 source=F32 PPU_ADMISSION=NOT_TESTED");
  } catch(std::exception const& e) {std::fprintf(stderr,"FAIL %s\n",e.what()); return 1;}
}
