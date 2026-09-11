// Real CUDA runner for NCU without a PyTorch installation. Host fixtures carry
// independent official-GGUF FP64 dots; timings contain only queued CUDA work.
#include "standalone_support.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc,char** argv) {
  try {
    require(argc>=4,"usage: standalone fixture.bin library.so scalar|pair|mmvq|dmmv [columns warps split] [--profile]");
    std::ifstream file(argv[1],std::ios::binary);
    Header h{}; file.read(reinterpret_cast<char*>(&h),sizeof(h));
    require(bool(file) && h.magic==UINT64_C(0x3146584D5647514B) && h.version==1 && !h.reserved,"fixture header");
    require(h.n>0 && h.n%256==0 && h.k>0 && h.k%256==0 && h.rows>0 && h.rows<=32 &&
            h.experts>0 && h.experts<=1024 && h.channels>0 && h.channels<=h.rows,"fixture dimensions");
    std::vector<unsigned char> data[8];
    for (int i=0;i<8;++i) {
        require(h.lengths[i]<UINT64_C(1)<<30,"fixture allocation bound");
        data[i].resize(h.lengths[i]);
        if (h.lengths[i]) file.read(reinterpret_cast<char*>(data[i].data()),h.lengths[i]);
        require(bool(file),"truncated fixture");
    }
    require(file.peek()==EOF && h.lengths[4]==uint64_t(h.channels)*h.k*4 &&
            h.lengths[5]==uint64_t(h.rows)*4 && h.lengths[6]==uint64_t(h.rows)*h.n*8 &&
            h.lengths[7]==h.lengths[6],"fixture payload extent");
    auto ids=reinterpret_cast<int const*>(data[5].data());
    auto gold=reinterpret_cast<double const*>(data[6].data());
    auto denom=reinterpret_cast<double const*>(data[7].data());
    for (int r=0;r<h.rows;++r) require(ids[r]>=0 && ids[r]<h.experts,"expert IDs");
    for (int64_t i=0;i<int64_t(h.rows)*h.n;++i)
        require(std::isfinite(gold[i]) && std::isfinite(denom[i]) && denom[i]>=0,"finite oracle");
    require(h.lengths[0] && h.lengths[1] && h.lengths[3],"missing weight planes");
    std::unique_ptr<Device> planes[4];
    for (int j=0;j<4;++j) if (h.lengths[j]) {
        require(h.lengths[j]%h.rows==0,"per-expert fixture slices");
        size_t slice=h.lengths[j]/h.rows;
        planes[j]=std::make_unique<Device>(slice*h.experts);
        check(cudaMemset(planes[j]->ptr,0,planes[j]->bytes));
        for (int r=0;r<h.rows;++r) planes[j]->put(data[j].data()+r*slice,slice,ids[r]*slice);
    }
    Device a(data[4].size()),id(data[5].size()),out((size_t(h.rows)*h.n+8)*4),
           workspace(size_t(h.rows)*h.n*8*4+32),quantized(size_t(h.channels)*(h.k/32)*36);
    a.put(data[4].data(),data[4].size()); id.put(data[5].data(),data[5].size());
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    qkg_call_v1 c{}; c.version=1; c.size=sizeof(c); c.qtype=h.q; c.n=h.n; c.k=h.k; c.experts=h.experts;
    c.rows=h.rows; c.mode=h.experts>1?QKG_INDEXED:QKG_DENSE; c.input_type=QKG_F32;
    c.channels=h.channels; c.topk=h.experts>1?h.rows:1; c.a_row_stride=h.k; c.a_token_stride=int64_t(h.channels)*h.k;
    c.ids_stride=h.rows; c.out_row_stride=h.n; c.a=a.ptr;
    c.low=static_cast<uint8_t*>(planes[1]->ptr); c.high=planes[2]?static_cast<uint8_t*>(planes[2]->ptr):nullptr;
    c.units=static_cast<uint8_t*>(planes[3]->ptr); c.ids=h.experts>1?static_cast<int32_t*>(id.ptr):nullptr;
    c.output=static_cast<float*>(out.ptr)+4; c.workspace=static_cast<float*>(workspace.ptr)+4; c.stream=stream;
    std::string kind=argv[3]; bool profile=std::string(argv[argc-1])=="--profile";
    qkg_config_v1 cfg{1,sizeof(cfg),16,4,1};
    if (argc>=7) { cfg.columns=std::stoi(argv[4]); cfg.warps=std::stoi(argv[5]); cfg.split=std::stoi(argv[6]); }
    Library library(argv[2]); std::function<int()> run;
    if (kind=="scalar" || kind=="pair") {
        using Query=int(*)(qkg_call_v1 const*,qkg_config_v1 const*,quactlize_ppu_placed_arrangement_v2 const*,qkg_sizes_v1*);
        using Run=int(*)(qkg_call_v1 const*,qkg_config_v1 const*,quactlize_ppu_placed_arrangement_v2 const*);
        auto query=library.symbol<Query>(kind=="pair"?"quactlize_kpack_gemv_pair_query_v1":"quactlize_kpack_gemv_query_v1");
        auto launch=library.symbol<Run>(kind=="pair"?"quactlize_kpack_gemv_pair_run_v1":"quactlize_kpack_gemv_run_v1");
        qkg_sizes_v1 sizes{};
        require(query(&c,&cfg,&h.arrangement,&sizes)==0,"GEMV query");
        require(sizes.workspace_bytes<=workspace.bytes-32,"workspace capacity");
        require(sizes.low_bytes==planes[1]->bytes && sizes.units_bytes==planes[3]->bytes &&
                sizes.high_bytes==(planes[2]?planes[2]->bytes:0),"weight plane sizes");
        c.workspace_bytes=sizes.workspace_bytes;
        run=[&,launch]{return launch(&c,&cfg,&h.arrangement);};
    } else {
        require(kind=="mmvq" || kind=="dmmv","unknown kind");
        using Run=int(*)(int,int,int,int,int,int,void const*,float const*,int const*,float*,void*,void*);
        auto launch=library.symbol<Run>("llama_reference_run");
        run=[&,launch]{return launch(kind=="dmmv",h.q,h.n,h.k,h.rows,h.channels,planes[0]->ptr,
            static_cast<float*>(a.ptr),c.ids,c.output,quantized.ptr,stream);};
    }
    auto launch=[&]{int rc=run(); if(rc) throw std::runtime_error("launch rc="+std::to_string(rc));};
    std::vector<float> got(out.bytes/4);
    auto download=[&]{check(cudaStreamSynchronize(stream));check(cudaMemcpy(got.data(),out.ptr,out.bytes,cudaMemcpyDeviceToHost));};
    auto error=[&] {
        double worst=0;
        for(size_t i=0;i<size_t(h.rows)*h.n;++i) {
            require(std::isfinite(got[4+i]),"nonfinite output");
            worst=std::max(worst,std::abs(double(got[4+i])-gold[i])/std::max(denom[i],1.e-30));
        }
        return worst;
    };
    check(cudaMemsetAsync(out.ptr,0xff,out.bytes,stream));
    check(cudaMemsetAsync(workspace.ptr,0xff,workspace.bytes,stream));
    launch(); download(); double err=error(); require(err<.005,"independent GGUF oracle");
    for(int j=0;j<4;++j) require(std::isnan(got[j]) && std::isnan(got[got.size()-1-j]),"output guard");
    if (kind=="scalar" || kind=="pair") {
        std::vector<float> parts(workspace.bytes/4);
        check(cudaMemcpy(parts.data(),workspace.ptr,workspace.bytes,cudaMemcpyDeviceToHost));
        for(int j=0;j<4;++j) require(std::isnan(parts[j]) && std::isnan(parts[parts.size()-1-j]),"workspace guard");
        for(size_t j=4+c.workspace_bytes/4;j<parts.size();++j)
            require(std::isnan(parts[j]),"unused workspace overwritten");
        if (cfg.split>1) for (int r=0;r<h.rows;++r) for(int n=0;n<h.n;++n) {
            float sum=0;
            for(int s=0;s<cfg.split;++s) sum+=parts[4+(int64_t(r)*cfg.split+s)*h.n+n];
            require(std::memcmp(&sum,&got[4+int64_t(r)*h.n+n],4)==0,"ordered reducer");
        }
    }
    if (!profile && h.experts>1) {
        std::vector<int> wrong(h.rows,ids[0]); id.put(wrong.data(),data[5].size());
        launch(); download(); require(error()>.005,"wrong-expert negative insensitive");
        id.put(data[5].data(),data[5].size());
    }
    if (!profile && (kind=="scalar" || kind=="pair")) {
        check(cudaMemsetAsync(planes[1]->ptr,h.q==8?128:0,planes[1]->bytes,stream));
        launch(); download(); require(error()>.005,"missing-code negative insensitive");
        size_t slice=h.lengths[1]/h.rows;
        for(int r=0;r<h.rows;++r) planes[1]->put(data[1].data()+r*slice,slice,ids[r]*slice);
    }
    for(int j=0;j<5;++j) launch(); check(cudaStreamSynchronize(stream));
    std::vector<float> times;
    if (!profile) {
        cudaGraph_t graph;cudaGraphExec_t instance;cudaEvent_t begin,end;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        for(int j=0;j<32;++j) launch();
        check(cudaStreamEndCapture(stream,&graph)); check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
        check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
        for(int j=-5;j<15;++j) {
            check(cudaEventRecord(begin,stream));check(cudaGraphLaunch(instance,stream));
            check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
            float ms;check(cudaEventElapsedTime(&ms,begin,end)); if(j>=0)times.push_back(ms*1000/32);
        }
        check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
        check(cudaGraphExecDestroy(instance));check(cudaGraphDestroy(graph));
        download();require(error()<.005,"post-timing oracle");
    }
    int device;cudaDeviceProp prop{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
    auto sorted=times; std::sort(sorted.begin(),sorted.end());
    std::printf("GEMV_STANDALONE q=%d shape=%dx%dx%d experts=%d channels=%d kind=%s config=%d-%d-%d error=%.8g status=PASS sm=%d median_us=%.6f samples=[",
        h.q,h.rows,h.n,h.k,h.experts,h.channels,kind.c_str(),cfg.columns,cfg.warps,cfg.split,err,prop.multiProcessorCount,
        sorted.empty()?0.f:sorted[sorted.size()/2]);
    for(size_t i=0;i<times.size();++i)std::printf("%s%.6f",i?",":"",times[i]); std::puts("]");
    check(cudaStreamDestroy(stream)); return 0;
  } catch(std::exception const& e) { std::fprintf(stderr,"GEMV_STANDALONE FAIL: %s\n",e.what());return 1; }
}
