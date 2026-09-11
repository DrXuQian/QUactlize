// Profile one complete, warmed Q4 call. Use application replay and no
// profiler cache flush so each pass recreates the requested cache state.
#include "standalone_support.hpp"
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

int main(int argc,char** argv) {
  try {
    require(argc==10,"usage: profile_xplane fixture.bin xplane.so kpack.so xplane|kpack C W S warm|rotating profile|timing");
    std::ifstream input(argv[1],std::ios::binary);
    Header h{}; input.read(reinterpret_cast<char*>(&h),sizeof(h));
    require(bool(input) && h.magic==UINT64_C(0x3146584D5647514B) && h.version==1 && !h.reserved,
            "fixture header");
    require(h.q==12 && h.n>0 && h.n%256==0 && h.k>=1024 && h.k<=8192 && h.k%1024==0 &&
            h.experts==1 && h.rows==1 && h.channels==1,"dense Q4 fixture");
    uint64_t const nk=uint64_t(h.n)*h.k;
    require(h.lengths[0]==nk/256*144 && h.lengths[1]==nk/2 && !h.lengths[2] &&
            h.lengths[3]==nk/256*16 && h.lengths[4]==uint64_t(h.k)*4 && h.lengths[5]==4 &&
            h.lengths[6]==uint64_t(h.n)*8 && h.lengths[7]==h.lengths[6],"fixture lengths");
    std::vector<uint8_t> data[8];
    for(int j=0;j<8;++j) {
        data[j].resize(h.lengths[j]);
        if(h.lengths[j]) input.read(reinterpret_cast<char*>(data[j].data()),h.lengths[j]);
        require(bool(input),"truncated fixture");
    }
    require(input.peek()==EOF,"fixture trailing bytes");
    Library xp(argv[2]),kp(argv[3]);
    using Pack=int(*)(int,int,void const*,void*,void*);
    using Xrun=int(*)(int,int,int,int,void const*,void const*,void const*,void*,void*);
    using Query=int(*)(qkg_call_v1 const*,qkg_config_v1 const*,quactlize_ppu_placed_arrangement_v2 const*,qkg_sizes_v1*);
    using Krun=int(*)(qkg_call_v1 const*,qkg_config_v1 const*,quactlize_ppu_placed_arrangement_v2 const*);
    auto pack=xp.symbol<Pack>("q4_xplane_pack");
    auto xrun=xp.symbol<Xrun>("q4_xplane_run");
    auto query=kp.symbol<Query>("quactlize_kpack_gemv_pair_query_v1");
    auto krun=kp.symbol<Krun>("quactlize_kpack_gemv_pair_run_v1");
    std::vector<uint8_t> xlow(nk/2),xunits(h.lengths[3]);
    require(pack(h.n,h.k,data[0].data(),xlow.data(),xunits.data())==0,"Xplane round trip");
    require(xunits==data[3],"metadata identity");
    std::string arm=argv[4],mode=argv[8],purpose=argv[9];
    require(arm=="xplane" || arm=="kpack","arm");
    require(mode=="warm" || mode=="rotating","cache mode");
    require(purpose=="profile" || purpose=="timing" || purpose=="batch","purpose");
    std::vector<qkg_config_v1> configurations;
    if(purpose=="batch") {
        std::string list=argv[5];size_t begin=0;
        while(begin<list.size()) {
            size_t end=list.find(',',begin);
            auto item=list.substr(begin,end==std::string::npos ? end : end-begin);
            int c=0,w=0,s=0;char tail;
            require(std::sscanf(item.c_str(),"%d:%d:%d%c",&c,&w,&s,&tail)==3,"batch recipe");
            require(c>0 && w>0 && s>0,"positive recipe");
            configurations.push_back({1,sizeof(qkg_config_v1),c,w,s});
            if(end==std::string::npos) break;
            begin=end+1;
        }
        require(!configurations.empty() && configurations.size()<=128,"bounded recipe batch");
    } else configurations.push_back({1,sizeof(qkg_config_v1),std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7])});
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,0));
    require(prop.l2CacheSize>0,"device L2 size");
    uint64_t const one=h.lengths[1]+h.lengths[3];
    int const copies=mode=="warm" ? 1 : int((9*uint64_t(prop.l2CacheSize)+4*one-1)/(4*one));
    Device low(h.lengths[1]*copies),units(h.lengths[3]*copies),a(h.k*2),out((h.n+8)*4),ws((8*h.n+8)*4);
    for(int i=0;i<copies;++i) {
        low.put(arm=="xplane" ? xlow.data() : data[1].data(),h.lengths[1],i*h.lengths[1]);
        units.put(data[3].data(),h.lengths[3],i*h.lengths[3]);
    }
    std::vector<half> ah(h.k);
    for(int i=0;i<h.k;++i) {
        float f; std::memcpy(&f,data[4].data()+4*i,4);
        ah[i]=__float2half_rn(f);
        require(std::isfinite(f) && __half2float(ah[i])==f,"F16-exact A");
    }
    a.put(ah.data(),h.k*2);
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    qkg_call_v1 call{}; call.version=1;call.size=sizeof(call);call.qtype=12;call.n=h.n;call.k=h.k;
    call.experts=1;call.rows=1;call.mode=QKG_DENSE;call.input_type=QKG_F16;call.channels=1;call.topk=1;
    call.a_row_stride=h.k;call.a_token_stride=h.k;call.ids_stride=1;call.out_row_stride=h.n;
    call.a=a.ptr;call.low=static_cast<uint8_t*>(low.ptr);call.units=static_cast<uint8_t*>(units.ptr);
    call.output=static_cast<float*>(out.ptr)+4;call.stream=stream;
    for(auto const& cfg:configurations) {
    require(arm!="xplane" || cfg.split==1,"Xplane has no inter-CTA Split-K");
    qkg_sizes_v1 sizes{};
    if(arm=="kpack") {
        require(query(&call,&cfg,&h.arrangement,&sizes)==0,"K-pack query");
        require(sizes.workspace_bytes<=ws.bytes-32,"workspace size");
        call.workspace=sizes.workspace_bytes ? static_cast<float*>(ws.ptr)+4 : nullptr;
        call.workspace_bytes=sizes.workspace_bytes;
    }
    auto run=[&](int copy) {
        call.low=static_cast<uint8_t*>(low.ptr)+copy*h.lengths[1];
        call.units=static_cast<uint8_t*>(units.ptr)+copy*h.lengths[3];
        int rc=arm=="xplane" ? xrun(cfg.columns,cfg.warps,h.n,h.k,a.ptr,call.low,call.units,call.output,stream)
                            : krun(&call,&cfg,&h.arrangement);
        require(rc==0,"kernel launch");
    };
    auto validate=[&] {
        check(cudaStreamSynchronize(stream));
        std::vector<float> got(h.n+8);check(cudaMemcpy(got.data(),out.ptr,out.bytes,cudaMemcpyDeviceToHost));
        double worst=0;
        for(int i=0;i<4;++i) require(std::isnan(got[i]) && std::isnan(got[h.n+4+i]),"output guards");
        for(int i=0;i<h.n;++i) {
            double gold,denom;
            std::memcpy(&gold,data[6].data()+8*i,8);std::memcpy(&denom,data[7].data()+8*i,8);
            require(std::isfinite(gold) && std::isfinite(denom) && denom>=0 && std::isfinite(got[4+i]),"finite output/oracle");
            worst=std::max(worst,std::abs(double(got[4+i])-gold)/std::max(denom,1e-30));
        }
        require(worst<.005,"independent GGUF oracle");
        return worst;
    };
    check(cudaMemsetAsync(out.ptr,0xff,out.bytes,stream));
    check(cudaMemsetAsync(ws.ptr,0xff,ws.bytes,stream));
    run(0);double err=validate();
    for(int pass=0;pass<5;++pass) for(int i=0;i<copies;++i) run(i);
    check(cudaStreamSynchronize(stream));
    std::vector<float> times;
    if(purpose=="profile") {
        check(cudaProfilerStart());
        run(0);
        check(cudaStreamSynchronize(stream));
        check(cudaProfilerStop());
    } else {
        cudaGraph_t graph;cudaGraphExec_t instance;cudaEvent_t begin,end;
        int const calls=std::max(32,2*copies);
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        for(int i=0;i<calls;++i) run(i%copies);
        check(cudaStreamEndCapture(stream,&graph));
        check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
        check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
        for(int s=-5;s<15;++s) {
            check(cudaEventRecord(begin,stream));check(cudaGraphLaunch(instance,stream));
            check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
            float ms;check(cudaEventElapsedTime(&ms,begin,end));
            require(std::isfinite(ms) && ms>0,"event duration");
            if(s>=0) times.push_back(ms*1000/calls);
        }
        check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
        check(cudaGraphExecDestroy(instance));check(cudaGraphDestroy(graph));
    }
    err=std::max(err,validate());
    auto sorted=times;std::sort(sorted.begin(),sorted.end());
    std::printf("Q4_LAYOUT_PROFILE arm=%s config=%d-%d-%d shape=1x%dx%d sm=%d L2_bytes=%d copies=%d mode=%s purpose=%s error=%.9g median_us=%.6f status=PASS samples=[",
        arm.c_str(),cfg.columns,cfg.warps,cfg.split,h.n,h.k,prop.multiProcessorCount,prop.l2CacheSize,copies,
        mode.c_str(),purpose=="batch" ? "timing" : purpose.c_str(),err,sorted.empty()?0.f:sorted[sorted.size()/2]);
    for(size_t i=0;i<times.size();++i) std::printf("%s%.6f",i?",":"",times[i]);
    std::puts("]");std::fflush(stdout);
    }
    check(cudaStreamDestroy(stream));return 0;
  } catch(std::exception const& e) {
    std::fprintf(stderr,"Q4_LAYOUT_PROFILE FAIL: %s\n",e.what());return 1;
  }
}
