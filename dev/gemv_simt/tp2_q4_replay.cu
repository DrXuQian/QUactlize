// One frozen local shard. No llama graph, collective, JIT or policy selection.
#include <hggc_runtime.h>
#include "quactlize/execution/simt.h"
#ifndef QTP_SHIPPED_ONLY
#include "quactlize/execution/simt_kernel.cuh"
#endif
#include "quactlize/packing/api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#if defined(QTP_PPU) || defined(__HGGCCC_VER_MAJOR__)
#define RT(name) hggc##name
#else
#define RT(name) cuda##name
#endif
#define GPU(expr) do { auto rc=(expr); if(rc!=RT(Success)) { \
    std::fprintf(stderr,"Q4_TP2_RUNTIME operation=%s rc=%d text=%s\n",#expr,int(rc),RT(GetErrorString)(rc)); \
    return 2; } } while(0)

template<class T> bool read(FILE* f,std::vector<T>& value) {
    return std::fread(value.data(),sizeof(T),value.size(),f)==value.size();
}

template<class T> bool dump(std::string const& path,std::vector<T> const& value) {
    FILE* f=std::fopen(path.c_str(),"wbx");
    if(!f) return false;
    bool ok=std::fwrite(value.data(),sizeof(T),value.size(),f)==value.size();
    return std::fclose(f)==0 && ok;
}

uint64_t hash_bytes(void const* ptr,size_t bytes) {
    auto p=static_cast<uint8_t const*>(ptr);
    uint64_t hash=UINT64_C(14695981039346656037);
    for(size_t i=0;i<bytes;++i) hash=(hash^p[i])*UINT64_C(1099511628211);
    return hash;
}

// An independent ownership control: one thread owns a complete output.
// It uses the real canonical readers but no warp/shared reduction.
#ifndef QTP_SHIPPED_ONLY
template<int Compute>
__global__ void scalar_canonical(qkg_call_v1 c) {
    using namespace quactlize::execution;
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=c.rows*c.n) return;
    int row=index/c.n,col=index%c.n;
    auto r=q4_s1::locate(c,row);
    auto low=reinterpret_cast<uint16_t const*>(c.low+uint64_t(r.expert)*c.n*c.k/2);
    auto units=c.units+uint64_t(r.expert)*c.n*c.k/256*16;
    simt::Activation<1,Compute> a{static_cast<float const*>(c.a)+r.a};
    float total=0;
    for(int g=0;g<c.k/32;++g) {
        float2 m=simt::affine<12>(simt::load_meta<12>(units,c.n,col,g),g);
        float dot=0,sum=0;
        for(int kk=g*32;kk<(g+1)*32;kk+=4) {
            float4 v=a.values4(kk);
            float av[4]={v.x,v.y,v.z,v.w};
            for(int j=0;j<4;++j) {
                auto address=simt::Format<12>::Low::word_index(col&~1,kk+j,c.n);
                uint32_t word=*reinterpret_cast<uint32_t const*>(low+address);
                float2 q=simt::codes<12>(word,0,col&~1,kk+j);
                dot=fmaf(av[j],col&1?q.y:q.x,dot);
            }
            sum+=(v.x+v.y)+(v.z+v.w);
        }
        total+=fmaf(m.x,dot,m.y*sum);
    }
    c.output[r.output+col]=total;
}
#endif

int launch_local(qkg_simt_call_v2 const& typed,int reader) {
#ifndef QTP_SHIPPED_ONLY
    if(reader==0) return quactlize::execution::simt::launch_v2<12,0,4,4,4>(typed,1);
    auto stream=static_cast<RT(Stream_t)>(typed.call.stream);
#ifdef QTP_FIELD_AB
    if(reader==2) {
        using namespace quactlize::execution::simt;
        int blocks=typed.call.rows*(typed.call.n/16);
        // Exactly the failed V0/C4/W4/P4/S1 row; Changes=1 replaces only
        // packed scale/min extraction with the existing fixed-register reader.
        if(typed.compute_type) register_reuse<12,1,0,4,4,4,1,1><<<blocks,128,0,stream>>>(typed.call,1);
        else register_reuse<12,1,0,4,4,4,0,1><<<blocks,128,0,stream>>>(typed.call,1);
        return RT(GetLastError)()==RT(Success)?0:QKG_RUNTIME;
    }
#endif
    if(typed.compute_type) scalar_canonical<1><<<8,128,0,stream>>>(typed.call);
    else scalar_canonical<0><<<8,128,0,stream>>>(typed.call);
    return RT(GetLastError)()==RT(Success)?0:QKG_RUNTIME;
#else
    return QKG_INVALID;
#endif
}

int main(int argc,char** argv) {
    if(argc!=4 && argc!=6) return 2;
    bool shipped=!std::strcmp(argv[2],"shipped");
    if((shipped && argc!=6) || (!shipped && std::strcmp(argv[2],"fresh"))) return 2;
#ifdef QTP_SHIPPED_ONLY
    if(!shipped) return 2;
#else
    if(shipped) return 2;
#endif
    FILE* f=std::fopen(argv[1],"rb");
    if(!f) return 2;
    constexpr int N=512,K=512,E=4,Rows=2;
    std::vector<uint8_t> raw(E*N*K/256*144),low(E*N*K/2),units(E*N*K/256*16);
    std::vector<float> a(Rows*K),gold(Rows*N),got(Rows*N);
    std::vector<int32_t> ids(Rows);
    if(!read(f,raw)||!read(f,low)||!read(f,units)||!read(f,a)||!read(f,ids)||!read(f,gold)||std::fgetc(f)!=EOF) return 2;
    std::fclose(f);
    std::printf("Q4_TP2_INPUT raw=%016llx low=%016llx units=%016llx a=%016llx ids=%016llx golden=%016llx\n",
        (unsigned long long)hash_bytes(raw.data(),raw.size()),(unsigned long long)hash_bytes(low.data(),low.size()),
        (unsigned long long)hash_bytes(units.data(),units.size()),(unsigned long long)hash_bytes(a.data(),a.size()*4),
        (unsigned long long)hash_bytes(ids.data(),ids.size()*4),(unsigned long long)hash_bytes(gold.data(),gold.size()*4));
    int count=0;GPU(RT(GetDeviceCount)(&count));
    if(count!=1) { std::fprintf(stderr,"Q4_TP2_RUNTIME expected_one_visible_device got=%d\n",count);return 2; }
    RT(DeviceProp) device{};GPU(RT(GetDeviceProperties)(&device,0));
    std::printf("Q4_TP2_DEVICE name=%s warp=%d\n",device.name,device.warpSize);
    if(device.warpSize!=32) return 2;
    decltype(&quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2) pack=nullptr;
#ifndef QTP_SHIPPED_ONLY
    pack=&quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2;
#endif
    using Run=int(*)(qkg_simt_call_v2 const*,qkg_simt_config_v1 const*,quactlize_ppu_placed_arrangement_v2 const*);
    Run run=nullptr;
    if(shipped) {
        void* p=dlopen(argv[4],RTLD_NOW|RTLD_LOCAL);
        void* x=dlopen(argv[5],RTLD_NOW|RTLD_LOCAL);
        if(!p || !x) { std::fprintf(stderr,"Q4_TP2_DLOPEN %s\n",dlerror());return 2; }
        pack=reinterpret_cast<decltype(pack)>(dlsym(p,"quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2"));
        run=reinterpret_cast<Run>(dlsym(x,"quactlize_kpack_simt_run_v2"));
        if(!pack || !run) return 2;
        Dl_info pi{},xi{};
        if(!dladdr(reinterpret_cast<void*>(pack),&pi) || !dladdr(reinterpret_cast<void*>(run),&xi)) return 2;
        std::printf("Q4_TP2_LIBRARY pack=%s execution=%s\n",pi.dli_fname,xi.dli_fname);
    }
    RT(Stream_t) stream;GPU(RT(StreamCreateWithFlags)(&stream,RT(StreamNonBlocking)));
    uint8_t *dr,*dl,*du;float *da,*dy;int32_t* di;
    GPU(RT(Malloc)(&dr,raw.size()));GPU(RT(Malloc)(&dl,low.size()));GPU(RT(Malloc)(&du,units.size()));
    GPU(RT(Malloc)(&da,a.size()*4));GPU(RT(Malloc)(&dy,got.size()*4));GPU(RT(Malloc)(&di,ids.size()*4));
    GPU(RT(MemcpyAsync)(dr,raw.data(),raw.size(),RT(MemcpyHostToDevice),stream));
    GPU(RT(MemcpyAsync)(da,a.data(),a.size()*4,RT(MemcpyHostToDevice),stream));
    GPU(RT(MemcpyAsync)(di,ids.data(),ids.size()*4,RT(MemcpyHostToDevice),stream));
    quactlize_ppu_placed_arrangement_v2 arrangement{};
    if(quactlize_ppu_kpack_canonical_arrangement_v1(12,&arrangement)) return 2;
    qkg_call_v1 c{};c.version=1;c.size=sizeof(c);c.qtype=12;c.n=N;c.k=K;c.experts=E;
    c.rows=Rows;c.mode=QKG_INDEXED;c.input_type=QKG_F32;c.channels=2;c.topk=2;
    c.a_row_stride=K;c.a_token_stride=Rows*K;c.ids_stride=2;c.out_row_stride=N;
    c.a=da;c.ids=di;c.low=dl;c.units=du;c.output=dy;c.stream=stream;
    qkg_simt_config_v1 config{1,sizeof(config),0,4,4,4,1};
    int failures=0,cells=0;
    std::vector<float> legacy;
    int readers=shipped?1:2;
#ifdef QTP_FIELD_AB
    readers=3;
#endif
    for(int producer=0;producer<2;++producer) {
        if(producer==0) {
            GPU(RT(MemcpyAsync)(dl,low.data(),low.size(),RT(MemcpyHostToDevice),stream));
            GPU(RT(MemcpyAsync)(du,units.data(),units.size(),RT(MemcpyHostToDevice),stream));
        } else {
            GPU(RT(MemsetAsync)(dl,0xa5,low.size(),stream));
            GPU(RT(MemsetAsync)(du,0xa5,units.size(),stream));
            int rc=pack(dr,dl,nullptr,du,N,K,E,12,&arrangement,stream);
            if(rc) {std::fprintf(stderr,"Q4_TP2_PACK_RUNTIME rc=%d\n",rc);return 2;}
        }
        GPU(RT(StreamSynchronize)(stream));
        std::vector<uint8_t> check_low(low.size()),check_units(units.size());
        GPU(RT(Memcpy)(check_low.data(),dl,low.size(),RT(MemcpyDeviceToHost)));
        GPU(RT(Memcpy)(check_units.data(),du,units.size(),RT(MemcpyDeviceToHost)));
        size_t bad_low=0,bad_units=0;
        for(size_t i=0;i<low.size();++i) bad_low+=check_low[i]!=low[i];
        for(size_t i=0;i<units.size();++i) bad_units+=check_units[i]!=units[i];
        std::printf("Q4_TP2_PACK producer=%s low_bad=%zu units_bad=%zu\n",producer?"GPU":"HOST",bad_low,bad_units);
        if(bad_low && !dump(std::string(argv[3])+"/low-"+std::to_string(producer)+".bin",check_low)) return 2;
        if(bad_units && !dump(std::string(argv[3])+"/units-"+std::to_string(producer)+".bin",check_units)) return 2;
        failures+=bad_low||bad_units;
        for(int reader=0;reader<readers;++reader) for(int compute=0;compute<2;++compute) {
            GPU(RT(MemsetAsync)(dy,0xff,got.size()*4,stream));
            qkg_simt_call_v2 typed{2,sizeof(typed),c,compute};
            int rc=0;
            if(run) rc=run(&typed,&config,&arrangement);
            else rc=launch_local(typed,reader);
            if(rc) {std::fprintf(stderr,"Q4_TP2_COMPUTE_RUNTIME rc=%d\n",rc);return 2;}
            GPU(RT(GetLastError)());GPU(RT(StreamSynchronize)(stream));
            GPU(RT(Memcpy)(got.data(),dy,got.size()*4,RT(MemcpyDeviceToHost)));
            double error=0,denom=0,max_abs=0;size_t bad=0,nonfinite=0,first=got.size(),columns[16]{};
            for(size_t i=0;i<got.size();++i) {
                double delta=double(got[i])-gold[i];error+=delta*delta;denom+=double(gold[i])*gold[i];
                max_abs=std::max(max_abs,std::abs(delta));nonfinite+=!std::isfinite(got[i]);
                if(!std::isfinite(got[i])||std::abs(delta)>2e-6) {++bad;++columns[i%16];first=std::min(first,i);}
            }
            ++cells;failures+=bad!=0;
            const char* name=reader==2?"header32":reader?"scalar":"simt";
            if(producer==0 && reader==0 && compute==1) legacy=got;
            std::printf("Q4_TP2_LOCAL reader=%s producer=%s compute=%s variant=0 columns=4 warps=4 values=4 split=1 relative=%.9g max_abs=%.9g nonfinite=%zu bad=%zu/%zu status=%s\n",
                name,producer?"GPU":"HOST",compute?"BF16":"F16",std::sqrt(error/denom),max_abs,nonfinite,bad,got.size(),bad?"FAIL":"PASS");
            if(bad) {
                std::printf("Q4_TP2_MISMATCH first=%zu row=%zu n=%zu n_mod16=[",first,first/N,first%N);
                for(int j=0;j<16;++j) std::printf("%s%zu",j?",":"",columns[j]);
                std::puts("]");
                for(int i=0;i<8;++i) std::printf("Q4_TP2_VALUE index=%d want=%.9g got=%.9g\n",i,gold[i],got[i]);
                auto path=std::string(argv[3])+"/"+name+"-"+std::to_string(producer)+"-"+std::to_string(compute)+".bin";
                if(!dump(path,got)) return 2;
            }
        }
    }
#ifdef QTP_FIELD_AB
    // Counterfactual: zero just bits92..95 of the scale unit for N%4==0.
    // This is raw GGUF scale[6]&~15, not a layout change or another tactic.
    // Its complete output must reconstruct the historical SIMT failure.
    static_assert(quactlize::execution::simt::Format<12>::Unit::bit_of(6,0)==92);
    auto planted=units;
    for(int e=0;e<E;++e) for(int sb=0;sb<K/256;++sb) for(int n=0;n<N;n+=4)
        planted[((e*(K/256)+sb)*N+n)*16+11]&=0x0f;
    GPU(RT(MemcpyAsync)(du,planted.data(),planted.size(),RT(MemcpyHostToDevice),stream));
    GPU(RT(MemsetAsync)(dy,0xff,got.size()*4,stream));
    qkg_simt_call_v2 counterfactual{2,sizeof(counterfactual),c,QKG_COMPUTE_BF16};
    if(launch_local(counterfactual,2)) return 2;
    GPU(RT(StreamSynchronize)(stream));GPU(RT(Memcpy)(got.data(),dy,got.size()*4,RT(MemcpyDeviceToHost)));
    size_t legacy_bad=0,golden_bad=0,wrong_columns=0,nonfinite=0;double max_abs=0;
    for(size_t i=0;i<got.size();++i) {
        double delta=std::abs(double(got[i])-legacy[i]);
        max_abs=std::max(max_abs,delta);
        nonfinite+=!std::isfinite(got[i]);
        legacy_bad+=!std::isfinite(got[i])||delta>2e-6;
        bool red=!std::isfinite(got[i])||std::abs(double(got[i])-gold[i])>2e-6;
        golden_bad+=red;wrong_columns+=red&&(i%4!=0);
    }
    bool reproduced=!legacy_bad && !nonfinite && !wrong_columns && golden_bad==256;
    std::printf("Q4_TP2_FIELD_LOSS group_mod8=6 column_mod4=0 cleared_mask=15 legacy_bad=%zu golden_bad=%zu wrong_columns=%zu nonfinite=%zu max_abs=%.9g status=%s\n",
        legacy_bad,golden_bad,wrong_columns,nonfinite,max_abs,reproduced?"EXPECTED_RED":"PATTERN_DIFFERS");
    if(!dump(std::string(argv[3])+"/field-loss.bin",got)) return 2;
    GPU(RT(MemcpyAsync)(du,units.data(),units.size(),RT(MemcpyHostToDevice),stream));
#endif
    // Wrong-expert input must be rejected by the independent fixture oracle.
    std::swap(ids[0],ids[1]);GPU(RT(MemcpyAsync)(di,ids.data(),ids.size()*4,RT(MemcpyHostToDevice),stream));
    qkg_simt_call_v2 negative{2,sizeof(negative),c,QKG_COMPUTE_BF16};
    int rc=run?run(&negative,&config,&arrangement):launch_local(negative,0);
    if(rc) return 2;
    GPU(RT(StreamSynchronize)(stream));GPU(RT(Memcpy)(got.data(),dy,got.size()*4,RT(MemcpyDeviceToHost)));
    double error=0,denom=0;
    for(size_t i=0;i<got.size();++i) {error+=std::pow(double(got[i])-gold[i],2);denom+=double(gold[i])*gold[i];}
    double relative=std::sqrt(error/denom);
    bool red=std::isfinite(relative)&&relative>.02;
    std::printf("Q4_TP2_NEGATIVE kind=wrong_expert relative=%.9g status=%s\n",relative,red?"EXPECTED_RED":"FAIL");
    if(!red) return 2;
    GPU(RT(Free)(di));GPU(RT(Free)(dy));GPU(RT(Free)(da));GPU(RT(Free)(du));GPU(RT(Free)(dl));GPU(RT(Free)(dr));GPU(RT(StreamDestroy)(stream));
    std::printf("Q4_TP2_COMPLETE arm=%s cells=%d failures=%d\n",argv[2],cells,failures);
    return failures?1:0;
}
