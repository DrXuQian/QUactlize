#include "quactlize/prefill/dense_api.h"
#include "quactlize/prefill/dynamic.hpp"
#include "quactlize/prefill/dense_helpers.hpp"
#include <hggc_runtime_api.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <thread>

extern "C" hggcError_t hggcGetDevice(int* p){*p=0;return hggcSuccess;}
static const char* test_name="PPU-ZW810";
static int test_cu=72;
extern "C" hggcError_t hggcGetDeviceProperties(hggcDeviceProp* p,int){
    *p={};std::strcpy(p->name,test_name);p->multiProcessorCount=test_cu;return hggcSuccess;
}
extern "C" hggcError_t hggcStreamIsCapturing(hggcStream_t,hggcStreamCaptureStatus* p){*p=hggcStreamCaptureStatusNone;return hggcSuccess;}
extern "C" hggcError_t hggcMemsetAsync(void* p,int value,size_t bytes,hggcStream_t){std::memset(p,value,bytes);return hggcSuccess;}
extern "C" int quactlize_dense_io_v1(int storage,int input,int,void const* src,void* dst,int,int,int64_t,int const*,void*) {
    *static_cast<float*>(dst)=*static_cast<float const*>(src)+(input?100*storage:0);return 0;
}
extern "C" int quactlize_kpack_dequant_dense_v1(qzd_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const*,int storage,int variant){
    assert(qkp_dense_dequant_variant_supports(c->qtype,variant));
    *static_cast<float*>(c->output)=*static_cast<float const*>(c->low)+100*storage;return 0;
}
extern "C" int quactlize_kpack_dequant_v1(qzd_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const* a){return quactlize_kpack_dequant_dense_v1(c,a,1,0);}
extern "C" int qkp_stage_input(qkp_call_v1 const* c,quactlize::prefill::Layout const* l,void*){
    *reinterpret_cast<float*>(static_cast<char*>(c->workspace)+l->a)=c->a[0]+100;return 0;
}
extern "C" int qkp_stage_finish(qkp_call_v1 const* c,quactlize::prefill::Layout const* l,void*){
    c->output[0]=*reinterpret_cast<float*>(static_cast<char*>(c->workspace)+l->out);return 0;
}
int main(int argc,char** argv){
    assert(argc==2);using namespace quactlize::prefill;
    for(int q:{8,10,11,12,13,14}) {
    qkp_call_v1 shape{};shape.version=1;shape.size=sizeof(shape);shape.m=shape.a_rows=128;
    shape.a_stride=512;shape.output_stride=256;shape.weight={1,sizeof(qzd_call_v1),q,256,512,1,1,5};
    auto arr=q==8?q8_kpack2::arrangement():q==12?ppu_arrangements::q4_kpack4_transpose_v1():
        ppu_arrangements::kquant_kpack_transpose_v1(q);Layout l{};
    assert(!layout(shape,&arr,l,true));
    if(q==8)assert(layout(shape,&arr,l)==QKG_SHAPE);
    uint64_t bytes=0;assert(!quactlize_kpack_dense_full_workspace_v1(&shape,&arr,&bytes));
    assert(bytes==dynamic_bytes(shape,l));
    qkp_call_v1 calls[2]={shape,shape};
    for(int i=0;i<2;++i){auto& c=calls[i];
        c.a=static_cast<float*>(std::aligned_alloc(256,128*512*4));c.output=static_cast<float*>(std::aligned_alloc(256,128*256*4));
        c.workspace_bytes=dynamic_bytes(c,l);c.workspace=std::aligned_alloc(256,c.workspace_bytes);
        qkg_sizes_v1 sizes{};assert(!quactlize::execution::sizes(q,256,512,1,&arr,sizes));
        c.weight.low_bytes=sizes.low_bytes;c.weight.unit_bytes=sizes.units_bytes;
        c.weight.high_bytes=sizes.high_bytes;
        if(sizes.high_bytes)c.weight.high=std::aligned_alloc(256,sizes.high_bytes);
        c.weight.low=std::aligned_alloc(256,c.weight.low_bytes);c.weight.units=std::aligned_alloc(256,c.weight.unit_bytes);
        c.weight.stream=reinterpret_cast<void*>(uintptr_t(i+1));
        const_cast<float*>(c.a)[0]=i+2;static_cast<float*>(const_cast<void*>(c.weight.low))[0]=i+5;
    }
    for(int type:{0,1})for(int vector:{0,1,2,3})for(int variant=-2;variant<=43;++variant)for(int algorithm:{-1,99}){
        bool tuned=variant>=26 && variant<=34;
        bool half2=variant>=35 && variant<=42;
        bool typed=variant==0 || variant==8 || variant==20 || variant==21 || variant==22 || variant==23 || variant==24 || variant==25 || tuned || half2;
        assert(bool(qkp_dense_dequant_variant_valid(variant))==typed);
        qkp_dense_options_v1 o{1,sizeof(o),argv[1],type,vector,variant,16,algorithm};void* h=nullptr;
        int rc=quactlize_kpack_dense_full_prepare_v1(&shape,&arr,&o,&h);
        bool supported=variant==0 || (q==8 && (variant==24 || variant==25 || tuned || (half2 && type==0))) ||
            (q==8 || q==10 || q==11?(variant==22 || variant==23):(variant==8 || variant==20 || variant==21));
        if((variant==-1 && (q==8 || type==0)) || (variant!=-1 && !supported)){assert(rc && !h);continue;}
        assert(!rc && h);
        auto run=[&](int i){for(int r=0;r<8;++r){
            auto& c=calls[i];const_cast<float*>(c.a)[0]=i+2+r;
            assert(!quactlize_kpack_dense_full_run_v1(h,&c,-1));
            assert(c.output[0]==float(7+2*i+r+200*type));
        }};
        std::thread first(run,0),second(run,1);first.join();second.join();
        auto bad=calls[0];bad.output=static_cast<float*>(bad.workspace);assert(quactlize_kpack_dense_full_run_v1(h,&bad,-1));
        bad=calls[0];bad.workspace_bytes=1;assert(quactlize_kpack_dense_full_run_v1(h,&bad,-1));
        bad=calls[0];bad.m++;assert(quactlize_kpack_dense_full_run_v1(h,&bad,-1));
        bad=calls[0];bad.weight.config++;assert(quactlize_kpack_dense_full_run_v1(h,&bad,-1));
        quactlize_kpack_dense_full_destroy_v1(h);
    }
    qkp_dense_options_v1 invalid{1,sizeof(invalid),argv[1],0,1,43,16,-1};void* h=nullptr;
    assert(quactlize_kpack_dense_full_prepare_v1(&shape,&arr,&invalid,&h) && !h);
    invalid.dequant_variant=21;invalid.vector_io=4;
    assert(quactlize_kpack_dense_full_prepare_v1(&shape,&arr,&invalid,&h) && !h);
    for(auto& c:calls){std::free(const_cast<float*>(c.a));std::free(c.output);std::free(c.workspace);
        std::free(const_cast<void*>(c.weight.low));std::free(const_cast<void*>(c.weight.high));std::free(const_cast<void*>(c.weight.units));}
    }
    // Real runtime lookup, device guard and full table. No GPU timing claim.
    qkp_call_v1 shape{};shape.version=1;shape.size=sizeof(shape);shape.weight.version=1;shape.weight.size=sizeof(qzd_call_v1);
    shape.weight.experts=1;
    int cases[][6]={{12,25600,5120,21,3,3},{12,8192,5120,21,3,3},
        {12,5120,8192,20,1,1},{12,5120,25600,8,1,1},{12,1024,5120,20,2,2},
        {14,5120,25600,21,1,2},{14,1024,5120,20,2,2}};
    shape.m=shape.a_rows=2048;
    for(auto const& row:cases)for(int storage:{0,1}){
        shape.weight.qtype=row[0];shape.weight.n=shape.output_stride=row[1];shape.weight.k=shape.a_stride=row[2];
        int dq=-1,io=-1;assert(!quactlize_kpack_dense_helper_policy_v1(&shape,storage,&dq,&io));
        assert(dq==row[3] && io==row[4+storage]);
        auto miss=[&]{dq=io=99;assert(quactlize_kpack_dense_helper_policy_v1(&shape,storage,&dq,&io)==QKG_SHAPE);assert(dq==-1 && io==-1);};
        shape.m=shape.a_rows=1024;miss();shape.m=shape.a_rows=2048;
        shape.a_stride++;miss();shape.a_stride--;
        shape.output_stride++;miss();shape.output_stride--;
        shape.weight.experts=256;miss();shape.weight.experts=1;
        test_name="PPU-ZW610";test_cu=64;miss();test_name="PPU-ZW810";test_cu=72;
        test_cu=71;miss();test_cu=72;
    }
    for(auto const& row:dense_full_families)for(int i=0;i<5;++i)for(int storage:{0,1}) {
        shape.m=shape.a_rows=dense_full_m[i];
        shape.weight.qtype=row.q;shape.weight.n=shape.output_stride=row.n;shape.weight.k=shape.a_stride=row.k;
        int dq=-1,io=-1;
        assert(!quactlize_kpack_dense_full_helper_policy_v1(&shape,storage,&dq,&io));
        assert(dq==row.choices[storage][i].dequant && io==row.choices[storage][i].io);
        auto miss=[&]{dq=io=99;assert(quactlize_kpack_dense_full_helper_policy_v1(&shape,storage,&dq,&io)==QKG_SHAPE);assert(dq==-1 && io==-1);};
        shape.m++;shape.a_rows++;miss();shape.m--;shape.a_rows--;
        shape.a_stride++;miss();shape.a_stride--;
        shape.output_stride++;miss();shape.output_stride--;
        shape.weight.experts=256;miss();shape.weight.experts=1;
        shape.src_rows=reinterpret_cast<int*>(uintptr_t(256));miss();shape.src_rows=nullptr;
        test_name="PPU-ZW610";test_cu=64;miss();test_name="PPU-ZW810";test_cu=72;
        test_cu=71;miss();test_cu=72;
    }
}
