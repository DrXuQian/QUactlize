#include "quactlize/prefill/dense_api.h"
#include "quactlize/prefill/dynamic.hpp"
#include <hggc_runtime_api.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <thread>

extern "C" hggcError_t hggcGetDevice(int* p){*p=0;return hggcSuccess;}
extern "C" hggcError_t hggcStreamIsCapturing(hggcStream_t,hggcStreamCaptureStatus* p){*p=hggcStreamCaptureStatusNone;return hggcSuccess;}
extern "C" hggcError_t hggcMemsetAsync(void* p,int value,size_t bytes,hggcStream_t){std::memset(p,value,bytes);return hggcSuccess;}
extern "C" int quactlize_dense_io_v1(int storage,int input,int,void const* src,void* dst,int,int,int64_t,int const*,void*) {
    *static_cast<float*>(dst)=*static_cast<float const*>(src)+(input?100*storage:0);return 0;
}
extern "C" int quactlize_kpack_dequant_dense_v1(qzd_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const*,int storage,int variant){
    assert(variant>=0 && variant<=13);*static_cast<float*>(c->output)=*static_cast<float const*>(c->low)+100*storage;return 0;
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
    qkp_call_v1 shape{};shape.version=1;shape.size=sizeof(shape);shape.m=shape.a_rows=128;
    shape.a_stride=512;shape.output_stride=256;shape.weight={1,sizeof(qzd_call_v1),12,256,512,1,1,5};
    auto arr=ppu_arrangements::q4_kpack4_transpose_v1();Layout l{};assert(!layout(shape,&arr,l));
    qkp_call_v1 calls[2]={shape,shape};
    for(int i=0;i<2;++i){auto& c=calls[i];
        c.a=static_cast<float*>(std::aligned_alloc(256,128*512*4));c.output=static_cast<float*>(std::aligned_alloc(256,128*256*4));
        c.workspace_bytes=dynamic_bytes(c,l);c.workspace=std::aligned_alloc(256,c.workspace_bytes);
        c.weight.low_bytes=256*512/2;c.weight.unit_bytes=256*512/256*16;
        c.weight.low=std::aligned_alloc(256,c.weight.low_bytes);c.weight.units=std::aligned_alloc(256,c.weight.unit_bytes);
        c.weight.stream=reinterpret_cast<void*>(uintptr_t(i+1));
        const_cast<float*>(c.a)[0]=i+2;static_cast<float*>(const_cast<void*>(c.weight.low))[0]=i+5;
    }
    for(int type:{0,1})for(int vector:{0,1,2})for(int variant:{-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13})for(int algorithm:{-1,99}){
        qkp_dense_options_v1 o{1,sizeof(o),argv[1],type,vector,variant,16,algorithm};void* h=nullptr;
        int rc=quactlize_kpack_dense_full_prepare_v1(&shape,&arr,&o,&h);
        if(type==0 && variant==-1){assert(rc && !h);continue;}
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
    qkp_dense_options_v1 invalid{1,sizeof(invalid),argv[1],0,1,14,16,-1};void* h=nullptr;
    assert(quactlize_kpack_dense_full_prepare_v1(&shape,&arr,&invalid,&h) && !h);
    for(auto& c:calls){std::free(const_cast<float*>(c.a));std::free(c.output);std::free(c.workspace);
        std::free(const_cast<void*>(c.weight.low));std::free(const_cast<void*>(c.weight.units));}
}
