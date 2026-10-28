// Host ABI test of the actual prefill runtime. Not a device numerical test.
#include "quactlize/prefill/dynamic.hpp"
#include "quactlize/dequant/indexed.h"
#include <hggc_runtime.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <thread>

static int compute_units=72;
extern "C" hggcError_t hggcGetDevice(int* d) { *d=0;return hggcSuccess; }
extern "C" hggcError_t hggcGetDeviceProperties(hggcDeviceProp* p,int) {
    std::strcpy(p->name,compute_units==64?"PPU-ZW610":"PPU-ZW810");return hggcSuccess;
}
extern "C" hggcError_t hggcDeviceGetAttribute(int* n,hggcDeviceAttr attr,int) {
    assert(attr==hggcDevAttrMultiProcessorCount);*n=compute_units;return hggcSuccess;
}
extern "C" hggcError_t hggcStreamIsCapturing(hggcStream_t,hggcStreamCaptureStatus* s) {
    *s=hggcStreamCaptureStatusNone;return hggcSuccess;
}
extern "C" hggcError_t hggcGetLastError() { return hggcSuccess; }
extern "C" int qkp_stage_prepare(qkp_call_v1 const* c,quactlize::prefill::Layout const* l,void* stream) {
    assert(stream==c->weight.stream);
    *reinterpret_cast<float*>(static_cast<char*>(c->workspace)+l->a)=c->a[0];return 0;
}
extern "C" int qkp_stage_finish(qkp_call_v1 const* c,quactlize::prefill::Layout const* l,void* stream) {
    assert(stream==c->weight.stream);
    c->output[0]=*reinterpret_cast<float*>(static_cast<char*>(c->workspace)+l->out);return 0;
}
extern "C" int quactlize_kpack_dequant_v1(qzd_call_v1 const* c,quactlize_ppu_placed_arrangement_v2 const*) {
    *static_cast<float*>(c->output)=*static_cast<float const*>(c->low);return 0;
}
extern "C" int quactlize_kpack_dequant_indexed_v1(qzd_call_v1 const*,quactlize_ppu_placed_arrangement_v2 const*,
    int const*,int const*,int*) { assert(false);return 1; }

int main(int argc,char** argv) {
    assert(argc==2);
    for(int cu:{72,64}) {
    compute_units=cu;
    auto arr=ppu_arrangements::q4_kpack4_transpose_v1();
    qkp_call_v1 shape{};shape.version=1;shape.size=sizeof(shape);shape.m=shape.a_rows=128;
    shape.a_stride=512;shape.output_stride=256;shape.device=0;
    shape.weight={1,sizeof(qzd_call_v1),12,256,512,1,1,5};
    uint64_t bytes=0,old_bytes=0;
    assert(quactlize_kpack_prefill_query_v1(&shape,&arr,&old_bytes)==0);
    assert(quactlize_kpack_prefill_query_dynamic_v1(&shape,&arr,&bytes)==0);
    assert(bytes==old_bytes+(4u<<20));
    qkp_options_v1 options{1,sizeof(options),argv[1],nullptr,nullptr};void* h=nullptr;
    assert(quactlize_kpack_prefill_prepare_dynamic_v1(&shape,&arr,&options,&h)==0 && h);
    qkp_call_v1 calls[2]={shape,shape};
    for (int i=0;i<2;++i) {
        auto& c=calls[i];c.a=static_cast<float*>(std::aligned_alloc(256,128*512*4));
        c.output=static_cast<float*>(std::aligned_alloc(256,128*256*4));
        c.workspace=std::aligned_alloc(256,bytes);c.workspace_bytes=bytes;
        c.weight.low_bytes=256*512/2;c.weight.unit_bytes=256*512/256*16;
        c.weight.low=std::aligned_alloc(256,c.weight.low_bytes);
        c.weight.units=std::aligned_alloc(256,c.weight.unit_bytes);
        c.weight.stream=reinterpret_cast<void*>(uintptr_t(i+1));
        *const_cast<float*>(c.a)=float(i+3);
        *static_cast<float*>(const_cast<void*>(c.weight.low))=float(i+7);
    }
    auto execute=[&](int i) {
        for (int repeat=0;repeat<16;++repeat) {
            assert(quactlize_kpack_prefill_run_dynamic_v1(h,&calls[i])==0);
            assert(calls[i].output[0]==float(10+2*i));
        }
    };
    std::thread first(execute,0),second(execute,1);first.join();second.join();
    assert(quactlize_kpack_prefill_run_v1(h,nullptr)!=0);
    int32_t const* status=nullptr;
    assert(quactlize_kpack_prefill_device_status_v1(h,&status)!=0);
    int32_t const* statuses[2]{};
    for (int i=0;i<2;++i)
        assert(quactlize_kpack_prefill_device_status_dynamic_v1(h,&calls[i],&statuses[i])==0);
    assert(statuses[0]!=statuses[1]);
    for (int which=0;which<6;++which) {
        auto c=calls[0];
        if (which==0) c.workspace_bytes=old_bytes;
        if (which==1) c.output=static_cast<float*>(c.workspace);
        if (which==2) c.m++;
        if (which==3) c.a=nullptr;
        if (which==4) c.weight.config++;
        if (which==5) c.weight.high=c.weight.low;
        assert(quactlize_kpack_prefill_run_dynamic_v1(h,&c)!=0);
    }
    quactlize_kpack_prefill_destroy_v1(h);
    for (auto& c:calls) {
        std::free(const_cast<float*>(c.a));std::free(c.output);std::free(c.workspace);
        std::free(const_cast<void*>(c.weight.low));std::free(const_cast<void*>(c.weight.units));
    }
    }
}
