#include "dense_api.h"
#include "dynamic.hpp"
#include "dense_helpers.hpp"
#include <hggc_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <cstring>

extern "C" int qkp_stage_input(qkp_call_v1 const*,quactlize::prefill::Layout const*,void*);
extern "C" int qkp_stage_finish(qkp_call_v1 const*,quactlize::prefill::Layout const*,void*);

namespace {
using namespace quactlize::prefill;
thread_local std::string dense_error;
template<class T> T entry(void* lib,char const* name) {
    auto fn=reinterpret_cast<T>(dlsym(lib,name));
    if (!fn) throw std::runtime_error(std::string("dense provider missing ")+name);
    return fn;
}
void check(int rc,char const* name) { if(rc)throw std::runtime_error(std::string(name)+" rc="+std::to_string(rc)); }
struct DensePlan {
    qkp_call_v1 shape{};quactlize_ppu_placed_arrangement_v2 arr{};Layout layout{};
    int storage,vector_io,dequant_variant,algorithm;
    void *library=nullptr,*blas=nullptr;
    int(*destroy)(void*)=nullptr;int(*stream)(void*,void*)=nullptr;
    int(*workspace)(void*,void*,size_t)=nullptr;
    using Gemm=int(*)(void*,int,int,int,int,int,void const*,void const*,int,int,
        void const*,int,int,void const*,void*,int,int,int,int);
    Gemm gemm=nullptr;std::mutex mutex;
    ~DensePlan() {if(blas && destroy)destroy(blas);if(library)dlclose(library);}
};
}
extern "C" int quactlize_kpack_dense_full_workspace_v1(qkp_call_v1 const* c,
        quactlize_ppu_placed_arrangement_v2 const* arr, uint64_t* bytes) {
    if (bytes) *bytes=0;
    if (!c || !arr || !bytes || c->weight.experts!=1 || !address_free(*c)) return QKG_INVALID;
    try {
        Layout l{};int rc=layout(*c,arr,l,true);if(rc)return rc;
        *bytes=dynamic_bytes(*c,l);return QKG_OK;
    } catch (std::exception const& e) {dense_error=e.what();return QKG_OVERFLOW;}
}
static int dense_helper_policy(qkp_call_v1 const* c, int storage,
        int* dequant_variant, int* vector_io, bool full) {
    if (dequant_variant) *dequant_variant=-1;
    if (vector_io) *vector_io=-1;
    if (!c || !dequant_variant || !vector_io || c->version!=1 || c->size!=sizeof(*c) ||
        c->weight.version!=1 || c->weight.size!=sizeof(c->weight) || c->device<0 ||
        (storage!=0 && storage!=1)) return QKG_INVALID;
    if (c->weight.experts!=1 || c->a_rows!=c->m || c->a_stride!=c->weight.k ||
        c->output_stride!=c->weight.n || c->src_rows || c->dst_rows || c->offsets) return QKG_SHAPE;
    auto lookup=full?quactlize::prefill::dense_full_helpers:quactlize::prefill::dense_helpers;
    auto choice=lookup(c->weight.qtype,c->m,c->weight.n,c->weight.k,storage);
    if (choice.dequant<0) return QKG_SHAPE;
    hggcDeviceProp prop{};
    if (hggcGetDeviceProperties(&prop,c->device)!=hggcSuccess) return QKG_RUNTIME;
    if (std::strcmp(prop.name,"PPU-ZW810") || prop.multiProcessorCount!=72) return QKG_SHAPE;
    *dequant_variant=choice.dequant;*vector_io=choice.io;return QKG_OK;
}
extern "C" int quactlize_kpack_dense_helper_policy_v1(qkp_call_v1 const* c, int storage,
        int* dequant_variant, int* vector_io) {
    return dense_helper_policy(c,storage,dequant_variant,vector_io,false);
}
extern "C" int quactlize_kpack_dense_full_helper_policy_v1(qkp_call_v1 const* c, int storage,
        int* dequant_variant, int* vector_io) {
    return dense_helper_policy(c,storage,dequant_variant,vector_io,true);
}
extern "C" int quactlize_kpack_dense_full_prepare_v1(qkp_call_v1 const* c,
        quactlize_ppu_placed_arrangement_v2 const* arr,qkp_dense_options_v1 const* o,void** result) {
    if(!result)return QKG_INVALID;
    *result=nullptr;
    try {
        if(!c || !arr || !o || o->version!=1 || o->size!=sizeof(*o) || !o->sdk ||
            !std::filesystem::path(o->sdk).is_absolute() || c->weight.experts!=1 || !address_free(*c) ||
            (o->storage!=0 && o->storage!=1) || o->vector_io<0 || o->vector_io>3 ||
            (o->dequant_variant!=-1 && !qkp_dense_dequant_variant_valid(o->dequant_variant)) ||
            (o->dequant_variant==-1 && o->storage!=1) ||
            (o->dequant_variant>=QKP_DENSE_DEQUANT_Q8_HALF2_35 &&
             o->dequant_variant<=QKP_DENSE_DEQUANT_Q8_PAIRED_42 && o->storage!=0) ||
            (o->dequant_variant==-1 ? c->weight.qtype==8 :
                !qkp_dense_dequant_variant_supports(c->weight.qtype,o->dequant_variant)) ||
            (o->math_mode!=0 && o->math_mode!=1 && o->math_mode!=3 &&
             o->math_mode!=16 && o->math_mode!=17 && o->math_mode!=19) ||
            (o->algorithm!=-1 && o->algorithm!=99))return QKG_INVALID;
        auto h=std::make_unique<DensePlan>();
        int rc=layout(*c,arr,h->layout,true);if(rc)return rc;
        int device=-1;check(hggcGetDevice(&device),"device");if(device!=c->device)return QKG_INVALID;
        hggcStreamCaptureStatus capture;
        check(hggcStreamIsCapturing(static_cast<hggcStream_t>(c->weight.stream),&capture),"capture");
        if(capture!=hggcStreamCaptureStatusNone)return QKG_INVALID;
        h->shape=*c;h->arr=*arr;h->storage=o->storage;h->vector_io=o->vector_io;
        h->dequant_variant=o->dequant_variant;h->algorithm=o->algorithm;
        auto path=std::filesystem::path(o->sdk)/"CUDA_SDK/targets/x86_64-linux/lib/libcublas.so";
        h->library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);if(!h->library)throw std::runtime_error(dlerror());
        auto create=entry<int(*)(void**)>(h->library,"cublasCreate_v2");
        h->destroy=entry<decltype(h->destroy)>(h->library,"cublasDestroy_v2");
        h->stream=entry<decltype(h->stream)>(h->library,"cublasSetStream_v2");
        h->workspace=entry<decltype(h->workspace)>(h->library,"cublasSetWorkspace_v2");
        h->gemm=entry<DensePlan::Gemm>(h->library,"cublasGemmEx");
        check(create(&h->blas),"create");
        check(entry<int(*)(void*,int)>(h->library,"cublasSetMathMode")(h->blas,o->math_mode),"math mode");
        *result=h.release();dense_error.clear();return QKG_OK;
    }catch(std::exception const& e){dense_error=e.what();return QKG_RUNTIME;}
}
extern "C" int quactlize_kpack_dense_full_run_v1(void* handle,qkp_call_v1 const* c,int stage) {
    if(!handle || !c || stage < -1 || stage>4)return QKG_INVALID;
    auto& h=*static_cast<DensePlan*>(handle);
    if(!same_shape(*c,h.shape))return QKG_INVALID;
    int rc=validate_buffers(*c,h.arr,dynamic_bytes(*c,h.layout));if(rc)return rc;
    try {
        auto base=static_cast<uint8_t*>(c->workspace);auto stream=c->weight.stream;
        auto error=reinterpret_cast<int*>(base+h.layout.error);
        if(stage==-1 || stage==4)check(hggcMemsetAsync(error,0,4,static_cast<hggcStream_t>(stream)),"reset");
        if(stage==-1 || stage==0)check(h.storage==1 && !h.vector_io ? qkp_stage_input(c,&h.layout,stream):
            quactlize_dense_io_v1(h.storage,1,h.vector_io,c->a,
                base+h.layout.a,c->m,c->weight.k,c->a_stride,error,stream),"input");
        auto w=c->weight;w.output=base+h.layout.weights;w.output_bytes=uint64_t(w.n)*w.k*2;
        if(stage==-1 || stage==1) {
            rc=h.dequant_variant==-1 ? quactlize_kpack_dequant_v1(&w,&h.arr):
                quactlize_kpack_dequant_dense_v1(&w,&h.arr,h.storage,h.dequant_variant);
            check(rc,"dequant");
        }
        if(stage==-1 || stage==2) {
            // Per-enqueue stream/workspace rebinding. No persistent caller
            // addresses, allocation, synchronization or tuning in this layer.
            std::lock_guard<std::mutex> lock(h.mutex);
            check(h.stream(h.blas,stream),"stream");
            check(h.workspace(h.blas,base+h.layout.bytes,blas_workspace_bytes),"workspace");
            float alpha=1,beta=0;int type=h.storage?14:2;
            check(h.gemm(h.blas,1,0,w.n,c->m,w.k,&alpha,w.output,type,w.k,base+h.layout.a,type,w.k,
                &beta,base+h.layout.out,type,w.n,68,h.algorithm),"FP32 GEMM");
        }
        if(stage==-1 || stage==3)check(h.storage==1 && !h.vector_io ? qkp_stage_finish(c,&h.layout,stream):
            quactlize_dense_io_v1(h.storage,0,h.vector_io,base+h.layout.out,
                c->output,c->m,w.n,c->output_stride,error,stream),"output");
        return QKG_OK;
    }catch(std::exception const& e){dense_error=e.what();return QKG_RUNTIME;}
}
extern "C" void quactlize_kpack_dense_full_destroy_v1(void* h){delete static_cast<DensePlan*>(h);}
extern "C" char const* quactlize_kpack_dense_full_error_v1(){return dense_error.c_str();}
