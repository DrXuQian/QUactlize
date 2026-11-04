#pragma once
#include "matmul_validation.hpp"
#include "../dispatch/api.h"
#include "../runtime/library.hpp"
#include "../mxfp4/provider.hpp"
#include <memory>
#include <mutex>
#include <vector>

// Private producer seam shared by independent projections and FFN composition.
// Selection and module ownership remain here; no new caller-visible recipes.
namespace quactlize::moe::matmul {
struct Context;
struct Plan {
    Context* owner=nullptr;
    quactlize_ppu_moe_matmul_params_v1 params{};
    quactlize_ppu_moe_matmul_plan_info_v1 info{1,sizeof(info)};
    uint8_t const *low=nullptr,*high=nullptr,*units=nullptr;
    int kind=QKS_SMALLM_TC,dequant=-1,metadata=QK_METADATA_BF16;
    qks_smallm_choice_v2 small{};
    qks_choice_v1 tc{};
    qks_request_v1 request{};
    qzm_prepare_plan_v2 preparation{};
    void* handle=nullptr;
    std::shared_ptr<mxfp4::Provider> mxfp4;
    uint64_t src=0,dst=0,offsets=0,inverse=0,counts=0,active=0,count=0;
    uint64_t route=0,input=0,output=0,status=0,kernel=0,kernel_bytes=0;
    uint64_t scale=0,zero=0,plane_bytes=0;
    ~Plan() { if(handle) quactlize_kpack_dispatch_destroy_v1(handle); }
};
struct Context {
    int device=0;
    std::filesystem::path root;
    std::string sdk;
    void* dispatcher=nullptr;
    bool jit=false;
    std::mutex mutex;
    std::vector<std::unique_ptr<Plan>> plans;
    ~Context() { plans.clear();if(dispatcher) quactlize_kpack_dispatch_close_v1(dispatcher); }
    void enable_jit() {
        if(!jit) { library::enable_jit(dispatcher,root,sdk);jit=true; }
    }
};
void device_check(int device);
void outside_capture();
qkg_simt_call_v2 indexed(Plan const&);
qzm_prepare_v1 prepare_call(Plan const&);
void prepare(Plan&);
void prepare_input(Plan const&,float const*,int32_t const*,void* workspace,void* stream);
// Owns SF expansion and the dynamic call, but NOT routing or scatter. mode
// controls whether the actual Split-K reducer is left to the consumer.
qk_compute_device_call_v4 grouped(Plan const&,void const* input,int32_t const* offsets,
                                void* workspace,void* stream,int mode);
void run(Plan const&,float const*,int32_t const*,float*,void* workspace,void* stream);
}
