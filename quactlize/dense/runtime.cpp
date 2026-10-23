#include "api.h"
#include "validation.hpp"
#include "../dispatch/api.h"
#include "../dispatch/selection.hpp"
#include "../dispatch/smallm.hpp"
#include "../prefill/api.h"
#include "../prefill/dense_api.h"
#include "../execution/simt_validation.hpp"
#include <hggc_runtime_api.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

extern "C" int quactlize_dense_copy_f32(float const*,float*,int,int,int64_t,int64_t,void*);
extern "C" int quactlize_dense_convert_f16(int,void const*,void*,int,int,int64_t,int,void*);

namespace quactlize::dense {
namespace fs=std::filesystem;
thread_local std::string error;
void require(bool condition,char const* message) {
    if (!condition) throw std::runtime_error(message && *message ? message : "dense runtime failure");
}
void dispatch_check(int rc) { require(rc==QKS_OK,quactlize_kpack_dispatch_error_v1()); }
void prefill_check(int rc) { require(rc==QKG_OK,quactlize_kpack_prefill_error_v1()); }
void dense_full_check(int rc) { require(rc==QKG_OK,quactlize_kpack_dense_full_error_v1()); }
void device_check(int expected) {
    int current=-1;
    require(hggcGetDevice(&current)==hggcSuccess && current==expected,"dense context/current device mismatch");
}
std::string environment(char const* name,std::string fallback) {
    auto value=std::getenv(name);return value && *value ? value : std::move(fallback);
}
fs::path runtime_directory() {
    Dl_info image{};
    require(dladdr(reinterpret_cast<void*>(&quactlize_ppu_dense_create_v1),&image) && image.dli_fname,
        "cannot locate libquactlize_ppu.so");
    return fs::canonical(fs::path(image.dli_fname)).parent_path()/"quactlize-runtime";
}
std::string python_executable() {
    auto requested=environment("QUACTLIZE_PPU_PYTHON","");
    if (!requested.empty()) {
        require(fs::path(requested).is_absolute() && access(requested.c_str(),X_OK)==0,
            "QUACTLIZE_PPU_PYTHON must name an executable absolute path");
        return fs::canonical(requested).string();
    }
    auto path=environment("PATH","/usr/bin:/bin");size_t begin=0;
    for (;;) {
        auto end=path.find(':',begin);auto dir=path.substr(begin,end-begin);
        if (!dir.empty()) {
            auto candidate=fs::path(dir)/"python3";
            if (fs::is_regular_file(candidate) && access(candidate.c_str(),X_OK)==0)
                return fs::canonical(candidate).string();
        }
        if (end==std::string::npos) break;
        begin=end+1;
    }
    throw std::runtime_error("python3 not found for selected-parent JIT");
}

struct Context;
enum class Kind { Simt, Q4, Tc, Full };
struct Plan {
    Context* owner=nullptr;
    Kind kind=Kind::Tc;
    quactlize_ppu_dense_params_v1 params{};
    quactlize_ppu_dense_plan_info_v1 info{1,sizeof(info)};
    uint8_t const *low=nullptr,*high=nullptr,*units=nullptr;
    qks_smallm_choice_v2 small{};
    qks_choice_v1 tc{};
    qks_request_v1 request{};
    qks_prefill_choice_v1 prefill{};
    bool confirmed_prefill=false;
    dispatch::dense_prefill::Decision dense_prefill{};
    int compute=QK_COMPUTE_F16,metadata_type=QK_METADATA_F16;
    void* handle=nullptr;
    uint64_t input_offset=0,output_offset=0,kernel_offset=0,scale_offset=0,zero_offset=0;
    uint64_t kernel_bytes=0,plane_bytes=0;
    bool copy_input=false,copy_output=false;
    bool native_f16=false;
    bool legacy_adapters=false;
    int helper_io=-1,helper_dequant=-1;
    bool typed_full=false;
    int full_storage=1; // legacy Full-BF16; measured dense policy explicitly uses F16
    ~Plan() {
        if (kind==Kind::Tc && handle) quactlize_kpack_dispatch_destroy_v1(handle);
        if (kind==Kind::Full && handle) {
            if (typed_full) quactlize_kpack_dense_full_destroy_v1(handle);
            else quactlize_kpack_prefill_destroy_v1(handle);
        }
    }
};
struct Context {
    int device,compute;
    std::string sdk;
    fs::path root;
    void* dispatcher=nullptr;
    bool jit=false;
    int dense_device=-1;
    std::mutex mutex;
    std::vector<std::unique_ptr<Plan>> plans;
    ~Context() {
        plans.clear();
        if (dispatcher) quactlize_kpack_dispatch_close_v1(dispatcher);
    }
    bool dense_policy_device() {
        if(dense_device<0) {
            hggcDeviceProp prop{};
            require(hggcGetDeviceProperties(&prop,device)==hggcSuccess,"dense device properties unavailable");
            dense_device=!std::strcmp(prop.name,"PPU-ZW810") && prop.multiProcessorCount==72 && prop.warpSize==32;
        }
        return dense_device==1;
    }
    // JIT configuration is only needed for TC. Context construction and a
    // pure SIMT plan need neither Python nor a writable compilation cache.
    void enable_jit() {
        if (jit) return;
        auto cache=environment("QUACTLIZE_PPU_CACHE_DIR",environment("XDG_CACHE_HOME",
            environment("HOME","/tmp")+"/.cache")+"/quactlize/ppu");
        auto python=python_executable();auto helper=(root/"jit/tools/kpack_jit.py").string();
        require(fs::path(cache).is_absolute(),"QUACTLIZE_PPU_CACHE_DIR must be absolute");
        qks_jit_options_v1 options{1,sizeof(options),python.c_str(),helper.c_str(),sdk.c_str(),cache.c_str()};
        dispatch_check(quactlize_kpack_dispatch_enable_jit_v1(dispatcher,&options));jit=true;
    }
};

qks_request_v1 request_for(quactlize_ppu_dense_params_v1 const& p) {
    return {1,sizeof(qks_request_v1),p.qtype,p.qtype==8?QK_DENSE_SF:QK_DENSE_FQ,
        p.m,p.n,p.k,1,p.m,p.arrangement.mapping_id};
}
qkg_simt_call_v2 simt_call(Plan const& p) {
    qkg_call_v1 c{1,sizeof(c)};
    c.qtype=p.params.qtype;c.n=p.params.n;c.k=p.params.k;c.experts=1;
    c.rows=p.params.m;c.mode=QKG_DENSE;c.input_type=QKG_F32;c.channels=c.topk=1;
    c.a_row_stride=c.a_token_stride=p.params.k;c.out_row_stride=p.params.n;c.ids_stride=1;
    c.low=p.low;c.high=p.high;c.units=p.units;
    return {2,sizeof(qkg_simt_call_v2),c,p.compute};
}
qkp_call_v1 full_call(Plan const& p) {
    qkp_call_v1 c{1,sizeof(c)};auto& w=c.weight;
    w.version=1;w.size=sizeof(w);w.qtype=p.params.qtype;
    w.n=p.params.n;w.k=p.params.k;w.experts=1;w.operation=1;w.config=p.prefill.dequant_config;
    c.m=c.a_rows=p.params.m;c.device=p.owner->device;
    c.a_stride=p.params.input_stride;c.output_stride=p.params.output_stride;
    return c;
}

// Mirrors the working PoC order: matched pool -> original decode table ->
// same-family TC -> old dense fallback -> explicit coverage floor.
// Module tickets are authoritative.
// No re-comparison with the unmaterialized host selector or ticket parsing.
void select(Plan& p) {
    auto& ctx=*p.owner;p.request=request_for(p.params);
    if (p.params.m<=8) {
        auto typed=simt_call(p);
        // The pure decision tells us whether JIT may be needed, avoiding a
        // Python/cache dependency for a SIMT hit. It never supplies a ticket.
        auto decision=dispatch::select_smallm(typed,p.params.arrangement);
        if (decision.status==QKS_OK && decision.choice.base.kind==QKS_SMALLM_TC) ctx.enable_jit();
        int rc=quactlize_kpack_dispatch_query_smallm_v3(ctx.dispatcher,&typed,&p.params.arrangement,&p.small);
        if (rc!=QKS_OK && rc!=QKS_MISS) dispatch_check(rc);
        if (rc==QKS_MISS) {
            // BF16's legacy table may also materialize TC. Enable only when
            // that donor actually selects TC; F16 Q4 has its separate board.
            auto donor=dispatch::smallm::select(typed.call);
            if (donor.row && !dispatch::smallm::data::kChoices[donor.row->choice].simt) ctx.enable_jit();
            if (p.compute==QK_COMPUTE_F16 && p.params.qtype==12) {
                rc=quactlize_kpack_q4_decode_select_v2(&typed,&p.params.arrangement,&p.small.q4,&p.small.base.sizes);
                if (rc==QKG_OK) {
                    p.small.base.kind=QKS_SMALLM_Q4;p.small.base.policy=QKS_DECODE_MEASURED;
                } else { require(rc==QKG_SHAPE,"Q4 decode selection failed");rc=QKS_MISS; }
            } else {
                rc=quactlize_kpack_dispatch_query_smallm_v2(ctx.dispatcher,&typed,&p.params.arrangement,&p.small.base);
                if (rc!=QKS_OK && rc!=QKS_MISS) dispatch_check(rc);
                if (rc==QKS_OK && p.compute==QK_COMPUTE_BF16 && p.params.qtype==12 &&
                    p.small.base.kind==QKS_SMALLM_SIMT && p.small.base.policy==QKS_COMPUTE_INITIAL) {
                    qkg_sizes_v1 sizes{};
                    int qrc=quactlize_kpack_q4_decode_select_v2(&typed,&p.params.arrangement,&p.small.q4,&sizes);
                    if (qrc==QKG_OK) {p.small.base.kind=QKS_SMALLM_Q4;p.small.base.sizes=sizes;}
                    else require(qrc==QKG_SHAPE,"BF16 Q4 decode selection failed");
                }
            }
        }
        if (rc==QKS_OK) {
            if (p.small.base.kind==QKS_SMALLM_TC) {
                p.tc=p.small.base.tc;
                // The SF expansion was timed independently, but is part of
                // this selected call. Preserve its winning implementation too.
                if (decision.status==QKS_OK && decision.dequant_config>=0)
                    p.prefill={1,sizeof(p.prefill),1,decision.dequant_config,0,1,0,0};
            } else {
                p.kind=p.small.base.kind==QKS_SMALLM_Q4?Kind::Q4:Kind::Simt;
                p.kernel_bytes=p.small.base.sizes.workspace_bytes;return;
            }
        }
    }
    if (!p.tc.ticket) {
        // The complete GPU-call winner is authoritative, including its exact
        // adapters/precision. An older helper/TC-only table cannot overwrite it.
        auto candidate=dispatch::dense_prefill::select(p.request,p.compute);
        if(candidate && p.params.input_stride==p.params.k && p.params.output_stride==p.params.n &&
            ctx.dense_policy_device()) p.dense_prefill=candidate;
        if(p.dense_prefill) {
            auto const& c=*p.dense_prefill.choice;
            p.helper_io=c.io;p.helper_dequant=c.dequant;
            p.prefill={1,sizeof(p.prefill),c.route,c.route==2?5:c.dequant,p.params.m,
                p.dense_prefill.policy==QKS_DENSE_PREFILL_BUCKET,0,0};
            if(c.route==2) {p.kind=Kind::Full;p.full_storage=0;return;}
            p.request.route=c.route;p.native_f16=true;
            ctx.enable_jit();
            dispatch_check(quactlize_kpack_dispatch_query_dense_prefill_v1(ctx.dispatcher,&p.request,&p.tc));
        } else {
            qks_prefill_choice_v2 confirmed{};
            int rc=p.params.input_stride==p.params.k && p.params.output_stride==p.params.n ?
                quactlize_kpack_dispatch_prefill_v2(&p.request,7,p.compute,1,&confirmed) : QKS_MISS;
            if (rc==QKS_OK) {
                p.prefill=confirmed.base;p.confirmed_prefill=true;
            } else if (rc==QKS_MISS) {
                rc=quactlize_kpack_dispatch_prefill_v1(&p.request,7,&p.prefill);
            }
            if (rc!=QKS_OK && rc!=QKS_MISS) dispatch_check(rc);
            if (rc==QKS_OK) {
                // Existing isolated F16 component ranking is a route proposal
                // for BF16, not relabeled measured BF16 latency.
                if (p.compute==QK_COMPUTE_BF16) p.prefill.predicted=1;
                if (p.prefill.route==2) {p.kind=Kind::Full;return;}
                p.request.route=p.prefill.route;
            }
            ctx.enable_jit();
            if (p.confirmed_prefill) {
                p.native_f16=true;
                dispatch_check(quactlize_kpack_dispatch_query_prefill_v1(ctx.dispatcher,&p.request,p.compute,1,&p.tc));
            }
            if (p.params.m<=8 && p.params.qtype==12) {
                rc=quactlize_kpack_dispatch_query_compute_v1(ctx.dispatcher,&p.request,p.compute,QKD_F32,1,&p.tc);
                if (rc!=QKS_OK && rc!=QKS_MISS) dispatch_check(rc);
            }
            if (!p.tc.ticket) {
                auto candidate=dispatch::select_dense_dynamic(p.request,p.compute);
                p.legacy_adapters=candidate.policy==QKS_LEGACY_DENSE;
                p.native_f16=p.compute==QK_COMPUTE_F16 && (p.params.m>8 || p.legacy_adapters);
                dispatch_check(p.native_f16 ? quactlize_kpack_dispatch_query_dense_native_v1(ctx.dispatcher,&p.request,&p.tc) :
                    quactlize_kpack_dispatch_query_dense_dynamic_v1(ctx.dispatcher,&p.request,p.compute,&p.tc));
            }
        }
    }
    p.kind=Kind::Tc;
    int compute=-1;
    dispatch_check(quactlize_kpack_dispatch_dense_contract_v1(ctx.dispatcher,&p.tc,&p.request,&compute,&p.metadata_type));
    require(compute==p.compute,"dense selected precision differs");
    p.kernel_bytes=p.tc.workspace_bytes;
    if (p.request.route==QK_DENSE_SF && p.params.qtype!=8)
        p.plane_bytes=uint64_t(p.params.n)*p.params.k/p.params.arrangement.group_size*2;
}

void select_candidate(Plan& p,quactlize_ppu_dense_candidate_v1 const& c) {
    require(c.version==1 && c.size==sizeof(c) && c.kind>=0 && c.kind<=3 &&
        (c.native_f16==0 || c.native_f16==1) && (c.legacy_adapters==0 || c.legacy_adapters==1) &&
        c.dequant_config>=-1 && c.dequant_config<=12 && std::memchr(c.parent,0,sizeof(c.parent)),
        "invalid dense candidate ABI");
    require(!c.native_f16 || p.compute==QK_COMPUTE_F16,"native F16 candidate in BF16 context");
    require(!c.legacy_adapters || c.native_f16,"legacy adapter requires native F16");
    bool tile=c.tm || c.tn || c.tk || c.wm || c.wn || c.stages || c.ap || c.delivery_n || c.grid_mode || c.grid_b;
    bool reader=c.variant || c.columns || c.warps || c.values;
    p.request=request_for(p.params);p.request.route=c.route;
    if (c.kind==3) {
        require(c.route==QUACTLIZE_DENSE_FULL_BF16 && p.params.qtype!=8 && p.params.m>=128 &&
            !tile && !reader && !c.parent[0] && !c.native_f16 && !c.legacy_adapters && !c.split &&
            c.dequant_config>=0 && (c.dequant_config<=5 || p.params.qtype==12 || p.params.qtype==13),
            "invalid full-dequant candidate");
        p.kind=Kind::Full;p.prefill={1,sizeof(p.prefill),2,c.dequant_config,0,1,0,0};return;
    }
    if (c.kind==1 || c.kind==2) {
        require(c.route==QUACTLIZE_DENSE_FQ_TC && p.params.m<=8 && !tile && !c.parent[0] &&
            !c.native_f16 && !c.legacy_adapters && c.dequant_config==-1,"invalid SIMT candidate");
        auto d=simt_call(p);p.small.base.policy=QKS_PREDICTED;
        if (c.kind==1) {
            p.kind=Kind::Simt;p.small.base.simt={1,sizeof(p.small.base.simt),c.variant,c.columns,c.warps,c.values,c.split};
            require(quactlize_kpack_simt_query_v2(&d,&p.small.base.simt,&p.params.arrangement,&p.small.base.sizes)==QKG_OK,
                "explicit SIMT candidate is not available");
        } else {
            require(!reader && !c.split && p.params.qtype==12,"invalid selected Q4 candidate");
            p.kind=Kind::Q4;
            require(quactlize_kpack_q4_decode_select_v2(&d,&p.params.arrangement,&p.small.q4,&p.small.base.sizes)==QKG_OK,
                "selected Q4 reader is not available");
        }
        p.kernel_bytes=p.small.base.sizes.workspace_bytes;return;
    }
    require((c.route==QK_DENSE_FQ || c.route==QK_DENSE_SF) && !reader &&
        (c.route==QK_DENSE_SF || c.dequant_config==-1) &&
        (p.params.qtype!=8 || c.dequant_config==-1),"invalid TC candidate route/metadata");
    require(c.dequant_config<=3 || ((p.params.qtype==12 || p.params.qtype==13) && c.dequant_config<=5),
        "invalid dense SF expansion config");
    p.owner->enable_jit();
    qks_tc_candidate_v1 tc{1,sizeof(tc),c.tm,c.tn,c.tk,c.wm,c.wn,c.stages,c.ap,c.delivery_n,-1,
        c.split,c.grid_mode,c.grid_b,{}};
    std::memcpy(tc.parent,c.parent,sizeof(tc.parent));
    dispatch_check(quactlize_kpack_dispatch_query_candidate_v1(p.owner->dispatcher,&p.request,&tc,p.compute,c.native_f16,&p.tc));
    p.kind=Kind::Tc;p.native_f16=c.native_f16;p.legacy_adapters=c.legacy_adapters;
    int compute=-1;
    dispatch_check(quactlize_kpack_dispatch_dense_contract_v1(p.owner->dispatcher,&p.tc,&p.request,&compute,&p.metadata_type));
    require(compute==p.compute,"candidate changed compute precision");
    p.kernel_bytes=p.tc.workspace_bytes;
    if (c.route==QK_DENSE_SF && p.params.qtype!=8) {
        p.plane_bytes=uint64_t(p.params.n)*p.params.k/p.params.arrangement.group_size*2;
        if (c.dequant_config>=0) p.prefill={1,sizeof(p.prefill),1,c.dequant_config,0,1,0,0};
    }
}

void prepare(Plan& p,quactlize_ppu_dense_candidate_v1 const* candidate=nullptr) {
    if (candidate) select_candidate(p,*candidate);else select(p);
    auto& info=p.info;info.compute_type=p.compute;
    if (!p.dense_prefill && (p.kind==Kind::Full || (p.kind==Kind::Tc && p.native_f16 && p.compute==QK_COMPUTE_F16))) {
        auto shape=full_call(p);
        // Full's admitted route is BF16 even in the default F16 context.
        // Helper promotion must not silently change that arithmetic contract.
        int storage=p.kind==Kind::Full ? 1 : 0;
        auto lookup=p.kind==Kind::Full?quactlize_kpack_dense_full_helper_policy_v1:
            quactlize_kpack_dense_helper_policy_v1;
        int rc=lookup(&shape,storage,&p.helper_dequant,&p.helper_io);
        require(rc==QKG_OK || rc==QKG_SHAPE,"dense helper policy query failed");
    }
    if (p.kind==Kind::Full) {
        auto shape=full_call(p);
        p.typed_full=p.helper_dequant>=0;
        if (p.typed_full) {
            dense_full_check(quactlize_kpack_dense_full_workspace_v1(&shape,&p.params.arrangement,&info.workspace_bytes));
            qkp_dense_options_v1 options{1,sizeof(options),p.owner->sdk.c_str(),p.full_storage,p.helper_io,p.helper_dequant,16,-1};
            dense_full_check(quactlize_kpack_dense_full_prepare_v1(&shape,&p.params.arrangement,&options,&p.handle));
        } else {
            prefill_check(quactlize_kpack_prefill_query_dynamic_v1(&shape,&p.params.arrangement,&info.workspace_bytes));
            qkp_options_v1 options{1,sizeof(options),p.owner->sdk.c_str(),nullptr,nullptr};
            prefill_check(quactlize_kpack_prefill_prepare_dynamic_v1(&shape,&p.params.arrangement,&options,&p.handle));
        }
        info.route=QUACTLIZE_DENSE_FULL;info.compute_type=p.full_storage;
        info.policy=p.dense_prefill?p.dense_prefill.policy:p.confirmed_prefill?QKS_PREFILL_COMPONENT_MEASURED:
            p.prefill.predicted?QKS_PREDICTED:QKS_COMPONENT_MEASURED;info.split=1;
        if (p.typed_full) std::snprintf(info.config,sizeof(info.config),"full-%s:cublas:typed-dq%d:io%d",p.full_storage?"bf16":"f16",p.helper_dequant,p.helper_io);
        else std::snprintf(info.config,sizeof(info.config),"full-bf16:cublas:dq%d",p.prefill.dequant_config);
        return;
    }
    uint64_t cursor=0;
    p.copy_input=p.native_f16 || p.params.input_stride!=p.params.k;
    p.copy_output=p.native_f16 || p.params.output_stride!=p.params.n;
    if (p.copy_input) p.input_offset=reserve(cursor,uint64_t(p.params.m)*p.params.k*(p.native_f16?2:4));
    if (p.copy_output) p.output_offset=reserve(cursor,uint64_t(p.params.m)*p.params.n*(p.native_f16?2:4));
    if (p.plane_bytes) {p.scale_offset=reserve(cursor,p.plane_bytes);p.zero_offset=reserve(cursor,p.plane_bytes);}
    p.kernel_offset=reserve(cursor,p.kernel_bytes);info.workspace_bytes=cursor;
    if (p.kind==Kind::Tc) {
        dispatch_check(quactlize_kpack_dispatch_prepare_dynamic_v1(p.owner->dispatcher,&p.tc,&p.handle));
        info.route=p.request.route==QK_DENSE_FQ?QUACTLIZE_DENSE_FQ_TC:QUACTLIZE_DENSE_SF_TC;
        info.policy=p.tc.policy;info.split=p.tc.split;
        if(p.dense_prefill) std::snprintf(info.config,sizeof(info.config),"%s:dq%d:io%d",p.tc.parent,p.prefill.dequant_config,p.helper_io);
        else std::snprintf(info.config,sizeof(info.config),"%s",p.tc.parent);
    } else {
        info.route=QUACTLIZE_DENSE_SIMT;info.policy=p.small.base.policy;
        if (p.kind==Kind::Simt) {
            auto const& f=p.small.base.simt;info.split=f.split;
            std::snprintf(info.config,sizeof(info.config),"simt:v%d:c%d:w%d:x%d:s%d",f.variant,f.columns,f.warps,f.values,f.split);
        } else {
            auto const& f=p.small.q4;info.split=1;
            std::snprintf(info.config,sizeof(info.config),"q4:r%d:v%d:w%d:x%d:c%d",f.reader,f.variant,f.warps,f.values,f.columns);
        }
    }
}

void expand_sf(Plan const& p,uint8_t* base,void* stream,void const*& metadata,void const*& zero) {
    metadata=p.units;zero=nullptr;
    if (!p.plane_bytes) return;
    auto scales=reinterpret_cast<uint16_t*>(base+p.scale_offset);
    auto zeros=reinterpret_cast<uint16_t*>(base+p.zero_offset);
    int rc;
    if (p.prefill.version && p.prefill.route==1) {
        qzd_call_v1 c{1,sizeof(c)};c.qtype=p.params.qtype;c.n=p.params.n;c.k=p.params.k;c.experts=1;
        c.operation=0;c.config=p.prefill.dequant_config;c.low=p.low;c.high=p.high;c.units=p.units;
        c.low_bytes=p.params.sizes.low_bytes;c.high_bytes=p.params.sizes.high_bytes;c.unit_bytes=p.params.sizes.units_bytes;
        c.output=scales;c.zero=zeros;c.output_bytes=p.plane_bytes;c.stream=stream;
        qzd_call_v2 d{2,sizeof(d),c,p.metadata_type};
        rc=quactlize_kpack_dequant_v2(&d,&p.params.arrangement);
    } else rc=quactlize_kpack_sf_prepare_v2(p.params.qtype,p.params.n,p.params.k,1,p.units,p.params.sizes.units_bytes,
        scales,zeros,p.plane_bytes,&p.params.arrangement,p.metadata_type,stream);
    require(rc==QKG_OK,"dense SF expansion failed");metadata=scales;zero=zeros;
}

void copy_endpoint(Plan const& p,bool input,void const* src,void* dst,void* stream) {
    int width=input?p.params.k:p.params.n;
    int64_t stride=input?p.params.input_stride:p.params.output_stride;
    int rc;
    if (!p.native_f16) rc=quactlize_dense_copy_f32(static_cast<float const*>(src),static_cast<float*>(dst),
        p.params.m,width,input?stride:width,input?width:stride,stream);
    else if (p.helper_io>=0) rc=quactlize_dense_io_v1(0,input,p.helper_io,src,dst,p.params.m,width,stride,nullptr,stream);
    else rc=quactlize_dense_convert_f16(input,src,dst,p.params.m,width,stride,p.legacy_adapters,stream);
    require(!rc,input?"dense input copy failed":"dense output copy failed");
}

void run(Plan const& p,float const* a,float* output,void* workspace,uint64_t bytes,void* stream) {
    device_check(p.owner->device);
    require(buffers(p.params,p.low,p.high,p.units,a,output,workspace,bytes,p.info.workspace_bytes),
        "dense buffers overlap, overflow, or violate alignment/workspace requirements");
    if (p.kind==Kind::Full) {
        auto c=full_call(p);c.weight.low=p.low;c.weight.high=p.high;c.weight.units=p.units;
        c.weight.low_bytes=p.params.sizes.low_bytes;c.weight.high_bytes=p.params.sizes.high_bytes;
        c.weight.unit_bytes=p.params.sizes.units_bytes;c.weight.stream=stream;
        c.a=a;c.output=output;c.workspace=workspace;c.workspace_bytes=bytes;
        if (p.typed_full) dense_full_check(quactlize_kpack_dense_full_run_v1(p.handle,&c,-1));
        else prefill_check(quactlize_kpack_prefill_run_dynamic_v1(p.handle,&c));
        return;
    }
    auto base=static_cast<uint8_t*>(workspace);
    void const *metadata=nullptr,*zero=nullptr;
    // Preserve the PoC enqueue order: SF -> input adapter -> TC -> output
    // adapter. Reordering these would change the measured cache conditions.
    expand_sf(p,base,stream,metadata,zero);
    float const* input=a;float* result=output;
    if (p.copy_input) {
        auto dst=reinterpret_cast<float*>(base+p.input_offset);
        copy_endpoint(p,true,a,dst,stream);
        input=dst;
    }
    if (p.copy_output) result=reinterpret_cast<float*>(base+p.output_offset);
    void* scratch=p.kernel_bytes ? base+p.kernel_offset : nullptr;
    if (p.kind==Kind::Simt || p.kind==Kind::Q4) {
        auto d=simt_call(p);auto& c=d.call;
        c.a=input;c.output=result;c.workspace=scratch;c.workspace_bytes=p.kernel_bytes;c.stream=stream;
        int rc=p.kind==Kind::Simt?quactlize_kpack_simt_run_v2(&d,&p.small.base.simt,&p.params.arrangement):
            quactlize_kpack_q4_decode_run_v2(&d,&p.small.q4,&p.params.arrangement);
        require(rc==QKG_OK,"selected dense SIMT launch failed");
    } else {
        qk_call_v1 c{1,sizeof(c)};c.m=p.params.m;c.n=p.params.n;c.k=p.params.k;c.experts=1;
        c.group_size=p.params.arrangement.group_size;c.device=p.owner->device;c.compute_units=p.tc.compute_units;
        c.mapping_id=p.params.arrangement.mapping_id;c.a=input;c.low=p.low;c.high=p.high;c.metadata=metadata;c.zero=zero;
        c.output=result;c.workspace=scratch;c.workspace_bytes=p.kernel_bytes;c.stream=stream;
        dispatch_check(quactlize_kpack_dispatch_run_dynamic_v1(p.handle,&c));
    }
    if (p.copy_output) copy_endpoint(p,false,result,output,stream);
}

template<class F> int boundary(F fn) {
    error.clear();
    try {fn();return 0;} catch (std::exception const& e) {error=e.what();}
    catch (...) {error="unknown dense runtime error";}
    return 1;
}
} // namespace quactlize::dense

extern "C" int quactlize_ppu_dense_supported_v1(quactlize_ppu_dense_params_v1 const* p) {
    quactlize::dense::error.clear();return quactlize::dense::admission(p);
}
extern "C" int quactlize_ppu_dense_create_with_options_v1(int device,quactlize_ppu_dense_options_v1 const* options,void** out) {
    using namespace quactlize::dense;
    if (out) *out=nullptr;
    return boundary([&] {
        require(out && device>=0 && options && options->version==1 && options->size==sizeof(*options) &&
            !options->reserved && (options->compute_type==QUACTLIZE_DENSE_F16 || options->compute_type==QUACTLIZE_DENSE_BF16),"invalid dense context ABI");
        device_check(device);
        hggcStreamCaptureStatus capture;
        require(hggcStreamIsCapturing(nullptr,&capture)==hggcSuccess && capture==hggcStreamCaptureStatusNone,"dense create during capture");
        auto ctx=std::make_unique<Context>();ctx->device=device;ctx->compute=options->compute_type;
        ctx->sdk=environment("PPU_SDK","/usr/local/PPU_SDK");ctx->root=runtime_directory();
        require(fs::path(ctx->sdk).is_absolute(),"PPU_SDK must be an absolute SDK root");
        dispatch_check(quactlize_kpack_dispatch_open_v1(ctx->root.c_str(),&ctx->dispatcher));
        *out=ctx.release();
    });
}
extern "C" int quactlize_ppu_dense_create_v1(int device,void** out) {
    quactlize_ppu_dense_options_v1 options{1,sizeof(options),QUACTLIZE_DENSE_F16,0};
    return quactlize_ppu_dense_create_with_options_v1(device,&options,out);
}
static int prepare_entry(void* context,quactlize_ppu_dense_params_v1 const* params,
    void const* low,void const* high,void const* units,quactlize_ppu_dense_candidate_v1 const* candidate,
    void** out,quactlize_ppu_dense_plan_info_v1* info) {
    using namespace quactlize::dense;if (out) *out=nullptr;
    return boundary([&] {
        require(context && out && info && info->version==1 && info->size==sizeof(*info),"invalid dense prepare ABI");
        require(admission(params)==1 && params->m>0,"unsupported dense shape/layout/strides");
        require(low && units && (params->sizes.high_bytes!=0)==(high!=nullptr) &&
            (uintptr_t(low)|uintptr_t(high)|uintptr_t(units))%16==0,"invalid dense weight planes");
        auto& ctx=*static_cast<Context*>(context);device_check(ctx.device);
        hggcStreamCaptureStatus capture;
        require(hggcStreamIsCapturing(nullptr,&capture)==hggcSuccess && capture==hggcStreamCaptureStatusNone,"dense prepare during capture");
        std::lock_guard<std::mutex> lock(ctx.mutex);
        auto p=std::make_unique<quactlize::dense::Plan>();p->owner=&ctx;p->params=*params;p->compute=ctx.compute;
        p->low=static_cast<uint8_t const*>(low);p->high=static_cast<uint8_t const*>(high);p->units=static_cast<uint8_t const*>(units);
        prepare(*p,candidate);auto raw=p.get();ctx.plans.push_back(std::move(p));*info=raw->info;*out=raw;
    });
}
extern "C" int quactlize_ppu_dense_prepare_v1(void* context,quactlize_ppu_dense_params_v1 const* params,
    void const* low,void const* high,void const* units,void** out,quactlize_ppu_dense_plan_info_v1* info) {
    return prepare_entry(context,params,low,high,units,nullptr,out,info);
}
extern "C" int quactlize_ppu_dense_prepare_candidate_v1(void* context,quactlize_ppu_dense_params_v1 const* params,
    void const* low,void const* high,void const* units,quactlize_ppu_dense_candidate_v1 const* candidate,
    void** out,quactlize_ppu_dense_plan_info_v1* info) {
    if (!candidate) {if (out) *out=nullptr;quactlize::dense::error="null dense candidate";return 1;}
    return prepare_entry(context,params,low,high,units,candidate,out,info);
}
extern "C" int quactlize_ppu_dense_run_v1(void* plan,float const* input,float* output,void* workspace,uint64_t bytes,void* stream) {
    using namespace quactlize::dense;
    return boundary([&] {require(plan,"null dense plan");run(*static_cast<quactlize::dense::Plan*>(plan),input,output,workspace,bytes,stream);});
}
extern "C" void quactlize_ppu_dense_destroy_v1(void* context) {delete static_cast<quactlize::dense::Context*>(context);}
extern "C" char const* quactlize_ppu_dense_error_v1(void) {return quactlize::dense::error.c_str();}
