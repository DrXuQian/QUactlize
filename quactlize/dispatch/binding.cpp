#include "policy.hpp"
#include "decode.hpp"
#include "smallm.hpp"
#include "jit.hpp"
#include "moe.hpp"
#include "../execution/moe.h"
#include "../execution/simt_validation.hpp"
#include <dlfcn.h>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

extern "C" int quactlize_kpack_dispatch_prefill_v1(qks_request_v1 const* r,
        uint32_t mask,qks_prefill_choice_v1* out) {
    if (!r || !out || !quactlize::dispatch::valid(*r)) return QKS_INVALID;
    return quactlize::dispatch::cost::query(*r,mask,*out);
}

namespace {
using namespace quactlize::dispatch;
struct Image {
    std::string parent, key, contract;
    int qtype, route, tm, tn, tk, wm, wn, stages, ap, dn;
    std::filesystem::path path;
    bool dense_io=false;
};
// Generated from full compiler receipts, not geometry-only config names.
#include "catalog.inc"
thread_local std::string last_error;

template<class T> T symbol(void* h,char const* name) {
    auto p=reinterpret_cast<T>(dlsym(h,name));
    if (!p) throw std::runtime_error(std::string("missing entry: ")+name);
    return p;
}
struct Module {
    void* library=nullptr;
    decltype(&quactlize_kpack_query_v1) query=nullptr;
    decltype(&quactlize_kpack_prepare_v1) prepare=nullptr;
    decltype(&quactlize_kpack_grouped_query_v2) query_device=nullptr;
    decltype(&quactlize_kpack_grouped_prepare_v2) prepare_device=nullptr;
    decltype(&quactlize_kpack_run_v1) run=nullptr;
    decltype(&quactlize_kpack_destroy_v1) destroy=nullptr;
    decltype(&quactlize_kpack_decode_dense_query_v1) query_dense_io=nullptr;
    decltype(&quactlize_kpack_decode_dense_prepare_v1) prepare_dense_io=nullptr;
    decltype(&quactlize_kpack_bind_llama_indexed_v1) bind_indexed=nullptr;
    decltype(&quactlize_kpack_moe_projection_v1) moe_projection=nullptr;
    decltype(&quactlize_kpack_moe_stage_v1) moe_stage=nullptr;
    ~Module() { if (library) dlclose(library); }
};
struct Plan {
    qks_request_v1 request;
    qks_choice_v1 choice;
    qk_recipe_v1 recipe;
    std::shared_ptr<Module> module;
    int endpoint_type=0;
};
struct MoeExecution {
    void* library=nullptr;
    decltype(&quactlize_kpack_moe_simt_query_v1) query=nullptr;
    decltype(&quactlize_kpack_moe_simt_bind_v1) bind=nullptr;
    decltype(&quactlize_kpack_moe_mixed_stage_v1) stage=nullptr;
    decltype(&quactlize_kpack_moe_weighted_finish_v1) finish=nullptr;
    decltype(&quactlize_kpack_q4_decode_select_v1) select=nullptr;
    decltype(&quactlize_kpack_q4_decode_run_v1) q4=nullptr;
    decltype(&quactlize_kpack_gemv_query_v1) query_generic=nullptr;
    decltype(&quactlize_kpack_gemv_run_v1) generic=nullptr;
    decltype(&quactlize_kpack_simt_query_v1) query_reuse=nullptr;
    decltype(&quactlize_kpack_simt_run_v1) reuse=nullptr;
    ~MoeExecution() { if (library) dlclose(library); }
};
using Key=std::tuple<int,int,int,int,int,int,int,uint64_t,int,int>;
Key key(qks_request_v1 const& r,int decode,int endpoint_type) {
    return {r.qtype,r.route,r.m,r.n,r.k,r.experts,r.max_rows,r.mapping_id,decode,endpoint_type};
}
struct Runtime {
    std::filesystem::path root;
    int device=-1, cu=0;
    std::mutex mutex;
    std::map<std::string,std::shared_ptr<Module>> modules;
    std::map<Key,uint64_t> requests;
    std::vector<Plan> plans;
    Jit jit;
    std::map<std::string,Image> jit_images;
    std::shared_ptr<MoeExecution> moe;
};
struct Handle {
    std::shared_ptr<Module> module;
    void* inner=nullptr;
    ~Handle() { if (inner) module->destroy(inner); }
};
struct MoeChain {
    Handle *gate=nullptr, *up=nullptr, *down=nullptr;
    qk_moe_plan_v1 plan{};
    std::shared_ptr<MoeExecution> execution;
    uint32_t simt_mask=0,q4_mask=0,reuse_mask=0;
    qkg_call_v1 calls[3]{};
    qkg_config_v1 configs[3]{};
    qkg_q4_decode_config_v1 q4_configs[3]{};
    qkg_simt_config_v1 reuse_configs[3]{};
    qkg_sizes_v1 simt_sizes[3]{};
    quactlize_ppu_placed_arrangement_v2 arrangements[3]{};
    qk_llama_moe_finish_v1 finish{};
};

bool overlaps_simt_workspace(MoeChain const& c,void const* pointer,uint64_t bytes) {
    MoeSpan candidate;
    if (!moe_span(pointer,bytes,candidate)) return true;
    for (int i=0;i<3;++i) if (c.simt_sizes[i].workspace_bytes) {
        MoeSpan workspace;
        if (!moe_span(c.calls[i].workspace,c.simt_sizes[i].workspace_bytes,workspace) ||
            moe_overlap(candidate,workspace)) return true;
    }
    return false;
}

bool compatible_simt_workspaces(MoeChain const& c) {
    if (!c.reuse_mask) return true;
    std::vector<MoeSpan> workspaces;
    for (int i=0;i<3;++i) if (c.simt_sizes[i].workspace_bytes) {
        MoeSpan span;
        if (!moe_span(c.calls[i].workspace,c.simt_sizes[i].workspace_bytes,span)) return false;
        for (auto other:workspaces) if (moe_overlap(span,other)) return false;
        workspaces.push_back(span);
    }
    auto disjoint=[&](void const* ptr,uint64_t bytes) {
        return !bytes || !overlaps_simt_workspace(c,ptr,bytes);
    };
    auto const& p=c.plan;
    if (!disjoint(p.gate.io.a,uint64_t(p.gate.io.a_token_stride)*p.gate.io.tokens*4) ||
        !disjoint(p.gate.io.ids,uint64_t(p.gate.io.ids_stride)*p.gate.io.tokens*4) ||
        !disjoint(p.down.io.output,uint64_t(p.down.io.out_row_stride)*p.down.m*4)) return false;
    qk_moe_projection_v1 const* parts[3]={&p.gate,p.merged?nullptr:&p.up,&p.down};
    for (int i=0;i<3;++i) {
        auto part=parts[i];if (!part) continue;
        uint64_t bytes=(c.simt_mask&(1u<<i))?4:2;
        if (!disjoint(part->a,uint64_t(part->m)*part->k*bytes) ||
            !disjoint(part->output,uint64_t(part->m)*part->n*bytes) ||
            !disjoint(part->workspace,part->workspace_bytes) ||
            !disjoint(part->offsets,uint64_t(part->experts+1)*4) ||
            !disjoint(part->io.row_ids,uint64_t(part->m)*4)) return false;
        if (c.simt_mask&(1u<<i)) {
            auto const& call=c.calls[i];auto const& sizes=c.simt_sizes[i];
            if (!disjoint(call.low,sizes.low_bytes) || !disjoint(call.high,sizes.high_bytes) ||
                !disjoint(call.units,sizes.units_bytes)) return false;
        }
    }
    return true;
}

std::shared_ptr<MoeExecution> load_moe(Runtime& r) {
    if (r.moe) return r.moe;
    auto lib=std::make_shared<MoeExecution>();
    lib->library=dlopen((r.root/"libquactlize_ppu_execution.so").c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!lib->library) throw std::runtime_error(dlerror());
    lib->query=symbol<decltype(lib->query)>(lib->library,"quactlize_kpack_moe_simt_query_v1");
    lib->bind=symbol<decltype(lib->bind)>(lib->library,"quactlize_kpack_moe_simt_bind_v1");
    lib->stage=symbol<decltype(lib->stage)>(lib->library,"quactlize_kpack_moe_mixed_stage_v1");
    lib->finish=reinterpret_cast<decltype(lib->finish)>(
        dlsym(lib->library,"quactlize_kpack_moe_weighted_finish_v1"));
    lib->select=symbol<decltype(lib->select)>(lib->library,"quactlize_kpack_q4_decode_select_v1");
    lib->q4=symbol<decltype(lib->q4)>(lib->library,"quactlize_kpack_q4_decode_run_v1");
    lib->query_generic=symbol<decltype(lib->query_generic)>(lib->library,"quactlize_kpack_gemv_query_v1");
    lib->generic=symbol<decltype(lib->generic)>(lib->library,"quactlize_kpack_gemv_run_v1");
    // Older execution libraries remain usable by v1/v2. Only a v3 caller
    // asking for this reader requires the new entries.
    lib->query_reuse=reinterpret_cast<decltype(lib->query_reuse)>(
        dlsym(lib->library,"quactlize_kpack_simt_query_v1"));
    lib->reuse=reinterpret_cast<decltype(lib->reuse)>(
        dlsym(lib->library,"quactlize_kpack_simt_run_v1"));
    r.moe=lib;return lib;
}

int run_mixed(MoeChain const& c,qk_moe_plan_v1 const& plan,void* stream) {
    auto produce=[&](int i,Handle* h) {
        if (!(c.simt_mask&(1u<<i)))
            return h->module->moe_stage(h->inner,&plan,QK_MOE_PRODUCER,stream)==QK_OK;
        auto call=c.calls[i];call.stream=stream;
        int rc;
        if (c.reuse_mask&(1u<<i))
            rc=c.execution->reuse(&call,&c.reuse_configs[i],&c.arrangements[i]);
        else if (c.q4_mask&(1u<<i))
            rc=c.execution->q4(&call,&c.q4_configs[i],&c.arrangements[i]);
        else rc=c.execution->generic(&call,&c.configs[i],&c.arrangements[i]);
        return rc==QKG_OK;
    };
    if (c.execution->stage(&plan,c.simt_mask,QK_MOE_PREPARE,stream)!=QKG_OK ||
        !produce(0,c.gate) || (!plan.merged && !produce(1,c.up)) ||
        c.execution->stage(&plan,c.simt_mask,QK_MOE_ACTIVATE,stream)!=QKG_OK ||
        !produce(2,c.down)) return QKS_RUNTIME;
    if (c.finish.version)
        return c.execution->finish(&plan,c.simt_mask,&c.finish,stream)==QKG_OK?QKS_OK:QKS_RUNTIME;
    if (!(c.simt_mask&4) && c.down->module->moe_stage(c.down->inner,&plan,QK_MOE_FINISH,stream)!=QK_OK)
        return QKS_RUNTIME;
    return QKS_OK;
}

std::shared_ptr<Module> load(Runtime& r,Image const& image,Config const& c,bool dense_io=false) {
    auto found=r.modules.find(image.key);
    if (found!=r.modules.end()) return found->second;
    if (image.qtype!=c.qtype || image.route!=c.route || image.tm!=c.tm || image.tn!=c.tn ||
        image.tk!=c.tk || image.wm!=c.wm || image.wn!=c.wn || image.stages!=c.stages ||
        image.ap!=c.ap || image.dn!=c.dn || !hex_digest(image.contract) || !hex_digest(image.key))
        throw std::runtime_error("catalog parent/compiler contract differs");
    auto path=image.path.empty() ? r.root/"modules"/image.key/"kernel.so" : image.path;
    auto module=std::make_shared<Module>();
    module->library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!module->library) throw std::runtime_error(dlerror());
    auto identity=symbol<decltype(&quactlize_kpack_identity_v1)>(module->library,
        dense_io?"quactlize_kpack_decode_dense_identity_v1":"quactlize_kpack_identity_v1")();
    if (!identity || identity->version!=1 || identity->size!=sizeof(*identity) ||
        !identity->parent || !identity->build_key || identity->parent!=image.parent ||
        identity->build_key!=image.key || identity->qtype!=c.qtype || identity->route!=c.route ||
        identity->tm!=c.tm || identity->tn!=c.tn || identity->tk!=c.tk || identity->wm!=c.wm ||
        identity->wn!=c.wn || identity->stages!=c.stages || identity->ap!=c.ap ||
        identity->delivery_n!=c.dn || identity->mapping_id!=c.mapping_id)
        throw std::runtime_error("loaded parent/build identity differs");
    char name[128]{}; int device=-1,cu=0;
    auto probe=symbol<decltype(&quactlize_kpack_device_v1)>(module->library,
        dense_io?"quactlize_kpack_decode_dense_device_v1":"quactlize_kpack_device_v1");
    int probe_rc=probe(name,sizeof(name),&device,&cu);
    if (probe_rc!=QK_OK || std::strcmp(name,"PPU-ZW810") || cu!=72 ||
        (r.device>=0 && (r.device!=device || r.cu!=cu)))
        throw std::runtime_error("loaded module device differs from policy: probe_rc="+std::to_string(probe_rc)+
            " name="+std::string(name)+" ordinal="+std::to_string(device)+" compute_units="+std::to_string(cu)+
            " expected_name=PPU-ZW810 expected_compute_units=72 previous_ordinal="+std::to_string(r.device)+
            " previous_compute_units="+std::to_string(r.cu));
    r.device=device; r.cu=cu;
    if (dense_io) {
        module->query_dense_io=symbol<decltype(module->query_dense_io)>(module->library,"quactlize_kpack_decode_dense_query_v1");
        module->prepare_dense_io=symbol<decltype(module->prepare_dense_io)>(module->library,"quactlize_kpack_decode_dense_prepare_v1");
        module->run=symbol<decltype(module->run)>(module->library,"quactlize_kpack_decode_dense_run_v1");
        module->destroy=symbol<decltype(module->destroy)>(module->library,"quactlize_kpack_decode_dense_destroy_v1");
    } else {
    module->query=symbol<decltype(module->query)>(module->library,"quactlize_kpack_query_v1");
    module->prepare=symbol<decltype(module->prepare)>(module->library,"quactlize_kpack_prepare_v1");
    module->query_device=symbol<decltype(module->query_device)>(module->library,"quactlize_kpack_grouped_query_v2");
    module->prepare_device=symbol<decltype(module->prepare_device)>(module->library,"quactlize_kpack_grouped_prepare_v2");
    module->run=symbol<decltype(module->run)>(module->library,"quactlize_kpack_run_v1");
    module->destroy=symbol<decltype(module->destroy)>(module->library,"quactlize_kpack_destroy_v1");
    module->bind_indexed=reinterpret_cast<decltype(module->bind_indexed)>(
        dlsym(module->library,"quactlize_kpack_bind_llama_indexed_v1"));
    module->moe_projection=reinterpret_cast<decltype(module->moe_projection)>(
        dlsym(module->library,"quactlize_kpack_moe_projection_v1"));
    module->moe_stage=reinterpret_cast<decltype(module->moe_stage)>(
        dlsym(module->library,"quactlize_kpack_moe_stage_v1"));
    }
    r.modules.emplace(image.key,module);
    return module;
}

Image const& jit_image(Runtime& r,Config const& config,bool dense_io=false) {
    std::string name=config.symbol;
    if (dense_io) name+="/dense-io";
    auto found=r.jit_images.find(name);
    if (found!=r.jit_images.end()) return found->second;
    auto receipt=compile_parent(r.jit,config,dense_io);
    std::istringstream in(receipt);
    std::string tag,build,contract,source,extra;
    if (!(in>>tag>>build>>contract>>source) || tag!="QK_JIT_V1" ||
        !hex_digest(build) || !hex_digest(contract) || source!=kJitSource || (in>>extra))
        throw std::runtime_error("JIT receipt identity differs");
    auto path=std::filesystem::canonical(std::filesystem::path(r.jit.cache)/build/"kernel.so");
    if (path.parent_path().parent_path()!=r.jit.cache || path.parent_path().filename()!=build ||
        path.filename()!="kernel.so" || !std::filesystem::is_regular_file(path))
        throw std::runtime_error("JIT module escapes its cache entry");
    Image resolved{config.symbol,build,contract,config.qtype,config.route,config.tm,config.tn,
        config.tk,config.wm,config.wn,config.stages,config.ap,config.dn,path,dense_io};
    return r.jit_images.emplace(name,std::move(resolved)).first->second;
}
qk_call_v1 call_for(qks_request_v1 const& req,Runtime const& r) {
    qk_call_v1 c{};
    c.version=1; c.size=sizeof(c); c.m=req.m; c.n=req.n; c.k=req.k; c.experts=req.experts;
    c.group_size=(req.qtype==8 || req.qtype==12 || req.qtype==13) ? 32 : 16;
    c.device=r.device; c.compute_units=r.cu; c.mapping_id=req.mapping_id;
    return c;
}
bool same_choice(qks_choice_v1 const& a,qks_choice_v1 const& b) {
    return a.version==b.version && a.size==b.size && a.ticket==b.ticket &&
        a.workspace_bytes==b.workspace_bytes && a.shared_bytes==b.shared_bytes &&
        a.policy==b.policy && a.algorithm==b.algorithm && a.split==b.split && a.grid==b.grid &&
        a.device==b.device && a.compute_units==b.compute_units &&
        std::memcmp(a.parent,b.parent,sizeof(a.parent))==0 &&
        std::memcmp(a.build_key,b.build_key,sizeof(a.build_key))==0;
}
} // namespace

extern "C" int quactlize_kpack_dispatch_q8_weight_supported_v1(
    int n,int k,int experts,int route,uint64_t mapping_id) {
    return quactlize::dispatch::q8_weight_supported(n,k,experts,route,mapping_id) ? 1 : 0;
}

extern "C" char const* quactlize_kpack_dispatch_error_v1() { return last_error.c_str(); }
extern "C" int quactlize_kpack_dispatch_open_v1(char const* root,void** out) {
    if (!out) return QKS_INVALID;
    *out=nullptr;
    try {
        if (!root || !*root) return QKS_INVALID;
        auto r=std::make_unique<Runtime>();
        r->root=std::filesystem::canonical(root);
        *out=r.release(); return QKS_OK;
    } catch (std::exception const& e) { last_error=e.what(); return QKS_BINDING; }
}
extern "C" void quactlize_kpack_dispatch_close_v1(void* runtime) { delete static_cast<Runtime*>(runtime); }

extern "C" int quactlize_kpack_dispatch_enable_jit_v1(void* runtime,qks_jit_options_v1 const* o) {
    if (!runtime || !o || o->version!=1 || o->size!=sizeof(*o) ||
        !o->python || !o->helper || !o->sdk || !o->cache) return QKS_INVALID;
    try {
        auto& r=*static_cast<Runtime*>(runtime);
        std::lock_guard<std::mutex> lock(r.mutex);
        if (r.jit.enabled() || !r.modules.empty() || !r.plans.empty() || !hex_digest(kJitSource)) return QKS_INVALID;
        for (auto path : {o->python,o->helper,o->sdk,o->cache})
            if (!std::filesystem::path(path).is_absolute()) return QKS_INVALID;
        Jit j{std::filesystem::canonical(o->python).string(),
              std::filesystem::canonical(o->helper).string(),
              std::filesystem::canonical(o->sdk).string(),{},kJitSource};
        if (access(j.python.c_str(),X_OK) || !std::filesystem::is_regular_file(j.helper) ||
            !std::filesystem::is_directory(j.sdk)) return QKS_INVALID;
        std::filesystem::create_directories(o->cache);
        j.cache=std::filesystem::canonical(o->cache).string();
        r.jit=std::move(j);
        return QKS_OK;
    } catch (std::exception const& e) { last_error=e.what(); return QKS_BINDING; }
}

static int query(void* runtime,qks_request_v1 const* req,qks_choice_v1* out,int decode,int endpoint_type=0,
                 Selected table={}) {
    if (!runtime || !req || !out || !valid(*req)) return QKS_INVALID;
    *out={};
    if (endpoint_type && (req->route>=2 || req->experts!=1 || req->m>8)) return QKS_MISS;
    try {
        auto& r=*static_cast<Runtime*>(runtime);
        std::lock_guard<std::mutex> lock(r.mutex);
        auto found=r.requests.find(key(*req,decode,endpoint_type));
        if (found!=r.requests.end()) { *out=r.plans.at(found->second-1).choice; return QKS_OK; }
        auto selected=table.config ? table : decode ? select_decode_tc(*req) : select(*req);
        if (!selected.config) { last_error="no same-family policy choice"; return QKS_MISS; }
        auto const& config=*selected.config;
        Image const* image=nullptr;
        for (auto const& candidate : kImages)
            if (candidate.parent==config.symbol && candidate.dense_io==(endpoint_type!=0)) { image=&candidate; break; }
        if (!image && r.jit.enabled()) image=&jit_image(r,config,endpoint_type!=0);
        if (!image) { last_error=std::string("selected parent not packaged: ")+config.symbol; return QKS_MISS; }
        auto module=load(r,*image,config,endpoint_type!=0);
        auto call=call_for(*req,r);
        auto rec=recipe(config,*req,1);
        qk_resources_v1 resources{};
        auto query=[&]() {
            if (endpoint_type) {
                qkd_dense_call_v1 typed{1,sizeof(typed),call,endpoint_type,endpoint_type};
                return module->query_dense_io(&typed,&rec,&resources);
            }
            qk_device_call_v2 d{2,sizeof(d),call,req->max_rows,0};
            return req->route>=2 ? module->query_device(&d,&rec,&resources) : module->query(&call,&rec,&resources);
        };
        int rc=query();
        if (rc!=QK_OK) { last_error="selected parent resource query rejected"; return rc==QK_UNSUPPORTED ? QKS_MISS : QKS_RUNTIME; }
        rec=recipe(config,*req,resources.occupancy);
        if (query()!=QK_OK) { last_error="selected recipe resource query rejected"; return QKS_RUNTIME; }
        qks_choice_v1 choice{};
        choice.version=1; choice.size=sizeof(choice); choice.ticket=r.plans.size()+1;
        choice.workspace_bytes=resources.workspace_bytes; choice.shared_bytes=resources.shared_bytes;
        choice.policy=selected.policy; choice.algorithm=rec.algorithm; choice.split=rec.split; choice.grid=rec.grid;
        choice.device=r.device; choice.compute_units=r.cu;
        if (image->parent.size()>=sizeof(choice.parent)) return QKS_BINDING;
        std::strcpy(choice.parent,image->parent.c_str()); std::strcpy(choice.build_key,image->key.c_str());
        r.plans.push_back({*req,choice,rec,module,endpoint_type});
        r.requests.emplace(key(*req,decode,endpoint_type),choice.ticket);
        *out=choice; return QKS_OK;
    } catch (std::exception const& e) { last_error=e.what(); return QKS_BINDING; }
}

extern "C" int quactlize_kpack_dispatch_query_v1(void* runtime,qks_request_v1 const* req,qks_choice_v1* out) {
    return query(runtime,req,out,false);
}
extern "C" int quactlize_kpack_dispatch_query_decode_v1(void* runtime,qks_request_v1 const* req,qks_choice_v1* out) {
    return query(runtime,req,out,true);
}

extern "C" int quactlize_kpack_dispatch_query_smallm_v1(void* runtime,qkg_call_v1 const* call,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,qks_smallm_choice_v1* out) {
    if (!runtime || !call || !out) return QKS_INVALID;
    *out={};
    int rc=smallm::validate(*call,arrangement);
    if (rc) return rc;
    auto selected=smallm::select(*call);
    if (!selected.row) return QKS_MISS;
    auto const& row=*selected.row;
    auto const& choice=smallm::data::kChoices[row.choice];
    qks_smallm_choice_v1 result{};
    result.version=1;result.size=sizeof(result);result.kind=choice.simt ? QKS_SMALLM_SIMT : QKS_SMALLM_TC;
    result.policy=selected.policy;result.source_n=row.n;result.source_k=row.k;result.source_tokens=row.tokens;
    if (choice.simt) {
        auto f=choice.reader;
        result.simt={1,sizeof(result.simt),f.variant,f.columns,f.warps,f.values,f.split};
        // Use the same query implementation as execution, without loading a
        // GPU library or compiling a TC parent that will not be used.
        rc=quactlize::execution::simt::query(*call,result.simt,arrangement,result.sizes);
        if (rc!=QKG_OK) return QKS_MISS;
    } else {
        auto const& c=*call;
        qks_request_v1 r{1,sizeof(r),c.qtype,choice.tc.route,c.rows,c.n,c.k,c.experts,
            smallm::tokens(c),arrangement->mapping_id};
        // The channel discriminator keeps shared/slot-specific proposals from
        // sharing a ticket with each other or with older selection entrypoints.
        rc=query(runtime,&r,&result.tc,2+c.channels,c.mode==QKG_DENSE ? QKD_F32 : 0,
                 {&choice.tc,selected.policy});
        if (rc!=QKS_OK) return rc;
    }
    *out=result;return QKS_OK;
}

extern "C" int quactlize_kpack_dispatch_query_dense_io_v1(void* runtime,qks_request_v1 const* req,
    int32_t endpoint_type,int32_t decode_policy,qks_choice_v1* out) {
    if ((endpoint_type!=QKD_F32 && endpoint_type!=QKD_BF16) || (decode_policy!=0 && decode_policy!=1))
        return QKS_INVALID;
    return query(runtime,req,out,decode_policy!=0,endpoint_type);
}

extern "C" int quactlize_kpack_dispatch_prepare_dense_io_v1(void* runtime,qks_choice_v1 const* choice,
    qkd_dense_call_v1 const* typed,void** out) {
    if (!out) return QKS_INVALID;*out=nullptr;
    if (!runtime || !choice || !typed || typed->version!=1 || typed->size!=sizeof(*typed)) return QKS_INVALID;
    try {
        auto& r=*static_cast<Runtime*>(runtime);
        std::lock_guard<std::mutex> lock(r.mutex);
        if (!choice->ticket || choice->ticket>r.plans.size()) return QKS_INVALID;
        auto const& plan=r.plans.at(choice->ticket-1);
        auto expected=call_for(plan.request,r);auto const& call=typed->call;
        if (!plan.endpoint_type || typed->input_type!=plan.endpoint_type || typed->output_type!=plan.endpoint_type ||
            !same_choice(*choice,plan.choice) || call.version!=1 || call.size!=sizeof(call) ||
            call.m!=expected.m || call.n!=expected.n || call.k!=expected.k || call.experts!=1 ||
            call.group_size!=expected.group_size || call.mapping_id!=expected.mapping_id ||
            call.device!=expected.device || call.compute_units!=expected.compute_units ||
            call.rows_host || call.rows_device || call.offsets_device || call.workspace_bytes<choice->workspace_bytes)
            return QKS_INVALID;
        auto h=std::make_unique<Handle>();h->module=plan.module;
        int rc=h->module->prepare_dense_io(typed,&plan.recipe,&h->inner);
        if (rc!=QK_OK) {last_error="typed decode prepare failed rc="+std::to_string(rc);return QKS_RUNTIME;}
        *out=h.release();return QKS_OK;
    } catch (std::exception const& e) {last_error=e.what();return QKS_RUNTIME;}
}

extern "C" int quactlize_kpack_dispatch_prepare_v1(void* runtime,qks_choice_v1 const* choice,
    qk_call_v1 const* call,void** out) {
    if (!out) return QKS_INVALID;
    *out=nullptr;
    if (!runtime || !choice || !call) return QKS_INVALID;
    try {
        auto& r=*static_cast<Runtime*>(runtime);
        std::lock_guard<std::mutex> lock(r.mutex);
        if (!choice->ticket || choice->ticket>r.plans.size()) return QKS_INVALID;
        auto const& plan=r.plans.at(choice->ticket-1);
        auto expected=call_for(plan.request,r);
        if (plan.endpoint_type || !same_choice(*choice,plan.choice) || call->version!=1 || call->size!=sizeof(*call) ||
            call->m!=expected.m || call->n!=expected.n || call->k!=expected.k || call->experts!=expected.experts ||
            call->group_size!=expected.group_size || call->mapping_id!=expected.mapping_id ||
            call->device!=expected.device || call->compute_units!=expected.compute_units ||
            call->rows_host || call->rows_device || call->workspace_bytes<choice->workspace_bytes) return QKS_INVALID;
        auto handle=std::make_unique<Handle>(); handle->module=plan.module;
        qk_device_call_v2 d{2,sizeof(d),*call,plan.request.max_rows,0};
        int rc=plan.request.route>=2 ? plan.module->prepare_device(&d,&plan.recipe,&handle->inner) :
            plan.module->prepare(call,&plan.recipe,&handle->inner);
        if (rc!=QK_OK) {
            last_error="selected handle preparation failed rc="+std::to_string(rc)+
                (rc==QK_UNSUPPORTED ? " (QK_UNSUPPORTED)" : "")+
                " parent="+std::string(plan.choice.parent)+
                " route="+std::to_string(plan.request.route)+
                " shape="+std::to_string(call->m)+"x"+std::to_string(call->n)+"x"+std::to_string(call->k)+
                " experts="+std::to_string(call->experts)+" max_rows="+std::to_string(plan.request.max_rows)+
                " split="+std::to_string(plan.recipe.split)+" grid="+std::to_string(plan.recipe.grid);
            return QKS_RUNTIME;
        }
        *out=handle.release(); return QKS_OK;
    } catch (std::exception const& e) { last_error=e.what(); return QKS_RUNTIME; }
}
extern "C" int quactlize_kpack_dispatch_run_v1(void* handle,void* stream) {
    if (!handle) return QKS_INVALID;
    auto& h=*static_cast<Handle*>(handle);
    return h.module->run(h.inner,stream)==QK_OK ? QKS_OK : QKS_RUNTIME;
}
extern "C" int quactlize_kpack_dispatch_bind_llama_indexed_v1(void* handle,qk_llama_indexed_v1 const* io) {
    if (!handle || !io) return QKS_INVALID;
    auto& h=*static_cast<Handle*>(handle);
    if (!h.module->bind_indexed) return QKS_MISS;
    int rc=h.module->bind_indexed(h.inner,io);
    if (rc==QK_OK) return QKS_OK;
    return rc==QK_UNSUPPORTED ? QKS_MISS : rc==QK_INVALID ? QKS_INVALID : QKS_RUNTIME;
}
extern "C" void quactlize_kpack_dispatch_destroy_v1(void* handle) { delete static_cast<Handle*>(handle); }
extern "C" int quactlize_kpack_dispatch_moe_create_v1(void* gate,void* up,void* down,void** out) {
    if (!gate || !down || !out || gate==down || (up && (up==gate || up==down))) return QKS_INVALID;
    *out=nullptr;
    try {
        auto chain=std::make_unique<MoeChain>();
        chain->gate=static_cast<Handle*>(gate); chain->up=static_cast<Handle*>(up);
        chain->down=static_cast<Handle*>(down);
        chain->plan.version=1; chain->plan.size=sizeof(chain->plan); chain->plan.merged=up?0:1;
        auto get=[](Handle* h,qk_moe_projection_v1& p) {
            return h->module->moe_projection && h->module->moe_stage &&
                h->module->moe_projection(h->inner,&p)==QK_OK;
        };
        if (!get(chain->gate,chain->plan.gate) || (up && !get(chain->up,chain->plan.up)) ||
            !get(chain->down,chain->plan.down)) return QKS_MISS;
        if (!compatible_moe(chain->plan)) {
            last_error="MoE chain geometry, routing, tuple ABI or scratch alias differs";
            return QKS_MISS;
        }
        *out=chain.release(); return QKS_OK;
    } catch (std::exception const& e) { last_error=e.what(); return QKS_RUNTIME; }
}
extern "C" int quactlize_kpack_dispatch_moe_run_v1(void* handle,void* stream) {
    if (!handle) return QKS_INVALID;
    auto& c=*static_cast<MoeChain*>(handle);
    if (c.simt_mask) return run_mixed(c,c.plan,stream);
    auto stage=[&](Handle* h,int phase) { return h->module->moe_stage(h->inner,&c.plan,phase,stream)==QK_OK; };
    if (!stage(c.gate,QK_MOE_PREPARE) || !stage(c.gate,QK_MOE_PRODUCER) ||
        (c.up && !stage(c.up,QK_MOE_PRODUCER)) || !stage(c.gate,QK_MOE_ACTIVATE) ||
        !stage(c.down,QK_MOE_PRODUCER)) return QKS_RUNTIME;
    if (c.finish.version)
        return c.execution->finish(&c.plan,0,&c.finish,stream)==QKG_OK?QKS_OK:QKS_RUNTIME;
    if (!stage(c.down,QK_MOE_FINISH)) return QKS_RUNTIME;
    return QKS_OK;
}
extern "C" int quactlize_kpack_dispatch_moe_run_router_v1(void* handle,qk_llama_router_v1 const* router,void* stream) {
    if (!handle || !router) return QKS_INVALID;
    auto& c=*static_cast<MoeChain*>(handle);
    auto const& r=*router;
    if (!compatible_router(c.plan,r,c.simt_mask)) return QKS_MISS;
    if (c.reuse_mask && (overlaps_simt_workspace(c,r.logits,uint64_t(c.plan.gate.io.tokens)*256*4) ||
        (r.bias && overlaps_simt_workspace(c,r.bias,256*4)) ||
        overlaps_simt_workspace(c,r.weights,uint64_t(c.plan.gate.m)*4))) return QKS_MISS;
    if (c.finish.version && (c.finish.weights!=r.weights || c.finish.weights_stride!=c.plan.down.io.topk))
        return QKS_MISS;
    // Copy the immutable host plan. Concurrent streams must not mutate its
    // router or retain a ready flag across graph replays.
    auto plan=c.plan; plan.router=r;
    if (c.simt_mask) return run_mixed(c,plan,stream);
    auto stage=[&](Handle* h,int phase) { return h->module->moe_stage(h->inner,&plan,phase,stream)==QK_OK; };
    if (!stage(c.gate,QK_MOE_PREPARE) || !stage(c.gate,QK_MOE_PRODUCER) ||
        (c.up && !stage(c.up,QK_MOE_PRODUCER)) || !stage(c.gate,QK_MOE_ACTIVATE) ||
        !stage(c.down,QK_MOE_PRODUCER)) return QKS_RUNTIME;
    if (c.finish.version)
        return c.execution->finish(&plan,0,&c.finish,stream)==QKG_OK?QKS_OK:QKS_RUNTIME;
    if (!stage(c.down,QK_MOE_FINISH)) return QKS_RUNTIME;
    return QKS_OK;
}
extern "C" void quactlize_kpack_dispatch_moe_destroy_v1(void* handle) { delete static_cast<MoeChain*>(handle); }

extern "C" int quactlize_kpack_dispatch_moe_bind_finish_v1(void* runtime,void* handle,
    qk_llama_moe_finish_v1 const* finish) {
    if (!runtime || !handle || !finish) return QKS_INVALID;
    auto& c=*static_cast<MoeChain*>(handle);
    if (c.finish.version) return QKS_INVALID;
    if (!compatible_finish(c.plan,*finish,c.simt_mask)) return QKS_MISS;
    if (c.reuse_mask && (overlaps_simt_workspace(c,finish->weights,
            uint64_t(finish->weights_stride)*c.plan.down.io.tokens*4) ||
        overlaps_simt_workspace(c,finish->output,
            uint64_t(finish->output_stride)*c.plan.down.io.tokens*4))) return QKS_MISS;
    try {
        auto& r=*static_cast<Runtime*>(runtime);std::lock_guard<std::mutex> lock(r.mutex);
        if (r.device>=0 && r.device!=c.plan.down.device) return QKS_MISS;
        auto execution=load_moe(r);
        if (!execution->finish) return QKS_MISS;
        c.execution=execution;c.finish=*finish;
        return QKS_OK;
    } catch (std::exception const& e) {last_error=e.what();return QKS_BINDING;}
}

extern "C" int quactlize_kpack_dispatch_moe_simt_scratch_v1(void* runtime,qkg_call_v1 const* call,uint64_t* bytes) {
    if (!runtime || !call || !bytes) return QKS_INVALID;
    *bytes=0;
    try {
        auto& r=*static_cast<Runtime*>(runtime);std::lock_guard<std::mutex> lock(r.mutex);
        int rc=load_moe(r)->query(call,bytes);
        return rc==QKG_OK?QKS_OK:rc==QKG_SHAPE?QKS_MISS:QKS_INVALID;
    } catch (std::exception const& e) { last_error=e.what();return QKS_BINDING; }
}

static int create_mixed_moe(void* runtime,qks_moe_endpoint_v2 const* gate,
    qks_moe_endpoint_v2 const* up,qks_moe_endpoint_v2 const* down,
    qkg_simt_config_v1 const* const (&reuse)[3],void** out) {
    if (!out) return QKS_INVALID;
    *out=nullptr;
    if (!runtime || !gate || !down || gate==down || (up && (up==gate || up==down))) return QKS_INVALID;
    qks_moe_endpoint_v2 const* endpoints[3]={gate,up,down};
    for (int i=0;i<3;++i) {
        auto e=endpoints[i];if (!e) continue;
        if (e->version!=2 || e->size!=sizeof(*e) ||
        bool(e->tc_handle)==bool(e->simt_call) ||
        (e->tc_handle && (e->q4_config || e->simt_config || reuse[i] || e->arrangement || e->scratch || e->scratch_bytes)) ||
        (e->simt_call && (!e->arrangement ||
            int(bool(e->q4_config))+int(bool(e->simt_config))+int(bool(reuse[i]))!=1))) return QKS_INVALID;
    }
    if (gate->tc_handle && down->tc_handle && (!up || up->tc_handle))
        return quactlize_kpack_dispatch_moe_create_v1(gate->tc_handle,up?up->tc_handle:nullptr,down->tc_handle,out);
    try {
        auto& r=*static_cast<Runtime*>(runtime);std::lock_guard<std::mutex> lock(r.mutex);
        auto c=std::make_unique<MoeChain>();c->execution=load_moe(r);
        c->plan.version=1;c->plan.size=sizeof(c->plan);c->plan.merged=up?0:1;
        qk_moe_projection_v1* parts[3]={&c->plan.gate,&c->plan.up,&c->plan.down};
        Handle** handles[3]={&c->gate,&c->up,&c->down};
        // Resolve a context from an existing TC handle, if present. All-SIMT
        // chains use the execution library's device probe at bind below.
        int device=r.device;
        for (int i=0;i<3;++i) if (endpoints[i] && endpoints[i]->tc_handle) {
            auto h=static_cast<Handle*>(endpoints[i]->tc_handle);*handles[i]=h;
            if (!h->module->moe_projection || !h->module->moe_stage ||
                h->module->moe_projection(h->inner,parts[i])!=QK_OK) return QKS_MISS;
            if (device>=0 && device!=parts[i]->device) return QKS_INVALID;
            device=parts[i]->device;
        }
        for (int i=0;i<3;++i) {
            auto e=endpoints[i];if (!e || !e->simt_call) continue;
            auto call=*e->simt_call;qkg_sizes_v1 sizes{};
            if (reuse[i]) {
                if (!c->execution->query_reuse || !c->execution->reuse) {
                    last_error="execution library lacks the register-reuse reader";return QKS_MISS;
                }
                if (c->execution->query_reuse(&call,reuse[i],e->arrangement,&sizes)!=QKG_OK)
                    return QKS_INVALID;
                c->reuse_mask|=1u<<i;c->reuse_configs[i]=*reuse[i];
            } else if (e->q4_config) {
                qkg_q4_decode_config_v1 selected{};
                int rc=c->execution->select(&call,e->arrangement,&selected,&sizes);
                if (rc!=QKG_OK) {last_error="mixed MoE Q4 selection rejected";return QKS_MISS;}
                if (std::memcmp(&selected,e->q4_config,sizeof(selected))) return QKS_INVALID;
                c->q4_mask|=1u<<i;c->q4_configs[i]=selected;
            } else {
                if (c->execution->query_generic(&call,e->simt_config,e->arrangement,&sizes)!=QKG_OK)
                    return QKS_INVALID;
                c->configs[i]=*e->simt_config;
            }
            // The SIMT down projection consumes one activation per slot.
            if (i==2 && call.channels!=call.topk) return QKS_MISS;
            if (c->execution->bind(&call,device,e->scratch,e->scratch_bytes,parts[i])!=QKG_OK)
                return QKS_INVALID;
            if (device<0) device=parts[i]->device;
            if (parts[i]->device!=device) return QKS_INVALID;
            if (i==2) {
                call.a=parts[i]->a;call.a_row_stride=call.k;
                call.a_token_stride=int64_t(call.topk)*call.k;
            } else { call.output=static_cast<float*>(parts[i]->output);call.out_row_stride=call.n; }
            if (reuse[i] && quactlize::execution::simt::buffers(call,sizes)!=QKG_OK) return QKS_INVALID;
            c->simt_sizes[i]=sizes;
            c->calls[i]=call;c->arrangements[i]=*e->arrangement;c->simt_mask|=1u<<i;
        }
        if (!compatible_moe(c->plan,c->simt_mask)) {
            last_error="mixed MoE row/type/tuple/scratch contract differs";return QKS_MISS;
        }
        if (!compatible_simt_workspaces(*c)) {
            last_error="SIMT Split-K workspace aliases live MoE storage";return QKS_INVALID;
        }
        *out=c.release();return QKS_OK;
    } catch (std::exception const& e) {last_error=e.what();return QKS_BINDING;}
}

extern "C" int quactlize_kpack_dispatch_moe_create_v2(void* runtime,qks_moe_endpoint_v2 const* gate,
    qks_moe_endpoint_v2 const* up,qks_moe_endpoint_v2 const* down,void** out) {
    qkg_simt_config_v1 const* reuse[3]={};
    return create_mixed_moe(runtime,gate,up,down,reuse,out);
}

extern "C" int quactlize_kpack_dispatch_moe_create_v3(void* runtime,qks_moe_endpoint_v3 const* gate,
    qks_moe_endpoint_v3 const* up,qks_moe_endpoint_v3 const* down,void** out) {
    if (!out) return QKS_INVALID;
    *out=nullptr;
    if (!runtime || !gate || !down || gate==down || (up && (up==gate || up==down))) return QKS_INVALID;
    qks_moe_endpoint_v3 const* inputs[3]={gate,up,down};
    qks_moe_endpoint_v2 endpoints[3]{};qkg_simt_config_v1 const* reuse[3]={};
    for (int i=0;i<3;++i) if (auto e=inputs[i]) {
        if (e->version!=3 || e->size!=sizeof(*e)) return QKS_INVALID;
        endpoints[i]={2,sizeof(endpoints[i]),e->tc_handle,e->simt_call,e->q4_config,
            e->simt_config,e->arrangement,e->scratch,e->scratch_bytes};
        reuse[i]=e->reuse_config;
    }
    return create_mixed_moe(runtime,&endpoints[0],up?&endpoints[1]:nullptr,&endpoints[2],reuse,out);
}
