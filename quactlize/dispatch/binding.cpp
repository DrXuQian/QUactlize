#include "policy.hpp"
#include "decode.hpp"
#include "jit.hpp"
#include "moe.hpp"
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
using Key=std::tuple<int,int,int,int,int,int,int,uint64_t,bool,int>;
Key key(qks_request_v1 const& r,bool decode,int endpoint_type) {
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
};
struct Handle {
    std::shared_ptr<Module> module;
    void* inner=nullptr;
    ~Handle() { if (inner) module->destroy(inner); }
};
struct MoeChain {
    Handle *gate, *up, *down;
    qk_moe_plan_v1 plan{};
};

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

static int query(void* runtime,qks_request_v1 const* req,qks_choice_v1* out,bool decode,int endpoint_type=0) {
    if (!runtime || !req || !out || !valid(*req)) return QKS_INVALID;
    *out={};
    if (endpoint_type && (req->route>=2 || req->experts!=1 || req->m>8)) return QKS_MISS;
    try {
        auto& r=*static_cast<Runtime*>(runtime);
        std::lock_guard<std::mutex> lock(r.mutex);
        auto found=r.requests.find(key(*req,decode,endpoint_type));
        if (found!=r.requests.end()) { *out=r.plans.at(found->second-1).choice; return QKS_OK; }
        auto selected=decode ? select_decode_tc(*req) : select(*req);
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
    auto stage=[&](Handle* h,int phase) { return h->module->moe_stage(h->inner,&c.plan,phase,stream)==QK_OK; };
    if (!stage(c.gate,QK_MOE_PREPARE) || !stage(c.gate,QK_MOE_PRODUCER) ||
        (c.up && !stage(c.up,QK_MOE_PRODUCER)) || !stage(c.gate,QK_MOE_ACTIVATE) ||
        !stage(c.down,QK_MOE_PRODUCER) || !stage(c.down,QK_MOE_FINISH)) return QKS_RUNTIME;
    return QKS_OK;
}
extern "C" int quactlize_kpack_dispatch_moe_run_router_v1(void* handle,qk_llama_router_v1 const* router,void* stream) {
    if (!handle || !router) return QKS_INVALID;
    auto& c=*static_cast<MoeChain*>(handle);
    auto const& r=*router;
    if (!compatible_router(c.plan,r)) return QKS_MISS;
    // Copy the immutable host plan. Concurrent streams must not mutate its
    // router or retain a ready flag across graph replays.
    auto plan=c.plan; plan.router=r;
    auto stage=[&](Handle* h,int phase) { return h->module->moe_stage(h->inner,&plan,phase,stream)==QK_OK; };
    if (!stage(c.gate,QK_MOE_PREPARE) || !stage(c.gate,QK_MOE_PRODUCER) ||
        (c.up && !stage(c.up,QK_MOE_PRODUCER)) || !stage(c.gate,QK_MOE_ACTIVATE) ||
        !stage(c.down,QK_MOE_PRODUCER) || !stage(c.down,QK_MOE_FINISH)) return QKS_RUNTIME;
    return QKS_OK;
}
extern "C" void quactlize_kpack_dispatch_moe_destroy_v1(void* handle) { delete static_cast<MoeChain*>(handle); }
