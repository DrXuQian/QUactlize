#include "policy.hpp"
#include "jit.hpp"
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

namespace {
using namespace quactlize::dispatch;
struct Image {
    std::string parent, key, contract;
    int qtype, route, tm, tn, tk, wm, wn, stages, ap, dn;
    std::filesystem::path path;
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
    decltype(&quactlize_kpack_query_v1) query;
    decltype(&quactlize_kpack_prepare_v1) prepare;
    decltype(&quactlize_kpack_grouped_query_v2) query_device;
    decltype(&quactlize_kpack_grouped_prepare_v2) prepare_device;
    decltype(&quactlize_kpack_run_v1) run;
    decltype(&quactlize_kpack_destroy_v1) destroy;
    ~Module() { if (library) dlclose(library); }
};
struct Plan {
    qks_request_v1 request;
    qks_choice_v1 choice;
    qk_recipe_v1 recipe;
    std::shared_ptr<Module> module;
};
using Key=std::tuple<int,int,int,int,int,int,int,uint64_t>;
Key key(qks_request_v1 const& r) { return {r.qtype,r.route,r.m,r.n,r.k,r.experts,r.max_rows,r.mapping_id}; }
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

std::shared_ptr<Module> load(Runtime& r,Image const& image,Config const& c) {
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
    auto identity=symbol<decltype(&quactlize_kpack_identity_v1)>(module->library,"quactlize_kpack_identity_v1")();
    if (!identity || identity->version!=1 || identity->size!=sizeof(*identity) ||
        !identity->parent || !identity->build_key || identity->parent!=image.parent ||
        identity->build_key!=image.key || identity->qtype!=c.qtype || identity->route!=c.route ||
        identity->tm!=c.tm || identity->tn!=c.tn || identity->tk!=c.tk || identity->wm!=c.wm ||
        identity->wn!=c.wn || identity->stages!=c.stages || identity->ap!=c.ap ||
        identity->delivery_n!=c.dn || identity->mapping_id!=c.mapping_id)
        throw std::runtime_error("loaded parent/build identity differs");
    char name[128]{}; int device=-1,cu=0;
    auto probe=symbol<decltype(&quactlize_kpack_device_v1)>(module->library,"quactlize_kpack_device_v1");
    if (probe(name,sizeof(name),&device,&cu)!=QK_OK || std::strcmp(name,"PPU-ZW810") || cu!=72 ||
        (r.device>=0 && (r.device!=device || r.cu!=cu)))
        throw std::runtime_error("loaded module device differs from policy");
    r.device=device; r.cu=cu;
    module->query=symbol<decltype(module->query)>(module->library,"quactlize_kpack_query_v1");
    module->prepare=symbol<decltype(module->prepare)>(module->library,"quactlize_kpack_prepare_v1");
    module->query_device=symbol<decltype(module->query_device)>(module->library,"quactlize_kpack_grouped_query_v2");
    module->prepare_device=symbol<decltype(module->prepare_device)>(module->library,"quactlize_kpack_grouped_prepare_v2");
    module->run=symbol<decltype(module->run)>(module->library,"quactlize_kpack_run_v1");
    module->destroy=symbol<decltype(module->destroy)>(module->library,"quactlize_kpack_destroy_v1");
    r.modules.emplace(image.key,module);
    return module;
}
qk_call_v1 call_for(qks_request_v1 const& req,Runtime const& r) {
    qk_call_v1 c{};
    c.version=1; c.size=sizeof(c); c.m=req.m; c.n=req.n; c.k=req.k; c.experts=req.experts;
    c.group_size=(req.qtype==12 || req.qtype==13) ? 32 : 16;
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

extern "C" int quactlize_kpack_dispatch_query_v1(void* runtime,qks_request_v1 const* req,qks_choice_v1* out) {
    if (!runtime || !req || !out || !valid(*req)) return QKS_INVALID;
    *out={};
    try {
        auto& r=*static_cast<Runtime*>(runtime);
        std::lock_guard<std::mutex> lock(r.mutex);
        auto found=r.requests.find(key(*req));
        if (found!=r.requests.end()) { *out=r.plans.at(found->second-1).choice; return QKS_OK; }
        auto selected=select(*req);
        if (!selected.config) { last_error="no same-family policy choice"; return QKS_MISS; }
        auto const& config=*selected.config;
        Image const* image=nullptr;
        for (auto const& candidate : kImages) if (candidate.parent==config.symbol) { image=&candidate; break; }
        if (!image && r.jit.enabled()) {
            auto found=r.jit_images.find(config.symbol);
            if (found==r.jit_images.end()) {
                auto receipt=compile_parent(r.jit,config);
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
                    config.tk,config.wm,config.wn,config.stages,config.ap,config.dn,path};
                found=r.jit_images.emplace(config.symbol,std::move(resolved)).first;
            }
            image=&found->second;
        }
        if (!image) { last_error=std::string("selected parent not packaged: ")+config.symbol; return QKS_MISS; }
        auto module=load(r,*image,config);
        auto call=call_for(*req,r);
        auto rec=recipe(config,*req,1);
        qk_resources_v1 resources{};
        auto query=[&]() {
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
        r.plans.push_back({*req,choice,rec,module}); r.requests.emplace(key(*req),choice.ticket);
        *out=choice; return QKS_OK;
    } catch (std::exception const& e) { last_error=e.what(); return QKS_BINDING; }
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
        if (!same_choice(*choice,plan.choice) || call->version!=1 || call->size!=sizeof(*call) ||
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
extern "C" void quactlize_kpack_dispatch_destroy_v1(void* handle) { delete static_cast<Handle*>(handle); }
