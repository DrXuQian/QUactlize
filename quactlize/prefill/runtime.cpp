#include "layout.hpp"
#include "../dequant/indexed.h"
#include <hggc_runtime.h>
#include <dlfcn.h>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <spawn.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;
extern "C" int qkp_stage_prepare(qkp_call_v1 const*,quactlize::prefill::Layout const*,void*);
extern "C" int qkp_stage_finish(qkp_call_v1 const*,quactlize::prefill::Layout const*,void*);

namespace {
using namespace quactlize::prefill;
thread_local std::string error;
template<class T> T entry(void* so,char const* name) {
    auto p=reinterpret_cast<T>(dlsym(so,name));
    if (!p) throw std::runtime_error(std::string("provider missing ")+name);
    return p;
}
void check(int rc,char const* operation) {
    if (rc) throw std::runtime_error(std::string(operation)+" rc="+std::to_string(rc));
}
struct Library {
    std::string path;
    void* so;
    explicit Library(std::string const& input):path(std::filesystem::canonical(input).string()),
        so(dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL)) {
        if (!so) throw std::runtime_error(dlerror());
    }
    ~Library() { if (so) dlclose(so); }
};
using DeepLaunch=void(*)(void const*,void const*,void*,int const*,void*,int,int,void*,int,int,void*,int*);
struct Deep {
    std::shared_ptr<Library> lib;
    DeepLaunch launch;
    int sms,smem;
};

std::string prewarm(qkp_call_v1 const& c,qkp_options_v1 const& o) {
    if (!o.python || !o.deepgemm_helper || !std::filesystem::path(o.python).is_absolute() ||
        !std::filesystem::path(o.deepgemm_helper).is_absolute())
        throw std::runtime_error("DeepGEMM requires absolute Python/helper paths");
    std::vector<std::string> args{o.python,o.deepgemm_helper,"--m",std::to_string(c.m),
        "--n",std::to_string(c.weight.n),"--k",std::to_string(c.weight.k),
        "--experts",std::to_string(c.weight.experts),"--device",std::to_string(c.device),"--sdk",o.sdk};
    std::vector<char*> argv;for (auto& a:args) argv.push_back(a.data());argv.push_back(nullptr);
    int fd[2];if (pipe2(fd,O_CLOEXEC)) throw std::runtime_error("provider receipt pipe failed");
    posix_spawn_file_actions_t actions;
    int rc=posix_spawn_file_actions_init(&actions);
    if (rc) { close(fd[0]);close(fd[1]);throw std::runtime_error("provider spawn setup failed"); }
    rc=posix_spawn_file_actions_adddup2(&actions,fd[1],STDOUT_FILENO);
    if (!rc) rc=posix_spawn_file_actions_addclose(&actions,fd[0]);
    if (!rc) rc=posix_spawn_file_actions_addclose(&actions,fd[1]);
    pid_t pid=-1;
    if (!rc) rc=posix_spawn(&pid,o.python,&actions,nullptr,argv.data(),environ);
    posix_spawn_file_actions_destroy(&actions);close(fd[1]);
    if (rc) {close(fd[0]);throw std::runtime_error("provider spawn failed");}
    std::string reply;char data[1024];bool failed=false;
    for (;;) {
        auto n=read(fd[0],data,sizeof(data));
        if (n<0 && errno==EINTR) continue;
        if (n<0) failed=true;
        if (n<=0) break;
        if (reply.size()+size_t(n)>4096) failed=true;
        else reply.append(data,size_t(n));
    }
    close(fd[0]);int status=0;
    while (waitpid(pid,&status,0)<0) if (errno!=EINTR) throw std::runtime_error("provider wait failed");
    if (failed || !WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("installed DeepGEMM compile-only prewarm failed");
    return reply;
}

std::shared_ptr<Deep> deep(qkp_call_v1 const& c,qkp_options_v1 const& o,Layout const& layout) {
    static std::mutex mutex;
    static std::map<std::string,std::shared_ptr<Deep>> cache;
    std::string key=std::string(o.sdk)+"|"+(o.deepgemm_helper ? o.deepgemm_helper : "")+
        "|"+(o.python ? o.python : "")+
        "|"+std::to_string(c.device)+"|"+std::to_string(c.m)+"|"+std::to_string(c.weight.n)+"|"+std::to_string(c.weight.k);
    std::lock_guard<std::mutex> lock(mutex);
    auto found=cache.find(key);if (found!=cache.end()) return found->second;
    auto reply=prewarm(c,o);std::istringstream in(reply);
    std::string tag,path,hash,extra;int m,n,k,e,sms,smem;uint64_t bytes;
    if (!(in>>tag>>m>>n>>k>>e>>sms>>smem>>bytes) || tag!="QKP_DEEPGEMM_V1" ||
        m!=c.m || n!=c.weight.n || k!=c.weight.k || e!=c.weight.experts || sms!=72 || smem<0 ||
        bytes>layout.bytes-layout.directory) throw std::runtime_error("provider shape/scratch receipt differs");
    in.ignore(1);std::getline(in,path);std::getline(in,hash);
    if (!std::filesystem::path(path).is_absolute() || hash.size()!=64 ||
        hash.find_first_not_of("0123456789abcdef")!=std::string::npos || (in>>extra))
        throw std::runtime_error("invalid provider image receipt");
    auto p=std::make_shared<Deep>();p->lib=std::make_shared<Library>(path);
    p->launch=entry<DeepLaunch>(p->lib->so,"launch");p->sms=sms;p->smem=smem;
    cache[key]=p;return p;
}

struct Handle {
    qkp_call_v1 call;
    quactlize_ppu_placed_arrangement_v2 arr;
    Layout layout;
    std::shared_ptr<Deep> dg;
    std::shared_ptr<Library> blas_lib;
    void* blas=nullptr;
    int(*destroy)(void*)=nullptr;
    int(*stream)(void*,void*)=nullptr;
    using Gemm=int(*)(void*,int,int,int,int,int,void const*,void const*,int,int,void const*,int,int,void const*,void*,int,int,int,int);
    Gemm gemm=nullptr;
    ~Handle() {if (blas && destroy) destroy(blas);}
};
}

extern "C" int quactlize_kpack_prefill_query_v1(qkp_call_v1 const* c,
        quactlize_ppu_placed_arrangement_v2 const* arr,uint64_t* bytes) {
    try {
        if (!c || !bytes) return QKG_INVALID;
        *bytes=0;Layout l{};int rc=layout(*c,arr,l);
        if (!rc) { *bytes=l.bytes;error.clear(); }
        else error="prefill layout rc="+std::to_string(rc);
        return rc;
    }
    catch (std::exception const& e) {error=e.what();return QKG_OVERFLOW;}
}
extern "C" int quactlize_kpack_prefill_prepare_v1(qkp_call_v1 const* c,
        quactlize_ppu_placed_arrangement_v2 const* arr,qkp_options_v1 const* o,void** result) {
    if (result) *result=nullptr;
    try {
        if (!c || !o || !result || o->version!=1 || o->size!=sizeof(*o) || !o->sdk ||
            !std::filesystem::path(o->sdk).is_absolute()) return QKG_INVALID;
        auto h=std::make_unique<Handle>();check(layout(*c,arr,h->layout),"prefill layout");
        if (!c->a || !c->output || !c->workspace || c->workspace_bytes<h->layout.bytes ||
            uintptr_t(c->workspace)%256 || (c->weight.experts>1 && (!c->src_rows || !c->dst_rows || !c->offsets)))
            return QKG_CAPACITY;
        qkg_sizes_v1 sizes{};check(quactlize::execution::sizes(c->weight.qtype,c->weight.n,c->weight.k,c->weight.experts,arr,sizes),"weight size");
        if (c->weight.low_bytes<sizes.low_bytes || c->weight.high_bytes<sizes.high_bytes ||
            c->weight.unit_bytes<sizes.units_bytes || !c->weight.low || !c->weight.units ||
            (sizes.high_bytes && !c->weight.high)) return QKG_CAPACITY;
        uintptr_t pointers[]={uintptr_t(c->workspace),uintptr_t(c->weight.low),uintptr_t(c->weight.high),
            uintptr_t(c->weight.units),uintptr_t(c->output),uintptr_t(c->src_rows),uintptr_t(c->dst_rows),uintptr_t(c->offsets),
            uintptr_t(c->a)};
        uint64_t lengths[]={h->layout.bytes,sizes.low_bytes,sizes.high_bytes,sizes.units_bytes,
            (uint64_t(c->m-1)*c->output_stride+c->weight.n)*4,
            c->weight.experts>1 ? uint64_t(c->m)*4:0,c->weight.experts>1 ? uint64_t(c->m)*4:0,
            c->weight.experts>1 ? uint64_t(c->weight.experts+1)*4:0,
            (uint64_t(c->a_rows-1)*c->a_stride+c->weight.k)*4};
        for (int i=0;i<9;++i) if (lengths[i]) {
            if (!quactlize::execution::span(pointers[i],lengths[i]) || pointers[i]%(i<4 ? 16:4)) return QKG_INVALID;
            for (int j=0;j<i;++j)
                if (quactlize::execution::overlap(pointers[i],lengths[i],pointers[j],lengths[j])) return QKG_INVALID;
        }
        hggcStreamCaptureStatus capture;
        check(hggcStreamIsCapturing(static_cast<hggcStream_t>(c->weight.stream),&capture),"capture query");
        if (capture!=hggcStreamCaptureStatusNone) return QKG_INVALID;
        int device=-1;check(hggcGetDevice(&device),"device query");
        hggcDeviceProp prop{};check(hggcGetDeviceProperties(&prop,device),"device properties");
        int sms=0;check(hggcDeviceGetAttribute(&sms,hggcDevAttrMultiProcessorCount,device),"SM attribute");
        if (device!=c->device || sms!=72 || !std::strstr(prop.name,"PPU-ZW810")) return QKG_INVALID;
        h->call=*c;h->arr=*arr;
        if (c->weight.experts>1) h->dg=deep(*c,*o,h->layout);
        else {
            auto path=std::filesystem::path(o->sdk)/"CUDA_SDK/targets/x86_64-linux/lib/libcublas.so";
            h->blas_lib=std::make_shared<Library>(path.string());
            auto create=entry<int(*)(void**)>(h->blas_lib->so,"cublasCreate_v2");
            h->destroy=entry<decltype(h->destroy)>(h->blas_lib->so,"cublasDestroy_v2");
            h->stream=entry<decltype(h->stream)>(h->blas_lib->so,"cublasSetStream_v2");
            h->gemm=entry<Handle::Gemm>(h->blas_lib->so,"cublasGemmEx");
            check(create(&h->blas),"cuBLAS create");
            check(entry<int(*)(void*,int)>(h->blas_lib->so,"cublasSetMathMode")(h->blas,16),"cuBLAS FP32 reduction");
            check(h->stream(h->blas,c->weight.stream),"cuBLAS stream");
        }
        *result=h.release();error.clear();return 0;
    } catch (std::exception const& e) {error=e.what();return QKG_RUNTIME;}
}
extern "C" int quactlize_kpack_prefill_run_v1(void* handle,void* stream) {
    try {
        if (!handle) return QKG_INVALID;
        auto& h=*static_cast<Handle*>(handle);auto c=h.call;
        // Stream and scratch are immutable for a prepared cuBLAS handle.
        if (stream!=c.weight.stream) return QKG_INVALID;
        auto base=static_cast<uint8_t*>(c.workspace);
        check(qkp_stage_prepare(&c,&h.layout,stream),"prefill input");
        auto w=c.weight;w.output=base+h.layout.weights;w.zero=nullptr;
        w.output_bytes=uint64_t(w.n)*w.k*w.experts*2;w.stream=stream;
        if (h.dg) {
            check(quactlize_kpack_dequant_indexed_v1(&w,&h.arr,reinterpret_cast<int*>(base+h.layout.ids),
                reinterpret_cast<int*>(base+h.layout.count),reinterpret_cast<int*>(base+h.layout.error)),"active weight expansion");
            int rc=0;
            h.dg->launch(base+h.layout.a,w.output,base+h.layout.out,reinterpret_cast<int*>(base+h.layout.rows),
                base+h.layout.directory,c.m,(c.m+w.experts-1)/w.experts,stream,h.dg->sms,h.dg->smem,nullptr,&rc);
            check(rc,"DeepGEMM launch");check(hggcGetLastError(),"DeepGEMM immediate launch");
        } else {
            check(quactlize_kpack_dequant_v1(&w,&h.arr),"weight expansion");
            float alpha=1,beta=0;
            // Row-major A[M,K] * B[N,K]^T, BF16 A/B/output, FP32 compute.
            check(h.gemm(h.blas,1,0,w.n,c.m,w.k,&alpha,w.output,14,w.k,
                base+h.layout.a,14,w.k,&beta,base+h.layout.out,14,w.n,68,-1),"cuBLAS BF16 GEMM");
        }
        return qkp_stage_finish(&c,&h.layout,stream);
    } catch (std::exception const& e) {error=e.what();return QKG_RUNTIME;}
}
extern "C" void quactlize_kpack_prefill_destroy_v1(void* p) {delete static_cast<Handle*>(p);}
extern "C" int quactlize_kpack_prefill_device_status_v1(void* p,int32_t const** status) {
    if (!p || !status) return QKG_INVALID;
    auto& h=*static_cast<Handle*>(p);
    *status=reinterpret_cast<int32_t const*>(static_cast<uint8_t*>(h.call.workspace)+h.layout.error);
    return 0;
}
extern "C" char const* quactlize_kpack_prefill_provider_image_v1(void* p) {
    if (!p) return nullptr;
    auto& h=*static_cast<Handle*>(p);
    return (h.dg ? h.dg->lib:h.blas_lib)->path.c_str();
}
extern "C" char const* quactlize_kpack_prefill_error_v1() {return error.c_str();}
