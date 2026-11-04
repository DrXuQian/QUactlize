#include "provider.hpp"
#include "../runtime/library.hpp"
#include <hggc_runtime_api.h>
#include <cerrno>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <spawn.h>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;
namespace quactlize::mxfp4 {
using library::require;
namespace {
// Same prepare-only subprocess boundary as selected-parent JIT. No fork,
// shell, Python runtime, subprocess or receipt parsing on the inference path.
std::string prewarm(std::vector<std::string> args) {
    std::vector<char*> argv;for(auto& a:args) argv.push_back(a.data());argv.push_back(nullptr);
    int fd[2];require(!pipe2(fd,O_CLOEXEC),"MXFP4 receipt pipe failed");
    posix_spawn_file_actions_t actions;
    int rc=posix_spawn_file_actions_init(&actions);
    if(rc) {close(fd[0]);close(fd[1]);throw std::runtime_error("MXFP4 spawn setup failed");}
    rc=posix_spawn_file_actions_adddup2(&actions,fd[1],STDOUT_FILENO);
    if(!rc) rc=posix_spawn_file_actions_addclose(&actions,fd[0]);
    if(!rc) rc=posix_spawn_file_actions_addclose(&actions,fd[1]);
    pid_t pid=-1;
    if(!rc) rc=posix_spawn(&pid,argv[0],&actions,nullptr,argv.data(),environ);
    posix_spawn_file_actions_destroy(&actions);close(fd[1]);
    if(rc) {close(fd[0]);throw std::runtime_error("MXFP4 provider spawn failed");}
    std::string reply;char data[1024];bool failed=false;
    for(;;) {
        auto n=read(fd[0],data,sizeof(data));
        if(n<0 && errno==EINTR) continue;
        if(n<0) failed=true;
        if(n<=0) break;
        if(reply.size()+size_t(n)>4096) failed=true;
        else if(!failed) reply.append(data,size_t(n));
    }
    close(fd[0]);int status=0;
    while(waitpid(pid,&status,0)<0) if(errno!=EINTR) throw std::runtime_error("MXFP4 provider wait failed");
    require(!failed && WIFEXITED(status) && !WEXITSTATUS(status),
            "MXFP4 Sail W4FA16 compile-only preparation failed; see preceding Python/compiler diagnostic");
    return reply;
}
}
Provider::~Provider() {if(library) dlclose(library);}
void Provider::run(void const* a,void const* weights,void const* scales,void* c,
                   int32_t const* counts,void* scratch,void* stream) const {
    int rc=0;
    launch(a,weights,scales,c,rows,(rows+experts-1)/experts,stream,compute_units,counts,scratch,&rc);
    require(!rc,"MXFP4 W4FA16 launch rejected");
    require(hggcPeekAtLastError()==hggcSuccess,"MXFP4 W4FA16 kernel launch failed");
}
std::shared_ptr<Provider> resolve(std::filesystem::path const& runtime,std::string const& sdk,
                                int device,int rows,int n,int k,int experts) {
    int sms=0;
    require(hggcDeviceGetAttribute(&sms,hggcDevAttrMultiProcessorCount,device)==hggcSuccess && sms>0,
            "MXFP4 device CU query failed");
    auto python=library::python_executable();
    auto helper=std::filesystem::canonical(runtime/"jit/tools/kpack_mxfp4_prewarm.py").string();
    std::vector<std::string> args{python,helper,"--sdk",sdk,"--device",std::to_string(device),
        "--m",std::to_string(rows),"--n",std::to_string(n),"--k",std::to_string(k),"--experts",std::to_string(experts)};
    // Repeated layer plans share code, not scratch or tensor addresses. Weak
    // ownership unloads images after their last context/plan is destroyed.
    static std::mutex mutex;
    static std::map<std::vector<std::string>,std::weak_ptr<Provider>> cache;
    std::lock_guard<std::mutex> guard(mutex);
    if(auto p=cache[args].lock()) return p;
    std::istringstream in(prewarm(args));std::string tag,path,config,extra;
    int rm=0,rn=0,rk=0,re=0,rsms=0;uint64_t bytes=0;
    require(bool(in>>tag>>rm>>rn>>rk>>re>>rsms>>bytes) && tag=="QZ_MXFP4_V1" &&
        rm==rows && rn==n && rk==k && re==experts && rsms==sms && bytes>0 &&
        bytes<=uint64_t(rows+experts)*16 && bytes%16==0,"MXFP4 provider shape/scratch receipt differs");
    in.ignore(1);std::getline(in,path);std::getline(in,config);
    require(std::filesystem::path(path).is_absolute() && config.rfind("w4fa16:",0)==0 &&
        config.size()<192 && !(in>>extra),"invalid MXFP4 provider image receipt");
    auto p=std::make_shared<Provider>();p->library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!p->library) throw std::runtime_error(std::string("MXFP4 provider dlopen: ")+dlerror());
    p->launch=reinterpret_cast<Provider::Launch>(dlsym(p->library,"launch"));
    require(p->launch,"MXFP4 provider missing dynamic launch");
    p->rows=rows;p->experts=experts;p->compute_units=sms;p->workspace_bytes=bytes;p->config=config;
    cache[args]=p;return p;
}
}
