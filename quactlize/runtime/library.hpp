#pragma once
#include "../dispatch/api.h"
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace quactlize::library {
inline void require(bool ok,char const* message) {
    if(!ok) throw std::runtime_error(message && *message?message:"Quactlize runtime failure");
}
inline void dispatch_check(int rc) { require(rc==QKS_OK,quactlize_kpack_dispatch_error_v1()); }
inline std::string environment(char const* name,std::string fallback) {
    auto p=std::getenv(name);return p && *p?p:std::move(fallback);
}
inline std::filesystem::path runtime_directory(void const* entry) {
    Dl_info image{};
    require(dladdr(entry,&image) && image.dli_fname,"cannot locate libncp_quactlize.so");
    return std::filesystem::canonical(image.dli_fname).parent_path()/"quactlize-runtime";
}
inline std::string python_executable() {
    namespace fs=std::filesystem;
    auto requested=environment("QUACTLIZE_PPU_PYTHON","");
    if(!requested.empty()) {
        require(fs::path(requested).is_absolute() && access(requested.c_str(),X_OK)==0,
                "QUACTLIZE_PPU_PYTHON must name an executable absolute path");
        return fs::canonical(requested).string();
    }
    auto path=environment("PATH","/usr/bin:/bin");size_t begin=0;
    for(;;) {
        auto end=path.find(':',begin);auto dir=path.substr(begin,end-begin);
        if(!dir.empty()) {
            auto candidate=fs::path(dir)/"python3";
            if(fs::is_regular_file(candidate) && access(candidate.c_str(),X_OK)==0)
                return fs::canonical(candidate).string();
        }
        if(end==std::string::npos) break;
        begin=end+1;
    }
    throw std::runtime_error("python3 not found for selected-parent JIT");
}
inline void enable_jit(void* dispatcher,std::filesystem::path const& root,std::string const& sdk) {
    auto cache=environment("QUACTLIZE_PPU_CACHE_DIR",environment("XDG_CACHE_HOME",
        environment("HOME","/tmp")+"/.cache")+"/quactlize/ppu");
    auto python=python_executable();auto helper=(root/"jit/tools/kpack_jit.py").string();
    require(std::filesystem::path(cache).is_absolute(),"QUACTLIZE_PPU_CACHE_DIR must be absolute");
    qks_jit_options_v1 options{1,sizeof(options),python.c_str(),helper.c_str(),sdk.c_str(),cache.c_str()};
    dispatch_check(quactlize_kpack_dispatch_enable_jit_v1(dispatcher,&options));
}
} // namespace quactlize::library
