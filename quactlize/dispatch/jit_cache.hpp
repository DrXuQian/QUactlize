#pragma once
#include "policy.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace quactlize::dispatch {
// Matches runtime/jit_cache.py. Only metadata is read on a disk hit; source
// hashing/compilation belongs to a miss. SDK/source installs must not change
// during a runtime's lifetime (the same rule as DeepGEMM's version cache).
inline std::string cache_stamp(std::filesystem::path const& path) {
    struct stat s{};
    if (stat(path.c_str(),&s)) throw std::runtime_error("JIT cache stat failed");
    std::ostringstream out;
    out<<s.st_dev<<' '<<s.st_ino<<' '<<s.st_size<<' '
       <<int64_t(s.st_mtim.tv_sec)*1000000000+s.st_mtim.tv_nsec<<' '
       <<int64_t(s.st_ctim.tv_sec)*1000000000+s.st_ctim.tv_nsec;
    return out.str();
}
inline std::string cache_environment(std::string const& sdk,std::string const& source) {
    std::istringstream search(std::getenv("PATH") ? std::getenv("PATH") : "");
    std::string directory,host;
    while (std::getline(search,directory,':')) {
        auto path=std::filesystem::path(directory)/"g++";
        if (!access(path.c_str(),X_OK) && std::filesystem::is_regular_file(path)) {
            host=std::filesystem::canonical(path).string(); break;
        }
    }
    if (host.empty()) throw std::runtime_error("JIT host compiler not found");
    std::string result="QK_PARENT_CACHE_V1\n"+source+"\n"+sdk+"\n"+host+"\n";
    for (auto file : {"bin/hgcc","bin/hgobjdump","lib/libhggc_wrapper.so",
                      "lib/libhggcrt1.so","lib/libhggc.so","lib/libhg_wrapper.so"})
        result+=cache_stamp(std::filesystem::path(sdk)/file)+"\n";
    return result+cache_stamp(host)+"\n";
}
inline std::string cache_name(std::string const& value) {
    // This names an index, not a module. A full request comparison below
    // makes hash collisions a cache miss rather than a different kernel.
    uint64_t hash=14695981039346656037ULL;
    for (unsigned char c : value) hash=(hash^c)*1099511628211ULL;
    std::ostringstream out;
    out<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;
    return out.str();
}
inline bool cache_digest(std::string const& s) {
    return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;
}
inline std::string cached_parent(std::string const& cache,std::string const& environment,
                                std::string const& source,Config const& c,bool dense,int compute) {
    if (environment.empty()) return {};
    std::ostringstream tuple;
    tuple<<c.qtype<<' '<<c.route<<' '<<c.tm<<' '<<c.tn<<' '<<c.tk<<' '<<c.wm<<' '<<c.wn<<' '
         <<c.stages<<' '<<c.ap<<' '<<c.dn<<' '<<c.parent_persistent<<' '<<int(dense)<<' '<<compute<<'\n';
    auto request=environment+c.symbol+"\n"+tuple.str();
    auto directory=std::filesystem::path(cache)/".jit-v1";
    auto path=directory/(cache_name(request)+".txt");
    try {
        if (std::filesystem::is_symlink(directory) || std::filesystem::is_symlink(path)) return {};
        std::ifstream file(path);
        char bytes[16385];
        file.read(bytes,sizeof(bytes));
        auto size=file.gcount();
        if (!file.eof() || size>16384) return {};
        std::string data(bytes,size);
        if (data.compare(0,request.size(),request)) return {};
        std::istringstream in(data.substr(request.size()));
        std::string library_stamp,manifest_stamp,tag,key,contract,observed,extra;
        if (!std::getline(in,library_stamp) || !std::getline(in,manifest_stamp) ||
            !(in>>tag>>key>>contract>>observed) || tag!="QK_JIT_V1" ||
            !cache_digest(key) || !cache_digest(contract) || observed!=source || (in>>extra)) return {};
        auto module=std::filesystem::path(cache)/key;
        auto library=module/"kernel.so",manifest=module/"manifest.json";
        for (auto const& p : {module,library,manifest})
            if (std::filesystem::is_symlink(p)) return {};
        if (cache_stamp(library)!=library_stamp || cache_stamp(manifest)!=manifest_stamp) return {};
        return tag+" "+key+" "+contract+" "+observed+"\n";
    } catch (std::exception const&) { return {}; } // Disposable index: use the compiler on a miss.
}
} // namespace quactlize::dispatch
