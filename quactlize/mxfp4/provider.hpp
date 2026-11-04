#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace quactlize::mxfp4 {
// Address-free recipe. All pointers, including scheduler scratch, belong to
// the invocation. The provider never retains an activation or router tensor.
struct Provider {
    using Launch=void(*)(void const*,void const*,void const*,void*,int,int,void*,int,int const*,void*,int*);
    void* library=nullptr;
    Launch launch=nullptr;
    int rows=0,experts=0,compute_units=0;
    uint64_t workspace_bytes=0;
    std::string config;
    ~Provider();
    Provider()=default;
    Provider(Provider const&)=delete;
    Provider& operator=(Provider const&)=delete;
    void run(void const* a,void const* weights,void const* scales,void* c,
             int32_t const* counts,void* scratch,void* stream) const;
};
std::shared_ptr<Provider> resolve(std::filesystem::path const& runtime,std::string const& sdk,
                                int device,int rows,int n,int k,int experts);
}
