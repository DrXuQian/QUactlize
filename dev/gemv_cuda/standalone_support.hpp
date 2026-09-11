#pragma once
#include <cuda_runtime.h>
#include "quactlize/execution/api.h"
#include <dlfcn.h>
#include <cstdint>
#include <stdexcept>

inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void require(bool ok, char const* what) {
    if (!ok) throw std::runtime_error(what);
}
struct Header {
    uint64_t magic;
    int32_t version,q,n,k,experts,rows,channels,reserved;
    quactlize_ppu_placed_arrangement_v2 arrangement;
    uint64_t lengths[8];
};
static_assert(sizeof(Header)==144);
struct Device {
    void* ptr=nullptr;
    size_t bytes;
    explicit Device(size_t n):bytes(n) { check(cudaMalloc(&ptr,n)); }
    Device(Device const&)=delete;
    Device& operator=(Device const&)=delete;
    ~Device() { cudaFree(ptr); }
    void put(void const* src,size_t n,size_t offset=0) {
        require(offset<=bytes && n<=bytes-offset,"upload bounds");
        check(cudaMemcpy(static_cast<char*>(ptr)+offset,src,n,cudaMemcpyHostToDevice));
    }
};
struct Library {
    void* handle;
    explicit Library(char const* path):handle(dlopen(path,RTLD_NOW|RTLD_LOCAL)) {
        if (!handle) throw std::runtime_error(dlerror());
    }
    Library(Library const&)=delete;
    Library& operator=(Library const&)=delete;
    ~Library() { dlclose(handle); }
    template<class T> T symbol(char const* name) {
        auto value=dlsym(handle,name);
        if (!value) throw std::runtime_error(dlerror());
        return reinterpret_cast<T>(value);
    }
};
