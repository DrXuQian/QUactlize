// Host test provider; never linked into a product library.
#include <cassert>
#include <cstddef>
#include <cstdint>
struct Blas { void* stream=nullptr;void* workspace=nullptr;size_t bytes=0; };
extern "C" int cublasCreate_v2(void** p) { *p=new Blas{};return 0; }
extern "C" int cublasDestroy_v2(void* p) { delete static_cast<Blas*>(p);return 0; }
extern "C" int cublasSetStream_v2(void* p,void* stream) {
    auto& b=*static_cast<Blas*>(p);b.stream=stream;b.workspace=nullptr;b.bytes=0;return 0;
}
extern "C" int cublasSetWorkspace_v2(void* p,void* workspace,size_t bytes) {
    auto& b=*static_cast<Blas*>(p);b.workspace=workspace;b.bytes=bytes;return 0;
}
extern "C" int cublasSetMathMode(void*,int mode) { return mode!=16; }
extern "C" int cublasGemmEx(void* p,int,int,int,int,int,void const*,void const* weight,int,int,
    void const* a,int,int,void const*,void* out,int,int,int,int) {
    auto const& b=*static_cast<Blas*>(p);
    assert(b.workspace && b.bytes==(4u<<20) && b.stream);
    *static_cast<float*>(b.workspace)=float(uintptr_t(b.stream));
    *static_cast<float*>(out)=*static_cast<float const*>(a)+*static_cast<float const*>(weight);
    return 0;
}
