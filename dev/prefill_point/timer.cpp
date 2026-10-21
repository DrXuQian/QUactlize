#include <hggc_runtime_api.h>
#include <cstdint>
#define POINT_RT(name) hggc##name
#include "events.hpp"
#undef POINT_RT

// Prebound CPU descriptors, never a Python callback in a timed interval.
struct PointCall {
    void * function;
    void * p[11];
    int32_t kind, v[6];
    int64_t stride;
};
static thread_local std::string error;
static int invoke(const PointCall & c) {
    switch (c.kind) {
        case 0: return reinterpret_cast<int(*)(void*,void*)>(c.function)(c.p[0],c.p[1]);
        case 1: return reinterpret_cast<int(*)(int,int,const void*,void*,int,int,int64_t,
            const int*,int,int*,void*)>(c.function)(c.v[0],c.v[1],c.p[0],c.p[1],c.v[2],c.v[3],
            c.stride,static_cast<const int*>(c.p[2]),c.v[4],static_cast<int*>(c.p[3]),c.p[4]);
        case 2: return reinterpret_cast<int(*)(void*,void*,const void*,const void*,void*,int,int,int)>(
            c.function)(c.p[0],c.p[1],c.p[2],c.p[3],c.p[4],c.v[0],c.v[1],c.v[2]);
        case 3: return int(hggcMemsetAsync(c.p[0],0,4,static_cast<hggcStream_t>(c.p[1])));
        default: return 1;
    }
}
extern "C" const char * point_error() { return error.c_str(); }
extern "C" int point_call_size() { return sizeof(PointCall); }
extern "C" int point_run(const PointCall * calls,int stages,int copy) {
    if (!calls || stages<1 || copy<0) {error="invalid point call";return 1;}
    for (int j=0;j<stages;++j) {
        int rc=invoke(calls[copy*stages+j]);
        if(rc) {error="point stage="+std::to_string(j)+" rc="+std::to_string(rc);return rc;}
    }
    error.clear();return 0;
}
extern "C" int point_samples(const PointCall * calls,int stages,int copies,void * stream,
        int mode,int stage,int samples,double * values) {
    try {
        prefill_point::sample(static_cast<hggcStream_t>(stream),stages,copies,mode,stage,samples,
            [&](int j,int i){return invoke(calls[i*stages+j]);},values);
        error.clear();return 0;
    } catch (const std::exception & e) {error=e.what();return 1;}
}
