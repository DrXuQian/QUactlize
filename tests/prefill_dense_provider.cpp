#include <cassert>
#include <cstddef>
struct State {void* stream;void* workspace;int math;};
extern "C" int cublasCreate_v2(void** p){*p=new State{};return 0;}
extern "C" int cublasDestroy_v2(void* p){delete static_cast<State*>(p);return 0;}
extern "C" int cublasSetStream_v2(void* p,void* s){auto h=static_cast<State*>(p);h->stream=s;h->workspace=nullptr;return 0;}
extern "C" int cublasSetWorkspace_v2(void* p,void* w,size_t n){assert(n==4u<<20);static_cast<State*>(p)->workspace=w;return 0;}
extern "C" int cublasSetMathMode(void* p,int m){static_cast<State*>(p)->math=m;return 0;}
extern "C" int cublasGemmEx(void* p,int ta,int tb,int n,int m,int k,void const* alpha,
    void const* b,int bt,int ldb,void const* a,int at,int lda,void const* beta,void* out,int ct,int ldc,int compute,int algorithm){
    auto h=static_cast<State*>(p);assert(h->stream && h->workspace && h->math==16);
    assert(ta==1 && tb==0 && n==256 && m==128 && k==512 && ldb==512 && lda==512 && ldc==256);
    int expected=*static_cast<float const*>(a)>=100?14:2;
    assert(bt==expected && at==expected && ct==expected && compute==68 && (algorithm==-1 || algorithm==99));
    assert(*static_cast<float const*>(alpha)==1 && *static_cast<float const*>(beta)==0);
    *static_cast<float*>(out)=*static_cast<float const*>(a)+*static_cast<float const*>(b);return 0;
}
