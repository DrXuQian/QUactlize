// Access-pattern probe (not a GEMV): reads one K-pack plane-shaped buffer [rows][n] of uint16
// with three load patterns and no arithmetic consumer, to separate DRAM access-pattern cost
// from reader cost.
//   frag : the register-direct MMA reader pattern. A warp owns 8*T columns, lane (m=lane/4,
//          t=lane%4) loads T columns at rows 8g+2t and 8g+2t+1 of granule g; WK warps split
//          the granules; U granules are issued before any is consumed.
//   rows : row-contiguous tile. Each warp instruction reads 512 B of ONE row (32 lanes x 16 B);
//          a CTA covers 256 columns and WK warps split the rows; U rows in flight per warp.
//   flat : the bw_probe pattern (each CTA a contiguous chunk, 16 B per lane).
// usage: pattern_probe < configs   (one "PATTERN n rows T WN WK U" per line; flat uses WN*WK CTAs of T*32
// threads). Every config is preceded by a memset of 1 MiB + 64*index bytes so an Asys trace can be
// attributed (tools/pattern_asys.py); the printed event time includes launch gaps.
#include <hggc_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

template<int T> struct W;
template<> struct W<2> { uint32_t x; __device__ void ld(uint16_t const* p) { x=*reinterpret_cast<uint32_t const*>(p); }
                         __device__ uint32_t h() const { return x; } };
template<> struct W<4> { uint2 x; __device__ void ld(uint16_t const* p) { x=*reinterpret_cast<uint2 const*>(p); }
                         __device__ uint32_t h() const { return x.x^x.y; } };
template<> struct W<8> { uint4 x; __device__ void ld(uint16_t const* p) { x=*reinterpret_cast<uint4 const*>(p); }
                         __device__ uint32_t h() const { return x.x^x.y^x.z^x.w; } };

template<int T,int WN,int WK,int U>
__global__ void __launch_bounds__(WN*WK*32) frag(uint16_t const* __restrict__ p,int n,int rows,uint32_t* out) {
    int const lane=threadIdx.x%32,warp=threadIdx.x/32,wn=warp%WN,wk=warp/WN,m=lane/4,t=lane%4;
    int const col=blockIdx.x*(WN*8*T)+wn*8*T+T*m;
    int const gran=rows/8,gb=gran*wk/WK,ge=gran*(wk+1)/WK;
    uint32_t acc=0;
    for(int g=gb;g<ge;g+=U) {
        W<T> r[U][2];
        #pragma unroll
        for(int u=0;u<U;++u)
            if(g+u<ge) {
                r[u][0].ld(p+size_t((g+u)*8+2*t)*n+col);
                r[u][1].ld(p+size_t((g+u)*8+2*t+1)*n+col);
            } else { r[u][0].x={}; r[u][1].x={}; }
        #pragma unroll
        for(int u=0;u<U;++u) acc^=r[u][0].h()^r[u][1].h();
    }
    if(acc==0x12345678u) out[blockIdx.x]=acc;
}

template<int WK,int U>
__global__ void __launch_bounds__(WK*32) rowtile(uint16_t const* __restrict__ p,int n,int rows,uint32_t* out) {
    int const lane=threadIdx.x%32,wk=threadIdx.x/32;
    int const col=blockIdx.x*256+lane*8;
    int const rb=rows*wk/WK,re=rows*(wk+1)/WK;
    uint32_t acc=0;
    for(int r=rb;r<re;r+=U) {
        uint4 v[U];
        #pragma unroll
        for(int u=0;u<U;++u) v[u]=r+u<re ? *reinterpret_cast<uint4 const*>(p+size_t(r+u)*n+col) : make_uint4(0,0,0,0);
        #pragma unroll
        for(int u=0;u<U;++u) acc^=v[u].x^v[u].y^v[u].z^v[u].w;
    }
    if(acc==0x12345678u) out[blockIdx.x]=acc;
}

template<int U>
__global__ void flat(uint4 const* __restrict__ p,size_t words,uint32_t* out) {
    uint4 const* base=p+size_t(blockIdx.x)*words;
    uint32_t acc=0;
    for(size_t i=threadIdx.x;i<words;i+=blockDim.x*U) {
        uint4 v[U];
        #pragma unroll
        for(int u=0;u<U;++u) { size_t j=i+size_t(u)*blockDim.x; v[u]=j<words ? base[j] : make_uint4(0,0,0,0); }
        #pragma unroll
        for(int u=0;u<U;++u) acc^=v[u].x^v[u].y^v[u].z^v[u].w;
    }
    if(acc==0x12345678u) out[blockIdx.x]=acc;
}

// seg : each warp instruction reads R rows x (512/R) contiguous bytes (lanes 32/R per row, 16 B
//       each); a CTA spans 256/R columns and WK warps split the rows; U instructions in flight.
template<int R,int WK,int U>
__global__ void __launch_bounds__(WK*32) seg(uint16_t const* __restrict__ p,int n,int rows,uint32_t* out) {
    constexpr int L=32/R;
    int const lane=threadIdx.x%32,wk=threadIdx.x/32,rl=lane/L,cl=lane%L;
    int const col=blockIdx.x*(L*8)+cl*8;
    // Rows split over gridDim.y CTA slices x WK warps (k split across CTAs for small n).
    int const parts=int(gridDim.y)*WK,part=int(blockIdx.y)*WK+wk;
    int const groups=rows/R,gb=int(int64_t(groups)*part/parts),ge=int(int64_t(groups)*(part+1)/parts);
    uint32_t acc=0;
    for(int g=gb;g<ge;g+=U) {
        uint4 v[U];
        #pragma unroll
        for(int u=0;u<U;++u)
            v[u]=g+u<ge ? *reinterpret_cast<uint4 const*>(p+size_t((g+u)*R+rl)*n+col) : make_uint4(0,0,0,0);
        #pragma unroll
        for(int u=0;u<U;++u) acc^=v[u].x^v[u].y^v[u].z^v[u].w;
    }
    if(acc==0x12345678u) out[blockIdx.x]=acc;
}

// cpa : the seg pattern through ppu.cp.async into a per-warp shared ring of U slots (16 B per lane
//       per slot), consumed with ld.shared after wait_group; the consumer sees every byte.
template<int R,int WK,int U>
__global__ void __launch_bounds__(WK*32) cpa(uint16_t const* __restrict__ p,int n,int rows,uint32_t* out) {
    constexpr int L=32/R;
    __shared__ uint4 ring[WK][U][32];
    int const lane=threadIdx.x%32,wk=threadIdx.x/32,rl=lane/L,cl=lane%L;
    int const col=blockIdx.x*(L*8)+cl*8;
    // Rows split over gridDim.y CTA slices x WK warps (k split across CTAs for small n).
    int const parts=int(gridDim.y)*WK,part=int(blockIdx.y)*WK+wk;
    int const groups=rows/R,gb=int(int64_t(groups)*part/parts),ge=int(int64_t(groups)*(part+1)/parts);
    auto issue=[&](int g,int slot) {
        uint32_t const s=uint32_t(__cvta_generic_to_shared(&ring[wk][slot][lane]));
        void const* src=p+size_t(g*R+rl)*n+col;
        int const bytes=g<ge ? 16 : 0;
        asm volatile("ppu.cp.async.cg.shared.global.LLC::128B [%0], [%1], 16, %2;\n" :: "r"(s),"l"(src),"r"(bytes));
        asm volatile("ppu.cp.async.commit_group;\n" ::);
    };
    uint32_t acc=0;
    #pragma unroll
    for(int u=0;u<U-1;++u) issue(gb+u,u);
    for(int g=gb;g<ge;++g) {
        issue(g+U-1,(g-gb+U-1)%U);
        asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(U-1));
        __syncwarp();
        uint4 const v=ring[wk][(g-gb)%U][lane];
        acc^=v.x^v.y^v.z^v.w;
        __syncwarp();
    }
    if(acc==0x12345678u) out[blockIdx.x]=acc;
}

using Launch=void(*)(uint16_t const*,int,int,uint32_t*,hggcStream_t);
static int g_split=1;   // seg/cpa: CTA slices along k (the WN field)

template<int R,int WK,int U>
void run_seg(uint16_t const* p,int n,int rows,uint32_t* o,hggcStream_t s) {
    seg<R,WK,U><<<dim3(n/(256/R),g_split),WK*32,0,s>>>(p,n,rows,o);
}
template<int R,int WK,int U>
void run_cpa(uint16_t const* p,int n,int rows,uint32_t* o,hggcStream_t s) {
    cpa<R,WK,U><<<dim3(n/(256/R),g_split),WK*32,0,s>>>(p,n,rows,o);
}
template<bool Cpa,int R,int WK>
Launch pick_seg_u(int u) {
    if constexpr(Cpa) {
        switch(u) { case 2: return run_cpa<R,WK,2>; case 4: return run_cpa<R,WK,4>; case 8: return run_cpa<R,WK,8>; }
    } else {
        switch(u) { case 1: return run_seg<R,WK,1>; case 2: return run_seg<R,WK,2>; case 4: return run_seg<R,WK,4>;
                    case 8: return run_seg<R,WK,8>; }
    }
    return nullptr;
}
template<bool Cpa,int R>
Launch pick_seg_wk(int wk,int u) {
    if(wk==4) return pick_seg_u<Cpa,R,4>(u);
    if(wk==8) return pick_seg_u<Cpa,R,8>(u);
    if(wk==16) return pick_seg_u<Cpa,R,16>(u);
    return nullptr;
}
template<bool Cpa>
Launch pick_seg(int r,int wk,int u) {
    if(r==1) return pick_seg_wk<Cpa,1>(wk,u);
    if(r==2) return pick_seg_wk<Cpa,2>(wk,u);
    if(r==4) return pick_seg_wk<Cpa,4>(wk,u);
    if(r==8) return pick_seg_wk<Cpa,8>(wk,u);
    return nullptr;
}

template<int T,int WN,int WK,int U>
void run_frag(uint16_t const* p,int n,int rows,uint32_t* o,hggcStream_t s) {
    frag<T,WN,WK,U><<<n/(WN*8*T),WN*WK*32,0,s>>>(p,n,rows,o);
}
template<int WK,int U>
void run_rows(uint16_t const* p,int n,int rows,uint32_t* o,hggcStream_t s) {
    rowtile<WK,U><<<n/256,WK*32,0,s>>>(p,n,rows,o);
}

template<int T,int WN,int WK>
Launch pick_frag_u(int u) {
    switch(u) {
        case 1: return run_frag<T,WN,WK,1>; case 2: return run_frag<T,WN,WK,2>;
        case 4: return run_frag<T,WN,WK,4>; case 8: return run_frag<T,WN,WK,8>;
    }
    return nullptr;
}
template<int T>
Launch pick_frag(int wn,int wk,int u) {
    if(wn==1 && wk==8) return pick_frag_u<T,1,8>(u);
    if(wn==2 && wk==4) return pick_frag_u<T,2,4>(u);
    if(wn==4 && wk==2) return pick_frag_u<T,4,2>(u);
    if(wn==4 && wk==1) return pick_frag_u<T,4,1>(u);
    if(wn==8 && wk==1) return pick_frag_u<T,8,1>(u);
    if(wn==2 && wk==8) return pick_frag_u<T,2,8>(u);
    return nullptr;
}
template<int WK>
Launch pick_rows_u(int u) {
    switch(u) {
        case 1: return run_rows<WK,1>; case 2: return run_rows<WK,2>;
        case 4: return run_rows<WK,4>; case 8: return run_rows<WK,8>;
    }
    return nullptr;
}
Launch pick_rows(int wk,int u) {
    if(wk==4) return pick_rows_u<4>(u);
    if(wk==8) return pick_rows_u<8>(u);
    if(wk==16) return pick_rows_u<16>(u);
    return nullptr;
}

int main() {
    int const reps=12;
    size_t const cap=size_t(1)<<30;
    void* buf=nullptr; uint32_t* out=nullptr; void* mark=nullptr;
    if(hggcMalloc(&buf,cap) || hggcMalloc(reinterpret_cast<void**>(&out),1<<22) || hggcMalloc(&mark,(1<<20)+64*4096)) {
        std::printf("alloc failed\n"); return 1;
    }
    hggcMemset(buf,1,cap);
    hggcStream_t s; hggcStreamCreate(&s);
    hggcEvent_t a,b; hggcEventCreate(&a); hggcEventCreate(&b);
    char pat[16];
    int n,rows,T,WN,WK,U,index=0;
    while(std::scanf("%15s %d %d %d %d %d %d",pat,&n,&rows,&T,&WN,&WK,&U)==7) {
        size_t const bytes=size_t(n)*rows*2,slot=(bytes+(1<<20)-1)>>20<<20;
        // Cold reads: the ring of distinct slots always exceeds the 64 MiB L2 by 2.5x.
        int const ring=int(std::max<size_t>(4,((size_t(160)<<20)+slot-1)/slot));
        Launch fn=nullptr;
        g_split=WN>0 ? WN : 1;
        if(!std::strcmp(pat,"frag")) fn=T==2 ? pick_frag<2>(WN,WK,U) : T==4 ? pick_frag<4>(WN,WK,U) : T==8 ? pick_frag<8>(WN,WK,U) : nullptr;
        else if(!std::strcmp(pat,"rows")) fn=pick_rows(WK,U);
        else if(!std::strcmp(pat,"seg")) fn=pick_seg<false>(T,WK,U);   // T carries R, WN the k slices
        else if(!std::strcmp(pat,"cpa")) fn=pick_seg<true>(T,WK,U);
        bool const is_flat=!std::strcmp(pat,"flat");
        if((!fn && !is_flat) || slot*ring>cap) { std::printf("skip %s\n",pat); ++index; continue; }
        hggcMemsetAsync(mark,0,(1<<20)+64*index,s);
        std::vector<float> ms;
        for(int rep=0;rep<reps+ring;++rep) {
            auto p=reinterpret_cast<uint16_t const*>(static_cast<char*>(buf)+size_t(rep%ring)*slot);
            hggcEventRecord(a,s);
            if(is_flat) {
                int const ctas=WN*WK;
                flat<4><<<ctas,T*32,0,s>>>(reinterpret_cast<uint4 const*>(p),bytes/16/ctas,out);
            } else fn(p,n,rows,out,s);
            hggcEventRecord(b,s);
            hggcEventSynchronize(b);
            float x; hggcEventElapsedTime(&x,a,b);
            if(rep>=ring) ms.push_back(x);
        }
        if(hggcGetLastError()!=hggcSuccess) { std::printf("launch error at %d\n",index); return 1; }
        std::sort(ms.begin(),ms.end());
        float const med=ms[ms.size()/2];
        std::printf("%d %s n=%d rows=%d T=%d WN=%d WK=%d U=%d bytes=%zu event_us=%.2f GBps=%.1f\n",index,pat,n,rows,T,WN,WK,
                    U,bytes,med*1e3,bytes/med/1e6);
        std::fflush(stdout);
        ++index;
    }
    hggcStreamSynchronize(s);
    return 0;
}
