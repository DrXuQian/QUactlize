// Canonical words -> cp.async -> bank-conflict-free ldmatrix transpose.
// One metadata decode per lane, four scale/zero warp exchanges, FP32 dot.
template<int TileN,int TileK,int Warps,bool Pipeline>
__global__ void q4_ldmatrix_v2(qkg_call_v1 c) {
    using namespace quactlize::dev::q4_native;
    constexpr int WarpK=TileN==8 ? 128 : 64;
    constexpr int StageWords=TileK/4*TileN;
    constexpr int Buffers=Pipeline ? 2 : 1;
    __shared__ __align__(16) uint16_t packed[Buffers*StageWords];
    __shared__ float partial[Warps*TileN];
    extern __shared__ __align__(16) unsigned char ashared[];
    auto a=reinterpret_cast<__half*>(ashared);
    auto low=reinterpret_cast<uint16_t const*>(c.low);
    int const tid=threadIdx.x,warp=tid/32,lane=tid%32;
    int const nlocal=lane/4,rpair=2*(lane%4),first=blockIdx.x*TileN;
    for(int i=tid;i<c.k/8;i+=Warps*32)
        reinterpret_cast<uint4*>(a)[i]=reinterpret_cast<uint4 const*>(c.a)[i];
    auto address=[](int row,int col) {
        return row*TileN+(TileN==16 ? col^((row&4)*2) : col);
    };
    auto issue=[&](int chunk,int buf) {
        #pragma unroll
        for(int i=tid;i<StageWords/8;i+=Warps*32) {
            int const row=i/(TileN/8),col=(i%(TileN/8))*8;
            unsigned dst=unsigned(__cvta_generic_to_shared(packed+buf*StageWords+address(row,col)));
            auto src=low+size_t(chunk*(TileK/4)+row)*c.n+first+col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst),"l"(src):"memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    int const chunks=c.k/TileK;
    float2 accum[4]{};
    if constexpr(Pipeline) issue(0,0);
    for(int chunk=0;chunk<chunks;++chunk) {
        int const buf=Pipeline ? chunk%2 : 0;
        if constexpr(!Pipeline) issue(chunk,0);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        // This also finishes readers of the buffer reused by the next issue.
        __syncthreads();
        if constexpr(Pipeline) if(chunk+1<chunks) issue(chunk+1,1-buf);
        #pragma unroll
        for(int micro=warp;micro<TileK/WarpK;micro+=Warps) {
            int const mat=lane/8;
            int const row=micro*(WarpK/4)+(TileN==8 ? mat : mat/2)*8+lane%8;
            int const col=TileN==8 ? 0 : (mat%2)*8;
            unsigned addr=unsigned(__cvta_generic_to_shared(packed+buf*StageWords+address(row,col)));
            uint32_t q[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                :"=r"(q[0]),"=r"(q[1]),"=r"(q[2]),"=r"(q[3]):"r"(addr));
            int const gbase=chunk*(TileK/32)+micro*(WarpK/32);
            int const owner=lane%4;
            int const own_g=gbase+(TileN==8 ? owner : owner/2);
            int const own_n=first+nlocal+(TileN==8 ? 0 : (owner%2)*8);
            auto sz=aligned_scale_zero(aligned_unit(c.units+(size_t(own_g/8)*c.n+own_n)*16),own_g&7);
            uint32_t sz_bits=uint32_t(__half_as_ushort(sz.scale))|(uint32_t(__half_as_ushort(sz.zero))<<16);
            #pragma unroll
            for(int v=0;v<4;++v) {
                uint32_t bits=__shfl_sync(0xffffffffu,sz_bits,(lane&~3)+v);
                __half2 scale=__half2half2(__ushort_as_half(uint16_t(bits)));
                __half2 zero=__half2half2(__ushort_as_half(uint16_t(bits>>16)));
                int const g=gbase+(TileN==8 ? v : v/2);
                #pragma unroll
                for(int slot=0;slot<4;++slot) {
                    __half2 code;
                    if(slot==0) code=codes<0>(q[v]);
                    else if(slot==1) code=codes<1>(q[v]);
                    else if(slot==2) code=codes<2>(q[v]);
                    else code=codes<3>(q[v]);
                    float2 w=__half22float2(__hfma2(code,scale,zero));
                    float2 av=__half22float2(*reinterpret_cast<__half2 const*>(a+g*32+rpair+slot*8));
                    accum[v].x=fmaf(w.x,av.x,accum[v].x);
                    accum[v].y=fmaf(w.y,av.y,accum[v].y);
                }
            }
        }
        if constexpr(!Pipeline) if(chunk+1<chunks) __syncthreads();
    }
    float sums[TileN/8]{};
    #pragma unroll
    for(int v=0;v<4;++v) sums[TileN==8 ? 0 : v%2]+=accum[v].x+accum[v].y;
    #pragma unroll
    for(int n=0;n<TileN/8;++n) {
        float v=sums[n];
        v+=__shfl_xor_sync(0xffffffffu,v,1);
        v+=__shfl_xor_sync(0xffffffffu,v,2);
        if(lane%4==0) partial[warp*TileN+n*8+nlocal]=v;
    }
    __syncthreads();
    if(tid<32) {
        float v=0;
        #pragma unroll
        for(int w=tid/TileN;w<Warps;w+=32/TileN) v+=partial[w*TileN+tid%TileN];
        #pragma unroll
        for(int d=TileN;d<32;d*=2) v+=__shfl_xor_sync(0xffffffffu,v,d);
        if(tid<TileN) c.output[first+tid]=v;
    }
}
