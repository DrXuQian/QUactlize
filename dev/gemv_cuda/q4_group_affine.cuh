// Distinct numerical contract: FP16 A, exact integer codes and FP32 grouped
// affine. No per-weight FP16 rounding or expanded workspace. This computes
// sum_g (d*s_g * dot(q_g,A_g) - dmin*m_g * sum(A_g)) in FP32.
// It is compared independently with original GGUF, not called FP16-weight GEMV.
__device__ __forceinline__ float2 q4_affine_header(uint4 u,unsigned group) {
    uint64_t run=(group&4) ? uint64_t(u.z>>16)|(uint64_t(u.w)<<16)
                          : uint64_t(u.y)|(uint64_t(u.z&0xffff)<<32);
    unsigned shift=6*(group&3);
    float sc=float(unsigned(run>>shift)&63),mn=float(unsigned(run>>(24+shift))&63);
    return make_float2(__half2float(__ushort_as_half(uint16_t(u.x)))*sc,
                      -__half2float(__ushort_as_half(uint16_t(u.x>>16)))*mn);
}

template<int Columns,int Warps,int P,int N,int K,bool Early>
__global__ void q4_group_affine(qkg_call_v1 c) {
    using namespace quactlize::dev::q4_native;
    constexpr int Workers=Warps*32/Columns,Pairs=P/2;
    int const tid=threadIdx.x,worker=tid/Columns,col=blockIdx.x*(Columns*P)+(tid%Columns)*P;
    auto low=reinterpret_cast<uint16_t const*>(c.low);
    float2 total[Pairs]{};
    #pragma unroll
    for(int pass=0;pass<(K/32+Workers-1)/Workers;++pass) {
        int const g=pass*Workers+worker;
        if(g>=K/32) continue;
        uint4 metadata[Early ? P : 1];
        if constexpr(Early) {
            #pragma unroll
            for(int p=0;p<P;++p) metadata[p]=aligned_unit(c.units+(size_t(g/8)*N+col+p)*16);
        }
        float2 dot[Pairs]{};
        float a_sum=0;
        #pragma unroll
        for(int half=0;half<2;++half) {
            uint32_t words[4][Pairs];
            #pragma unroll
            for(int r=0;r<4;++r) {
                auto ptr=low+size_t(g*8+half*4+r)*N+col;
                if constexpr(P==2) words[r][0]=*reinterpret_cast<uint32_t const*>(ptr);
                else if constexpr(P==4) {
                    uint2 v=*reinterpret_cast<uint2 const*>(ptr);words[r][0]=v.x;words[r][1]=v.y;
                } else {
                    uint4 v=*reinterpret_cast<uint4 const*>(ptr);
                    words[r][0]=v.x;words[r][1]=v.y;words[r][2]=v.z;words[r][3]=v.w;
                }
            }
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                float4 av=aligned_activation<0>(c.a,g*32+slot*8+half*4);
                float ax[4]={av.x,av.y,av.z,av.w};
                a_sum+=(av.x+av.y)+(av.z+av.w);
                #pragma unroll
                for(int r=0;r<4;++r) {
                    #pragma unroll
                    for(int p=0;p<Pairs;++p) {
                        __half2 q;
                        if(slot==0) q=codes<0>(words[r][p]);
                        else if(slot==1) q=codes<1>(words[r][p]);
                        else if(slot==2) q=codes<2>(words[r][p]);
                        else q=codes<3>(words[r][p]);
                        float2 v=__half22float2(q);
                        dot[p].x=fmaf(ax[r],v.x,dot[p].x);
                        dot[p].y=fmaf(ax[r],v.y,dot[p].y);
                    }
                }
            }
        }
        #pragma unroll
        for(int p=0;p<Pairs;++p) {
            uint4 u0,u1;
            if constexpr(Early) {u0=metadata[2*p];u1=metadata[2*p+1];}
            else {
                u0=aligned_unit(c.units+(size_t(g/8)*N+col+2*p)*16);
                u1=aligned_unit(c.units+(size_t(g/8)*N+col+2*p+1)*16);
            }
            float2 s0=q4_affine_header(u0,g&7),s1=q4_affine_header(u1,g&7);
            total[p].x+=fmaf(s0.x,dot[p].x,s0.y*a_sum);
            total[p].y+=fmaf(s1.x,dot[p].y,s1.y*a_sum);
        }
    }
    constexpr int TileN=Columns*P;
    static_assert(TileN<=32);
    float values[P];
    #pragma unroll
    for(int p=0;p<Pairs;++p) {values[2*p]=total[p].x;values[2*p+1]=total[p].y;}
    int const lane=tid%32;
    float value=q4_reduce_scatter_steps<P,Columns>(values,lane);
    __shared__ float partial[Warps*TileN];
    if(lane<TileN) partial[(tid/32)*TileN+(lane%Columns)*P+lane/Columns]=value;
    __syncthreads();
    if(tid<32) {
        float sum=0;
        #pragma unroll
        for(int w=tid/TileN;w<Warps;w+=32/TileN) sum+=partial[w*TileN+tid%TileN];
        #pragma unroll
        for(int d=TileN;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if(tid<TileN) c.output[blockIdx.x*TileN+tid]=sum;
    }
}
