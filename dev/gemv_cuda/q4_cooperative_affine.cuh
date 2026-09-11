// Eight residue lanes cooperate on one 32-K group. Dot and affine are FP32;
// scale is applied after the residue reduction, not once per weight.
template<int Count,int Step,int Group=8>
__device__ __forceinline__ float q4_group_scatter(float (&value)[Count],int lane) {
    if constexpr(Count==1) {
        float sum=value[0];
        #pragma unroll
        for(int d=Step;d<Group;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        return sum;
    } else {
        float next[Count/2];
        #pragma unroll
        for(int i=0;i<Count/2;++i) {
            bool odd=(lane&Step)!=0;
            float keep=odd ? value[2*i+1] : value[2*i];
            float send=odd ? value[2*i] : value[2*i+1];
            next[i]=keep+__shfl_xor_sync(0xffffffffu,send,Step);
        }
        return q4_group_scatter<Count/2,Step*2,Group>(next,lane);
    }
}

template<int Width,int Warps,int N,int K>
__global__ void q4_cooperative_affine(qkg_call_v1 c) {
    using namespace quactlize::dev::q4_native;
    static_assert(Width==4 || Width==8);
    constexpr int Pairs=Width/2;
    int const tid=threadIdx.x,lane=tid%32,warp=tid/32,residue=lane%8;
    int const first=blockIdx.x*Width;
    auto low=reinterpret_cast<uint16_t const*>(c.low);
    auto a=static_cast<__half const*>(c.a);
    float total=0;
    #pragma unroll
    for(int pass=0;pass<(K/128+Warps-1)/Warps;++pass) {
        int chunk=pass*Warps+warp;
        if(chunk>=K/128) continue;
        int g=chunk*4+lane/8;
        uint4 unit=aligned_unit(c.units+(size_t(g/8)*N+first+residue%Width)*16);
        uint32_t words[Pairs];
        if constexpr(Width==4) {
            uint2 b=*reinterpret_cast<uint2 const*>(low+size_t(g*8+residue)*N+first);
            words[0]=b.x;words[1]=b.y;
        } else {
            uint4 b=*reinterpret_cast<uint4 const*>(low+size_t(g*8+residue)*N+first);
            words[0]=b.x;words[1]=b.y;words[2]=b.z;words[3]=b.w;
        }
        float av[4],a_sum=0;
        #if defined(Q4_COOPERATIVE_VECTOR_A)
        float4 act=q4_warp_activation(c.a,g,residue);
        av[0]=act.x;av[1]=act.y;av[2]=act.z;av[3]=act.w;
        a_sum=(act.x+act.y)+(act.z+act.w);
        #else
        #pragma unroll
        for(int slot=0;slot<4;++slot) {
            av[slot]=__half2float(a[g*32+residue+slot*8]);
            a_sum+=av[slot];
        }
        #endif
        float dot[Width]{};
        #pragma unroll
        for(int p=0;p<Pairs;++p) {
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                __half2 q;
                if(slot==0) q=codes<0>(words[p]);
                else if(slot==1) q=codes<1>(words[p]);
                else if(slot==2) q=codes<2>(words[p]);
                else q=codes<3>(words[p]);
                float2 code=__half22float2(q);
                dot[2*p]=fmaf(av[slot],code.x,dot[2*p]);
                dot[2*p+1]=fmaf(av[slot],code.y,dot[2*p+1]);
            }
        }
        float sum=q4_group_scatter<Width,1>(dot,lane);
        #pragma unroll
        for(int d=1;d<8;d*=2) a_sum+=__shfl_xor_sync(0xffffffffu,a_sum,d);
        float2 sz=q4_affine_header(unit,g&7);
        total+=fmaf(sz.x,sum,sz.y*a_sum);
    }
    total+=__shfl_xor_sync(0xffffffffu,total,8);
    total+=__shfl_xor_sync(0xffffffffu,total,16);
    __shared__ float warp_value[Warps*Width];
    if(lane<Width) warp_value[warp*Width+lane]=total;
    __syncthreads();
    if(tid<32) {
        float sum=0;
        #pragma unroll
        for(int w=tid/Width;w<Warps;w+=32/Width) sum+=warp_value[w*Width+tid%Width];
        #pragma unroll
        for(int d=Width;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if(tid<Width) c.output[first+tid]=sum;
    }
}
