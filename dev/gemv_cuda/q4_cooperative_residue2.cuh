// Four columns/CTA and two K residues/lane. Four lanes cover one complete
// 32-K quantization group, so metadata has no duplicated donor lanes.
template<int Warps,int N,int K,bool Affine>
__global__ void q4_cooperative_residue2(qkg_call_v1 c) {
    using namespace quactlize::dev::q4_native;
    int const tid=threadIdx.x,lane=tid%32,warp=tid/32;
    int const residue=2*(lane%4),first=blockIdx.x*4;
    auto low=reinterpret_cast<uint16_t const*>(c.low);
    float total=0;
    #pragma unroll
    for(int pass=0;pass<(K/256+Warps-1)/Warps;++pass) {
        int chunk=pass*Warps+warp;
        if(chunk>=K/256) continue;
        int g=chunk*8+lane/4;
        uint4 unit=aligned_unit(c.units+(size_t(g/8)*N+first+lane%4)*16);
        uint2 word[2];
        #pragma unroll
        for(int r=0;r<2;++r)
            word[r]=*reinterpret_cast<uint2 const*>(low+size_t(g*8+residue+r)*N+first);
        float2 av[4];
        float a_sum=0;
        #pragma unroll
        for(int slot=0;slot<4;++slot) {
            auto ptr=static_cast<__half const*>(c.a)+g*32+residue+slot*8;
            av[slot]=__half22float2(*reinterpret_cast<__half2 const*>(ptr));
            a_sum+=av[slot].x+av[slot].y;
        }
        uint32_t sz_bits=0;
        if constexpr(!Affine) {
            auto sz=aligned_scale_zero(unit,g&7);
            sz_bits=uint32_t(__half_as_ushort(sz.scale))|(uint32_t(__half_as_ushort(sz.zero))<<16);
        }
        float dot[4]{};
        #pragma unroll
        for(int p=0;p<2;++p) {
            __half2 scale,zero;
            if constexpr(!Affine) {
                uint32_t s0=__shfl_sync(0xffffffffu,sz_bits,(lane&~3)+2*p);
                uint32_t s1=__shfl_sync(0xffffffffu,sz_bits,(lane&~3)+2*p+1);
                scale=__halves2half2(__ushort_as_half(uint16_t(s0)),__ushort_as_half(uint16_t(s1)));
                zero=__halves2half2(__ushort_as_half(uint16_t(s0>>16)),__ushort_as_half(uint16_t(s1>>16)));
            }
            #pragma unroll
            for(int r=0;r<2;++r) {
                uint32_t bits=p ? word[r].y : word[r].x;
                #pragma unroll
                for(int slot=0;slot<4;++slot) {
                    __half2 q;
                    if(slot==0) q=codes<0>(bits);
                    else if(slot==1) q=codes<1>(bits);
                    else if(slot==2) q=codes<2>(bits);
                    else q=codes<3>(bits);
                    if constexpr(!Affine) q=__hfma2(q,scale,zero);
                    float2 weight=__half22float2(q);
                    float act=r ? av[slot].y : av[slot].x;
                    dot[2*p]=fmaf(act,weight.x,dot[2*p]);
                    dot[2*p+1]=fmaf(act,weight.y,dot[2*p+1]);
                }
            }
        }
        float sum=q4_group_scatter<4,1,4>(dot,lane);
        if constexpr(Affine) {
            a_sum+=__shfl_xor_sync(0xffffffffu,a_sum,1);
            a_sum+=__shfl_xor_sync(0xffffffffu,a_sum,2);
            float2 sz=q4_affine_header(unit,g&7);
            total+=fmaf(sz.x,sum,sz.y*a_sum);
        } else total+=sum;
    }
    #pragma unroll
    for(int d=4;d<32;d*=2) total+=__shfl_xor_sync(0xffffffffu,total,d);
    __shared__ float partial[Warps*4];
    if(lane<4) partial[warp*4+lane]=total;
    __syncthreads();
    if(tid<32) {
        float sum=0;
        #pragma unroll
        for(int w=tid/4;w<Warps;w+=8) sum+=partial[w*4+tid%4];
        #pragma unroll
        for(int d=4;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if(tid<4) c.output[first+tid]=sum;
    }
}
