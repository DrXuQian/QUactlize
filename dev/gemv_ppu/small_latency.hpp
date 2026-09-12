// Development experiment. Included inside the frozen K-pack namespace.
// No output-format change, FP16 accumulator or inter-CTA synchronization.
__device__ __forceinline__ uint32_t latency_mux(uint32_t mask,uint32_t hi,uint32_t lo) {
    uint32_t value;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xca;" : "=r"(value) : "r"(mask),"r"(hi),"r"(lo));
    return value;
}

template<int Header>
__device__ __forceinline__ uint2 latency_fields(uint4 u,unsigned group) {
    uint32_t scales,mins;
    if constexpr(Header==0) {
        scales=(group&4) ? (u.z>>16)|(u.w<<16) : u.y;
        mins=(group&4) ? u.w>>8 : (u.y>>24)|(u.z<<8);
    } else {
        uint32_t mask=0u-((group>>2)&1);
        scales=latency_mux(mask,(u.z>>16)|(u.w<<16),u.y);
        mins=latency_mux(mask,u.w>>8,(u.y>>24)|(u.z<<8));
    }
    unsigned shift=6*(group&3);
    return make_uint2((scales>>shift)&63,(mins>>shift)&63);
}

template<int Header>
__device__ __forceinline__ quactlize::dev::q4_native::ScaleZero latency_half_header(uint4 u,unsigned group) {
    uint2 fields=latency_fields<Header>(u,group);
    __half2_raw codes_raw,header_raw;
    codes_raw.x=uint16_t(0x6400|fields.x);codes_raw.y=uint16_t(0x6400|fields.y);
    header_raw.x=uint16_t(u.x);header_raw.y=uint16_t(u.x>>16);
    __half2 const products=__hmul2(__hsub2(__half2(codes_raw),__float2half2_rn(1024.f)),__half2(header_raw));
    __half const scale=__low2half(products),zero=__hneg(__high2half(products));
    return {scale,__float2half_rn(__half2float(zero)+8.f*__half2float(scale))};
}

template<int Header>
__device__ __forceinline__ float2 latency_affine_header(uint4 u,unsigned group) {
    uint2 fields=latency_fields<Header>(u,group);
    return make_float2(__half2float(__ushort_as_half(uint16_t(u.x)))*float(fields.x),
                     -__half2float(__ushort_as_half(uint16_t(u.x>>16)))*float(fields.y));
}

// These compiler-only joins put all named values on the input side of the
// decoding boundary. They emit no device barrier. The build records actual
// native load/wait order; source ordering alone is not an admission claim.
__device__ __forceinline__ void latency_join(uint4& u,uint4& b,uint4& a) {
    asm volatile("" : "+r"(u.x),"+r"(u.y),"+r"(u.z),"+r"(u.w),
                      "+r"(b.x),"+r"(b.y),"+r"(b.z),"+r"(b.w),
                      "+r"(a.x),"+r"(a.y),"+r"(a.z),"+r"(a.w) : : "memory");
}

template<int AMode>
__device__ __forceinline__ uint4 latency_residue_a(__half const* a,unsigned g,unsigned residue) {
    if constexpr(AMode==0) {
        uint2 v=*reinterpret_cast<uint2 const*>(a+g*32+residue*4);
        return make_uint4(v.x,v.y,0,0);
    } else {
        return make_uint4(__half_as_ushort(a[g*32+residue]),__half_as_ushort(a[g*32+residue+8]),
                          __half_as_ushort(a[g*32+residue+16]),__half_as_ushort(a[g*32+residue+24]));
    }
}

__device__ __forceinline__ uint32_t latency_swap_half(uint32_t value,unsigned lane,unsigned bit) {
    uint32_t other=__shfl_xor_sync(0xffffffffu,value,bit);
    return __byte_perm(value,other,(lane&bit) ? 0x3276 : 0x5410);
}

template<int AMode>
__device__ __forceinline__ float4 latency_residue_values(uint4 raw,unsigned residue) {
    if constexpr(AMode==0) {
        uint2 v=make_uint2(raw.x,raw.y);
        v.x=latency_swap_half(v.x,residue,1);
        v.y=latency_swap_half(v.y,residue,1);
        uint32_t exchange=__shfl_xor_sync(0xffffffffu,(residue&2) ? v.x : v.y,2);
        if(residue&2) v.x=exchange; else v.y=exchange;
        v.x=latency_swap_half(v.x,residue,4);
        v.y=latency_swap_half(v.y,residue,4);
        raw=make_uint4(v.x,v.y,v.x>>16,v.y>>16);
    }
    return make_float4(__half2float(__ushort_as_half(uint16_t(raw.x))),
                       __half2float(__ushort_as_half(uint16_t(raw.y))),
                       __half2float(__ushort_as_half(uint16_t(raw.z))),
                       __half2float(__ushort_as_half(uint16_t(raw.w))));
}

template<int Header,int Loading,int AMode,int Warps,int N,int K>
__global__ void q4_small_pipeline(void const* a_ptr,uint8_t const* low_ptr,uint8_t const* units_ptr,float* output) {
    using namespace quactlize::dev::q4_native;
    constexpr unsigned Width=8;
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32,residue=lane%8;
    unsigned first=blockIdx.x*Width;
    auto low=reinterpret_cast<uint16_t const*>(low_ptr);
    auto a=static_cast<__half const*>(a_ptr);
    extern __shared__ __align__(16) unsigned char staged[];
    if constexpr(AMode==2) {
        for(unsigned i=tid;i<K/8;i+=Warps*32)
            reinterpret_cast<uint4*>(staged)[i]=reinterpret_cast<uint4 const*>(a)[i];
        __syncthreads();
        a=reinterpret_cast<__half const*>(staged);
    }
    float2 sums[4]{};
    #pragma unroll
    for(unsigned pass=0;pass<(K/128+Warps-1)/Warps;++pass) {
        unsigned chunk=pass*Warps+warp;
        if(chunk>=K/128) continue;
        unsigned g=chunk*4+lane/8;
        uint4 unit=aligned_unit(units_ptr+(size_t(g/8)*N+first+residue)*16);
        uint4 b,ab;
        ScaleZero sz;
        if constexpr(Loading==0) sz=latency_half_header<Header>(unit,g&7);
        b=*reinterpret_cast<uint4 const*>(low+size_t(g*8+residue)*N+first);
        ab=latency_residue_a<AMode>(a,g,residue);
        if constexpr(Loading==1) {
            latency_join(unit,b,ab);
            sz=latency_half_header<Header>(unit,g&7);
        }
        uint32_t meta=uint32_t(__half_as_ushort(sz.scale))|(uint32_t(__half_as_ushort(sz.zero))<<16);
        float4 av=latency_residue_values<AMode>(ab,residue);
        float act[4]={av.x,av.y,av.z,av.w};
        uint32_t words[4]={b.x,b.y,b.z,b.w};
        #pragma unroll
        for(int p=0;p<4;++p) {
            uint32_t s0=__shfl_sync(0xffffffffu,meta,(lane&~7)+2*p);
            uint32_t s1=__shfl_sync(0xffffffffu,meta,(lane&~7)+2*p+1);
            __half2 scale=__halves2half2(__ushort_as_half(uint16_t(s0)),__ushort_as_half(uint16_t(s1)));
            __half2 zero=__halves2half2(__ushort_as_half(uint16_t(s0>>16)),__ushort_as_half(uint16_t(s1>>16)));
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                __half2 q;
                if(slot==0) q=codes<0>(words[p]);
                else if(slot==1) q=codes<1>(words[p]);
                else if(slot==2) q=codes<2>(words[p]);
                else q=codes<3>(words[p]);
                float2 w=__half22float2(__hfma2(q,scale,zero));
                sums[p].x=fmaf(act[slot],w.x,sums[p].x);
                sums[p].y=fmaf(act[slot],w.y,sums[p].y);
            }
        }
    }
    float values[Width];
    #pragma unroll
    for(int p=0;p<4;++p) {values[2*p]=sums[p].x;values[2*p+1]=sums[p].y;}
    float value=q4_reduce_scatter_steps<Width,1>(values,lane);
    __shared__ float partial[Warps*Width];
    if(lane<Width) partial[warp*Width+lane]=value;
    __syncthreads();
    if(tid<32) {
        float sum=0;
        #pragma unroll
        for(unsigned w=tid/Width;w<Warps;w+=32/Width) sum+=partial[w*Width+tid%Width];
        #pragma unroll
        for(int d=Width;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
        if(tid<Width) output[first+tid]=sum;
    }
}
