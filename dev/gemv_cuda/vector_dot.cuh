// CUDA-only pair experiment. Keep canonical metadata and the pair reader's
// FP16 affine rounding, but load contiguous A vectors and expose four dot
// chains. The FP32 reduction order differs; independent GGUF admission is
// required. Unaligned F32 and F16 inputs retain scalar, in-bounds loads.
template<class Reader>
__device__ __forceinline__ float pair_dot(qkg_call_v1 const& c, int64_t a_base,
    uint16_t const* low, uint16_t const* high, uint8_t const* units,
    int col, int worker, int workers, int partition, int split) {
    using R=Reader;
    float accum[4]{};
    for (int g=partition*workers+worker; g<c.k/R::group; g+=split*workers) {
        auto const s=R::scale(units,col,g,c.n);
        __half const hs=__ushort_as_half(s.scale.raw());
        __half const hz=__ushort_as_half(s.zero.raw());
        __half2 const scale=__halves2half2(hs,hs), zero=__halves2half2(hz,hz);
        constexpr float bias=R::lo_bits==8 ? 1152.f : R::lo_bits==4 ? 1032.f : 1024.f;
        __half2 const offset=__float2half2_rn(bias);
        uint16_t words[8], high_words[8]{};
        #pragma unroll
        for (int r=0;r<8;++r) {
            words[r]=low[R::LowMap::word_index(col,g*R::group+r,c.n)];
            if constexpr (R::hi_bits!=0)
                high_words[r]=high[R::HighMap::word_index(col,g*R::group+r,c.n)];
        }
        #pragma unroll
        for (int slot=0;slot<R::group/8;slot+=2) {
            #pragma unroll
            for (int first=0;first<8;first+=4) {
                int const kk=g*R::group+8*slot+first;
                float a0[4],a1[4];
                auto const p=static_cast<float const*>(c.a)+a_base+kk;
                if (c.input_type==QKG_F32 && (uintptr_t(p)&15)==0) {
                    float4 const x=*reinterpret_cast<float4 const*>(p);
                    float4 const y=*reinterpret_cast<float4 const*>(p+8);
                    a0[0]=x.x; a0[1]=x.y; a0[2]=x.z; a0[3]=x.w;
                    a1[0]=y.x; a1[1]=y.y; a1[2]=y.z; a1[3]=y.w;
                } else {
                    #pragma unroll
                    for (int i=0;i<4;++i) {
                        a0[i]=c.input_type==QKG_F32 ? p[i]
                            : float(static_cast<Half const*>(c.a)[a_base+kk+i]);
                        a1[i]=c.input_type==QKG_F32 ? p[8+i]
                            : float(static_cast<Half const*>(c.a)[a_base+kk+8+i]);
                    }
                }
                #pragma unroll
                for (int i=0;i<4;++i) {
                    uint16_t lo=words[first+i];
                    if constexpr (R::lo_bits==8)
                        lo=low[R::LowMap::word_index(col,kk+i,c.n)];
                    int const c0=R::raw_from_words(lo,high_words[first+i],col,kk+i);
                    int const c1=R::raw_from_words(lo,high_words[first+i],col,kk+8+i);
                    __half2 codes=__halves2half2(__ushort_as_half(0x6400|c0),
                                                __ushort_as_half(0x6400|c1));
                    codes=__hsub2(codes,offset);
                    // Q8 has no affine zero; native multiply preserves its
                    // original single FP16 rounding, including signed zero.
                    __half2 weight;
                    if constexpr (R::lo_bits==8) weight=__hmul2(codes,scale);
                    else weight=__hfma2(codes,scale,zero);
                    float2 const w=__half22float2(weight);
                    float2 const a=__half22float2(__floats2half2_rn(a0[i],a1[i]));
                    accum[i]=fmaf(a.x,w.x,accum[i]);
                    accum[i]=fmaf(a.y,w.y,accum[i]);
                }
            }
        }
    }
    return (accum[0]+accum[1])+(accum[2]+accum[3]);
}
