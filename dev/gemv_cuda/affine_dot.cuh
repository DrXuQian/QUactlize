// Development experiment: accumulate code*A and sum(A) before the group's
// affine transform, like DMMV. Metadata products and the dot stay FP32.
// This removes intermediate FP16 rounding; it is checked against independent
// GGUF arithmetic, not claimed bit-identical to the mixed-input GEMM.
template<class Reader>
__device__ __forceinline__ float pair_dot(qkg_call_v1 const& c, int64_t a_base,
    uint16_t const* low, uint16_t const* high, uint8_t const* units,
    int col, int worker, int workers, int partition, int split) {
    using R=Reader;
    float accum=0.f;
    for (int g=partition*workers+worker; g<c.k/R::group; g+=split*workers) {
        using U=typename R::U;
        int const sb=g/U::kGroups;
        int64_t const offset=(int64_t(sb/U::kSbPerUnit)*c.n+col)*U::kUnitTotal
            +(sb%U::kSbPerUnit)*U::kSbBytes;
        auto const unit=units+offset;
        int sc=gguf_scale::packed_unit::code_of<type>(unit,g%U::kGroups,0);
        if constexpr (U::kSigned) {
            if (sc&(1<<(U::kScaleBits-1))) sc-=1<<U::kScaleBits;
        }
        float const d=float(Half::bitcast(uint16_t(unit[0])|(uint16_t(unit[1])<<8)));
        float const scale=d*float(sc-gguf_scale::Traits<type>::kScaleBias);
        float zero=0.f;
        if constexpr (U::kHasMin) {
            float const dm=float(Half::bitcast(uint16_t(unit[2])|(uint16_t(unit[3])<<8)));
            zero=-dm*float(gguf_scale::packed_unit::code_of<type>(unit,g%U::kGroups,1));
        }
        uint16_t lo[8], hi[8]{};
        #pragma unroll
        for (int r=0; r<8; ++r) {
            int const begin=g*R::group+r;
            lo[r]=low[R::LowMap::word_index(col,begin,c.n)];
            if constexpr (R::hi_bits!=0) hi[r]=high[R::HighMap::word_index(col,begin,c.n)];
        }
        float code_dot[4]{}, input_sum[4]{};
        #pragma unroll
        for (int slot=0; slot<R::group/8; ++slot) {
            #pragma unroll
            for (int first=0; first<8; first+=4) {
                int const kk=g*R::group+first+8*slot;
                float av[4];
                auto const p=static_cast<float const*>(c.a)+a_base+kk;
                if (c.input_type==QKG_F32 && (uintptr_t(p)&15)==0) {
                    float4 const a=*reinterpret_cast<float4 const*>(p);
                    av[0]=a.x; av[1]=a.y; av[2]=a.z; av[3]=a.w;
                } else {
                    #pragma unroll
                    for (int i=0;i<4;++i)
                        av[i]=c.input_type==QKG_F32 ? p[i]
                            : float(static_cast<Half const*>(c.a)[a_base+kk+i]);
                }
                #pragma unroll
                for (int i=0;i<4;++i) {
                    constexpr int bias=(R::lo_bits==4 ? 8 : 0)
                        -gguf_scale::packed_unit::kCanonicalPlacedZMul<type>;
                    int const code=R::raw_from_words(lo[first+i],hi[first+i],col,kk+i)-bias;
                    code_dot[i]=fmaf(float(code),av[i],code_dot[i]);
                    input_sum[i]+=av[i];
                }
            }
        }
        float const dot=(code_dot[0]+code_dot[1])+(code_dot[2]+code_dot[3]);
        float const sum=(input_sum[0]+input_sum[1])+(input_sum[2]+input_sum[3]);
        accum+=fmaf(dot,scale,sum*zero);
    }
    return accum;
}
