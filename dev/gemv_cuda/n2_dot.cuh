// Pair two adjacent output columns in each thread. Their canonical b16 words
// are adjacent, so a b32 load and half2 affine serve both dots. A is shared
// between the columns without changing its FP16 conversion boundary.
__device__ __forceinline__ uint32_t n2_word(uint16_t const* ptr) {
    // The public ABI promises b16, not b32, alignment. Do not narrow it for
    // the vector fast path; suballocated planes may begin at base+2 bytes.
    if ((uintptr_t(ptr)&3)==0) return *reinterpret_cast<uint32_t const*>(ptr);
    return uint32_t(ptr[0])|(uint32_t(ptr[1])<<16);
}

template<class Reader>
__device__ __forceinline__ float2 pair_dot(qkg_call_v1 const& c, int64_t a_base,
    uint16_t const* low, uint16_t const* high, uint8_t const* units,
    int col, int worker, int workers, int partition, int split) {
    using R=Reader;
    float x[4]{},y[4]{};
    for (int g=partition*workers+worker; g<c.k/R::group; g+=split*workers) {
        auto const s0=R::scale(units,col,g,c.n),s1=R::scale(units,col+1,g,c.n);
        __half2 const scale=__halves2half2(__ushort_as_half(s0.scale.raw()),__ushort_as_half(s1.scale.raw()));
        __half2 const zero=__halves2half2(__ushort_as_half(s0.zero.raw()),__ushort_as_half(s1.zero.raw()));
        constexpr float bias=R::lo_bits==8 ? 1152.f : R::lo_bits==4 ? 1032.f : 1024.f;
        __half2 const offset=__float2half2_rn(bias);
        uint32_t words[8],high_words[8]{};
        #pragma unroll
        for (int r=0;r<8;++r) {
            int const kk=g*R::group+r;
            words[r]=n2_word(low+R::LowMap::word_index(col,kk,c.n));
            if constexpr (R::hi_bits!=0) {
                // Q5 exchanges N bit 3 with K bits, but keeps N bit 0.
                // Thus even/odd N are adjacent words, not a duplicated word.
                high_words[r]=n2_word(high+R::HighMap::word_index(col,kk,c.n));
            }
        }
        #pragma unroll
        for (int slot=0;slot<R::group/8;++slot) {
            #pragma unroll
            for (int first=0;first<8;first+=4) {
                int const kk=g*R::group+8*slot+first;
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
                    uint32_t lo=words[first+i],hi=high_words[first+i];
                    if constexpr (R::lo_bits==8)
                        lo=n2_word(low+R::LowMap::word_index(col,kk+i,c.n));
                    int const c0=R::raw_from_words(uint16_t(lo),uint16_t(hi),col,kk+i);
                    int const c1=R::raw_from_words(uint16_t(lo>>16),uint16_t(hi>>16),col+1,kk+i);
                    __half2 codes=__halves2half2(__ushort_as_half(0x6400|c0),__ushort_as_half(0x6400|c1));
                    codes=__hsub2(codes,offset);
                    __half2 weight;
                    if constexpr (R::lo_bits==8) weight=__hmul2(codes,scale);
                    else weight=__hfma2(codes,scale,zero);
                    float2 const w=__half22float2(weight);
                    float const a=__half2float(__float2half_rn(av[i]));
                    x[i]=fmaf(a,w.x,x[i]); y[i]=fmaf(a,w.y,y[i]);
                }
            }
        }
    }
    return make_float2((x[0]+x[1])+(x[2]+x[3]),(y[0]+y[1])+(y[2]+y[3]));
}
