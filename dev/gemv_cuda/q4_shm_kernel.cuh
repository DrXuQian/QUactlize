// Development-only on-chip b16 transpose. Global bytes remain canonical
// K-pack: [K/4,N]. FP16 weights/A and FP32 dot/reduction are preserved.
template<int Word>
__device__ __forceinline__ void q4_shm_dot_word(uint32_t word, __half2 const (&a)[16],
    __half2 scale,__half2 zero,float2 (&partial)[4]) {
    using namespace quactlize::dev::q4_native;
    #pragma unroll
    for (int slot=0;slot<4;++slot) {
        // Keep slots compile-time constants for the mantissa insertion.
        __half2 q;
        if (slot==0) q=codes<0>(word);
        else if (slot==1) q=codes<1>(word);
        else if (slot==2) q=codes<2>(word);
        else q=codes<3>(word);
        float2 const w=__half22float2(__hfma2(q,scale,zero));
        float2 const x=__half22float2(a[Word+4*slot]);
        partial[slot].x=fmaf(w.x,x.x,partial[slot].x);
        partial[slot].y=fmaf(w.y,x.y,partial[slot].y);
    }
}

template<int Columns,int Warps>
__global__ void kpack_q4_shm(qkg_call_v1 c,int split) {
    using namespace quactlize::dev::q4_native;
    constexpr int TileN=Columns*2, TileK=1024, Pitch=TileK/4+8;
    constexpr int Outputs=TileN/Warps;
    static_assert(TileN%Warps==0);
    __shared__ __align__(16) uint16_t packed[TileN*Pitch];
    __shared__ __align__(16) __half activations[TileK];
    int const tile=blockIdx.x%(c.n/TileN), outer=blockIdx.x/(c.n/TileN);
    int const partition=outer%split,row=outer/split;
    int const expert=expert_for(c,row);
    int const tid=threadIdx.x,warp=tid/32,lane=tid%32;
    int const first_n=tile*TileN;
    if (expert<0 || expert>=c.experts) {
        if (tid<TileN) {
            auto dst=split==1 ? c.output+int64_t(row)*c.out_row_stride :
                static_cast<float*>(c.workspace)+(int64_t(row)*split+partition)*c.n;
            dst[first_n+tid]=__int_as_float(0x7fc00000);
        }
        return;
    }
    int64_t const a_base=c.mode==QKG_INDEXED
        ? int64_t(row/c.topk)*c.a_token_stride+(row%c.topk%c.channels)*c.a_row_stride
        : int64_t(row)*c.a_row_stride;
    auto low=reinterpret_cast<uint16_t const*>(c.low+int64_t(expert)*c.n*c.k/2);
    auto units=c.units+int64_t(expert)*c.n*(c.k/256)*16;
    float sums[Outputs]{};
    for (int chunk=partition;chunk*TileK<c.k;chunk+=split) {
        int const valid_k=min(TileK,c.k-chunk*TileK);
        for (int j=tid;j<TileK;j+=Warps*32) {
            __half v=__float2half_rn(0.f);
            if (j<valid_k) v=c.input_type==QKG_F32
                ? __float2half_rn(static_cast<float const*>(c.a)[a_base+chunk*TileK+j])
                : static_cast<__half const*>(c.a)[a_base+chunk*TileK+j];
            activations[j]=v;
        }
        // Each global vector reads eight consecutive N words. Shared memory
        // puts eight K8 words of one output column next to each other.
        for (int j=tid*8;j<TileN*(TileK/4);j+=Warps*32*8) {
            int const kw=j/TileN,nlocal=j%TileN;
            uint4 value=make_uint4(0,0,0,0);
            if (kw<valid_k/4) {
                auto p=low+int64_t(chunk*(TileK/4)+kw)*c.n+first_n+nlocal;
                if (!(uintptr_t(p)&15)) value=*reinterpret_cast<uint4 const*>(p);
                else {
                    uint32_t w[4];
                    #pragma unroll
                    for (int i=0;i<4;++i) w[i]=uint32_t(p[2*i])|(uint32_t(p[2*i+1])<<16);
                    value=make_uint4(w[0],w[1],w[2],w[3]);
                }
            }
            uint32_t const words[4]={value.x,value.y,value.z,value.w};
            #pragma unroll
            for (int i=0;i<4;++i) {
                packed[(nlocal+2*i)*Pitch+kw]=uint16_t(words[i]);
                packed[(nlocal+2*i+1)*Pitch+kw]=uint16_t(words[i]>>16);
            }
        }
        __syncthreads();
        if (lane<valid_k/32) {
            __half2 logical[16];
            #pragma unroll
            for (int j=0;j<16;++j)
                logical[j]=*reinterpret_cast<__half2 const*>(activations+lane*32+j*2);
            #pragma unroll
            for (int i=0;i<Outputs;++i) {
                int const nlocal=warp*Outputs+i;
                auto sz=scale_zero(load_unit(units+(int64_t(chunk*4+lane/8)*c.n+first_n+nlocal)*16),lane%8);
                __half2 const scale=__half2half2(sz.scale),zero=__half2half2(sz.zero);
                uint4 q=*reinterpret_cast<uint4 const*>(packed+nlocal*Pitch+lane*8);
                float2 part[4]={make_float2(0,0),make_float2(0,0),make_float2(0,0),make_float2(0,0)};
                q4_shm_dot_word<0>(q.x,logical,scale,zero,part);
                q4_shm_dot_word<1>(q.y,logical,scale,zero,part);
                q4_shm_dot_word<2>(q.z,logical,scale,zero,part);
                q4_shm_dot_word<3>(q.w,logical,scale,zero,part);
                sums[i] += ((part[0].x+part[1].x)+(part[2].x+part[3].x))+
                           ((part[0].y+part[1].y)+(part[2].y+part[3].y));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i=0;i<Outputs;++i) {
        float v=sums[i];
        #pragma unroll
        for (int distance=1;distance<32;distance*=2) v+=__shfl_xor_sync(0xffffffffu,v,distance);
        if (lane==0) {
            int const col=first_n+warp*Outputs+i;
            if (split==1) c.output[int64_t(row)*c.out_row_stride+col]=v;
            else static_cast<float*>(c.workspace)[(int64_t(row)*split+partition)*c.n+col]=v;
        }
    }
}
