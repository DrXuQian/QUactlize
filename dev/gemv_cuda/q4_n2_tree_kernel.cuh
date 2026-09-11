// Small-N control: two columns per lane, FP32 warp/CTA reduction, no
// independent reducer for S1. Canonical words and the FP16 affine stay intact.
template<int Columns,int Warps,int Type>
__global__ void kpack_q4_n2_tree(qkg_call_v1 c,int split) {
    using namespace quactlize::dev::q4_native;
    constexpr int Workers=Warps*32/Columns;
    int const row=blockIdx.z,partition=blockIdx.y;
    int const tid=threadIdx.x,worker=tid/Columns;
    int const col=(blockIdx.x*Columns+tid%Columns)*2;
    int const expert=expert_for(c,row);
    if (expert<0 || expert>=c.experts) {
        if(tid<Columns) {
            auto dst=split==1 ? c.output+int64_t(row)*c.out_row_stride :
                static_cast<float*>(c.workspace)+(int64_t(row)*split+partition)*c.n;
            dst[col]=__int_as_float(0x7fc00000);
            dst[col+1]=__int_as_float(0x7fc00000);
        }
        return;
    }
    int64_t const a_base=c.mode==QKG_INDEXED
        ? int64_t(row/c.topk)*c.a_token_stride+(row%c.topk%c.channels)*c.a_row_stride
        : int64_t(row)*c.a_row_stride;
    auto low=reinterpret_cast<uint16_t const*>(c.low+int64_t(expert)*c.n*c.k/2);
    auto units=c.units+int64_t(expert)*c.n*(c.k/256)*16;
    float x[4]{},y[4]{};
    for(int g=partition*Workers+worker;g<c.k/32;g+=split*Workers) {
        auto p=units+(int64_t(g/8)*c.n+col)*16;
        auto s0=aligned_scale_zero(aligned_unit(p),g&7);
        auto s1=aligned_scale_zero(aligned_unit(p+16),g&7);
        __half2 scale=__halves2half2(s0.scale,s1.scale);
        __half2 zero=__halves2half2(s0.zero,s1.zero);
        uint32_t words[8];
        #pragma unroll
        for(int r=0;r<8;++r)
            words[r]=aligned_word(low+R::LowMap::word_index(col,g*32+r,c.n));
        aligned_dot_slot<0,Type>(c.a,a_base+g*32,words,scale,zero,x,y);
        aligned_dot_slot<1,Type>(c.a,a_base+g*32,words,scale,zero,x,y);
        aligned_dot_slot<2,Type>(c.a,a_base+g*32,words,scale,zero,x,y);
        aligned_dot_slot<3,Type>(c.a,a_base+g*32,words,scale,zero,x,y);
    }
    float v0=(x[0]+x[1])+(x[2]+x[3]);
    float v1=(y[0]+y[1])+(y[2]+y[3]);
    #pragma unroll
    for(int distance=Columns;distance<32;distance*=2) {
        v0+=__shfl_xor_sync(0xffffffffu,v0,distance);
        v1+=__shfl_xor_sync(0xffffffffu,v1,distance);
    }
    constexpr int T=Warps*Columns;
    __shared__ float partial[2*T];
    if((tid&31)<Columns) {
        int i=(tid/32)*Columns+(tid&31);
        partial[i]=v0;partial[T+i]=v1;
    }
    __syncthreads();
    if(tid<Columns) {
        v0=0;v1=0;
        #pragma unroll
        for(int w=0;w<Warps;++w) {
            int i=w*Columns+tid;
            v0+=partial[i];v1+=partial[T+i];
        }
        auto dst=split==1 ? c.output+int64_t(row)*c.out_row_stride :
            static_cast<float*>(c.workspace)+(int64_t(row)*split+partition)*c.n;
        dst[col]=v0;dst[col+1]=v1;
    }
}
