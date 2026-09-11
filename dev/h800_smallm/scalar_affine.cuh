// One output column per lane: small dense batches can expose more CTAs
// without launching an inter-CTA Split-K reducer or changing resident bytes.
template<int Columns,int Warps,int N,int K,bool Indexed>
__global__ void q4_scalar_affine(qkg_call_v1 c) {
    using namespace quactlize::dev::q4_native;
    int const rr=blockIdx.y;
    int const token=rr/c.topk,slot=rr%c.topk;
    int const expert=!Indexed?0:c.ids[int64_t(token)*c.ids_stride+slot];
    int64_t const abase=!Indexed?int64_t(rr)*c.a_row_stride:
        int64_t(token)*c.a_token_stride+int64_t(slot%c.channels)*c.a_row_stride;
    auto a=static_cast<__half const*>(c.a)+abase;
    auto low=reinterpret_cast<uint16_t const*>(c.low)+int64_t(expert)*N*K/4;
    auto units=c.units+int64_t(expert)*N*K/16;
    auto output=c.output+int64_t(rr)*c.out_row_stride;
    constexpr int Workers=Warps*32/Columns;
    int const tid=threadIdx.x,lane=tid%32;
    int const worker=tid/Columns,col=blockIdx.x*Columns+tid%Columns;
    float total=0.f;
    #pragma unroll
    for(int pass=0;pass<(K/32+Workers-1)/Workers;++pass) {
        int const g=pass*Workers+worker;
        if(g>=K/32) continue;
        uint4 u=aligned_unit(units+(size_t(g/8)*N+col)*16);
        float dot=0.f,a_sum=0.f;
        #pragma unroll
        for(int half=0;half<2;++half) {
            uint32_t words[4];
            #pragma unroll
            for(int r=0;r<4;++r) words[r]=low[size_t(g*8+half*4+r)*N+col];
            #pragma unroll
            for(int s=0;s<4;++s) {
                float4 av=aligned_activation<0>(a,g*32+s*8+half*4);
                float ax[4]={av.x,av.y,av.z,av.w};
                a_sum+=(av.x+av.y)+(av.z+av.w);
                #pragma unroll
                for(int r=0;r<4;++r) {
                    __half2 q;
                    if(s==0) q=codes<0>(words[r]);
                    else if(s==1) q=codes<1>(words[r]);
                    else if(s==2) q=codes<2>(words[r]);
                    else q=codes<3>(words[r]);
                    dot=fmaf(ax[r],__half2float(__low2half(q)),dot);
                }
            }
        }
        float2 sz=q4_affine_header(u,g&7);
        total+=fmaf(sz.x,dot,sz.y*a_sum);
    }
    #pragma unroll
    for(int d=Columns;d<32;d*=2) total+=__shfl_xor_sync(0xffffffffu,total,d);
    __shared__ float partial[Warps*Columns];
    if(lane<Columns) partial[(tid/32)*Columns+lane]=total;
    __syncthreads();
    if(tid<Columns) {
        float sum=0.f;
        #pragma unroll
        for(int w=0;w<Warps;++w) sum+=partial[w*Columns+tid];
        output[blockIdx.x*Columns+tid]=sum;
    }
}
