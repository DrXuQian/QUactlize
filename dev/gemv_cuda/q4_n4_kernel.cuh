// Four adjacent N columns per lane, one b64 read per K-pack word row.
// No byte/layout/arithmetic change: each output keeps the N2 FP32 dot order.
template<int Columns,int Warps,int Type>
__global__ void kpack_q4_n4(qkg_call_v1 c,int split) {
    using namespace quactlize::dev::q4_native;
    constexpr int Workers=Warps*32/Columns;
    int const row=blockIdx.z,partition=blockIdx.y;
    int const tid=threadIdx.x,worker=tid/Columns;
    int const col=(blockIdx.x*Columns+tid%Columns)*4;
    int const expert=expert_for(c,row);
    if (expert<0 || expert>=c.experts) {
        if (tid<Columns) {
            auto dst=split==1 ? c.output+int64_t(row)*c.out_row_stride :
                static_cast<float*>(c.workspace)+(int64_t(row)*split+partition)*c.n;
            #pragma unroll
            for (int i=0;i<4;++i) dst[col+i]=__int_as_float(0x7fc00000);
        }
        return;
    }
    int64_t const a_base=c.mode==QKG_INDEXED
        ? int64_t(row/c.topk)*c.a_token_stride+(row%c.topk%c.channels)*c.a_row_stride
        : int64_t(row)*c.a_row_stride;
    auto low=reinterpret_cast<uint16_t const*>(c.low+int64_t(expert)*c.n*c.k/2);
    auto units=c.units+int64_t(expert)*c.n*(c.k/256)*16;
    float x[4]{},y[4]{},z[4]{},t[4]{};
    for (int g=partition*Workers+worker;g<c.k/32;g+=split*Workers) {
        auto p=units+(int64_t(g/8)*c.n+col)*16;
        auto s0=aligned_scale_zero(aligned_unit(p),g&7);
        auto s1=aligned_scale_zero(aligned_unit(p+16),g&7);
        auto s2=aligned_scale_zero(aligned_unit(p+32),g&7);
        auto s3=aligned_scale_zero(aligned_unit(p+48),g&7);
        __half2 scale0=__halves2half2(s0.scale,s1.scale),zero0=__halves2half2(s0.zero,s1.zero);
        __half2 scale1=__halves2half2(s2.scale,s3.scale),zero1=__halves2half2(s2.zero,s3.zero);
        uint32_t lo[8],hi[8];
        #pragma unroll
        for (int r=0;r<8;++r) {
            uint64_t word=*reinterpret_cast<uint64_t const*>(low+R::LowMap::word_index(col,g*32+r,c.n));
            lo[r]=uint32_t(word); hi[r]=uint32_t(word>>32);
        }
        #pragma unroll
        for (int slot=0;slot<4;++slot) {
            #pragma unroll
            for (int first=0;first<8;first+=4) {
                float4 a=aligned_activation<Type>(c.a,a_base+g*32+slot*8+first);
                float av[4]={a.x,a.y,a.z,a.w};
                #pragma unroll
                for (int i=0;i<4;++i) {
                    __half2 code0,code1;
                    if(slot==0) {code0=codes<0>(lo[first+i]);code1=codes<0>(hi[first+i]);}
                    else if(slot==1) {code0=codes<1>(lo[first+i]);code1=codes<1>(hi[first+i]);}
                    else if(slot==2) {code0=codes<2>(lo[first+i]);code1=codes<2>(hi[first+i]);}
                    else {code0=codes<3>(lo[first+i]);code1=codes<3>(hi[first+i]);}
                    float2 w0=__half22float2(__hfma2(code0,scale0,zero0));
                    float2 w1=__half22float2(__hfma2(code1,scale1,zero1));
                    x[i]=fmaf(av[i],w0.x,x[i]);y[i]=fmaf(av[i],w0.y,y[i]);
                    z[i]=fmaf(av[i],w1.x,z[i]);t[i]=fmaf(av[i],w1.y,t[i]);
                }
            }
        }
    }
    constexpr int T=Warps*32;
    __shared__ float partial[4*T];
    partial[tid]=(x[0]+x[1])+(x[2]+x[3]);
    partial[T+tid]=(y[0]+y[1])+(y[2]+y[3]);
    partial[2*T+tid]=(z[0]+z[1])+(z[2]+z[3]);
    partial[3*T+tid]=(t[0]+t[1])+(t[2]+t[3]);
    __syncthreads();
    if(tid<Columns) {
        float v0=0,v1=0,v2=0,v3=0;
        #pragma unroll
        for(int w=0;w<Workers;++w) {
            int i=w*Columns+tid;
            v0+=partial[i];v1+=partial[T+i];v2+=partial[2*T+i];v3+=partial[3*T+i];
        }
        auto dst=split==1 ? c.output+int64_t(row)*c.out_row_stride :
            static_cast<float*>(c.workspace)+(int64_t(row)*split+partition)*c.n;
        dst[col]=v0;dst[col+1]=v1;dst[col+2]=v2;dst[col+3]=v3;
    }
}
