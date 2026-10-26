#pragma once
#include "simt_q8_vector.cuh"

namespace quactlize::execution::simt {

// Q8 has scale-only units, K-pack2 words and an existing fast code_pair.
// Reuse those primitives and the measured cursor/half-window schedules;
// never pass Q8 through the K-quant metadata decoder. Window is one or two
// half-groups loaded before math. All tokens share the loaded words/codes.
template<int Tokens,int Columns,int Warps,int P,int Window>
__global__ void q8_token_cta_reader(qkg_call_v1 call,int split) {
    static_assert(Tokens>=2 && Tokens<=8 && (Window==1 || Window==2));
    static_assert(P==4 && ((Columns==4 && Warps==8) || (Columns==8 && Warps==4)));
    constexpr int TileN=Columns*P,Workers=Warps*32/Columns,Pairs=P/2;
    unsigned const tid=threadIdx.x,lane=tid%32,worker=tid/Columns;
    unsigned const tile=blockIdx.x,partition=blockIdx.z;
    unsigned const col=tile*TileN+(tid%Columns)*P;
    auto const* low=reinterpret_cast<uint16_t const*>(call.low);
    uint64_t weight_cursor=uint64_t(partition*Workers+worker)*16*call.n+col;
    uint64_t scale_cursor=2*(uint64_t(partition*Workers+worker)*call.n+col);
    uint64_t const weight_step=uint64_t(split)*Workers*16*call.n;
    uint64_t const scale_step=uint64_t(split)*Workers*2*call.n;
    float2 total[Tokens][Pairs]{};
    for(unsigned g=partition*Workers+worker;g<unsigned(call.k)/32;
            g+=split*Workers,weight_cursor+=weight_step,scale_cursor+=scale_step) {
        float scale[P];q8_vector::scales<P>(call.units+scale_cursor,scale);
        uint4 packet[Tokens];
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            Activation<QKG_F32,0> a{static_cast<float const*>(call.a)+int64_t(t)*call.a_row_stride};
            packet[t]=activation_packet<QKG_F32,0,Columns,32>(a,g,lane);
        }
        float2 dot[Tokens][Pairs]{};
        #pragma unroll
        for(int first=0;first<2;first+=Window) {
            uint32_t words[Window][2][4][Pairs];
            #pragma unroll
            for(int half=0;half<Window;++half)
                #pragma unroll
                for(int segment=0;segment<2;++segment)
                    #pragma unroll
                    for(int residue=0;residue<4;++residue)
                        load_words<P>(low+weight_cursor+uint64_t(segment*8+(first+half)*4+residue)*call.n,
                                      words[half][segment][residue]);
            // Same half -> segment -> slot -> residue FMA order as Q8 refresh.
            #pragma unroll
            for(int half=0;half<Window;++half)
                #pragma unroll
                for(int segment=0;segment<2;++segment)
                    #pragma unroll
                    for(int slot=0;slot<2;++slot) {
                        int const offset=segment*16+slot*8+(first+half)*4;
                        float values[Tokens][4];
                        #pragma unroll
                        for(int t=0;t<Tokens;++t) {
                            float2 x=activation_pair<0,Columns,32>(packet[t],offset,lane);
                            float2 y=activation_pair<0,Columns,32>(packet[t],offset+2,lane);
                            values[t][0]=x.x;values[t][1]=x.y;values[t][2]=y.x;values[t][3]=y.y;
                        }
                        #pragma unroll
                        for(int residue=0;residue<4;++residue)
                            #pragma unroll
                            for(int p=0;p<Pairs;++p) {
                                float2 code=slot==0?q8_vector::code_pair<0>(words[half][segment][residue][p]):
                                                   q8_vector::code_pair<1>(words[half][segment][residue][p]);
                                #pragma unroll
                                for(int t=0;t<Tokens;++t) {
                                    dot[t][p].x=fmaf(values[t][residue],code.x,dot[t][p].x);
                                    dot[t][p].y=fmaf(values[t][residue],code.y,dot[t][p].y);
                                }
                            }
                    }
        }
        #pragma unroll
        for(int t=0;t<Tokens;++t)
            #pragma unroll
            for(int p=0;p<Pairs;++p) {
                total[t][p].x+=scale[2*p]*dot[t][p].x;
                total[t][p].y+=scale[2*p+1]*dot[t][p].y;
            }
    }
    __shared__ float partial[Tokens][Warps*TileN];
    #pragma unroll
    for(int t=0;t<Tokens;++t) {
        float values[P];
        #pragma unroll
        for(int p=0;p<Pairs;++p) {values[2*p]=total[t][p].x;values[2*p+1]=total[t][p].y;}
        float value=q4_s1::q4_reduce_scatter_steps<P,Columns>(values,lane);
        if(lane<TileN)partial[t][(tid/32)*TileN+(lane%Columns)*P+lane/Columns]=value;
    }
    __syncthreads();
    if(tid<32) {
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            float sum=0;q4_s1::q4_medium_fold<0,Warps,TileN>(sum,partial[t],tid);
            #pragma unroll
            for(int d=TileN;d<32;d*=2)sum+=__shfl_xor_sync(0xffffffffu,sum,d);
            if(tid<TileN) {
                unsigned const n=tile*TileN+tid;
                if(split==1)call.output[int64_t(t)*call.out_row_stride+n]=sum;
                else static_cast<float*>(call.workspace)[(int64_t(t)*split+partition)*call.n+n]=sum;
            }
        }
    }
}

template<int Tokens,int Columns,int Warps,int P,int Window>
int q8_token_cta_launch(qkg_simt_call_v2 const& d,int split) {
    auto const& c=d.call;
    if(c.qtype!=8 || c.mode!=QKG_DENSE || c.rows!=Tokens || c.experts!=1 ||
       c.input_type!=QKG_F32 || d.compute_type!=QKG_COMPUTE_F16)return QKG_SHAPE;
    auto stream=static_cast<hggcStream_t>(c.stream);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    q8_token_cta_reader<Tokens,Columns,Warps,P,Window>
        <<<dim3(c.n/(Columns*P),1,split),Warps*32,0,stream>>>(c,split);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    launch_reduction<8>(c,split,stream);
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
}
