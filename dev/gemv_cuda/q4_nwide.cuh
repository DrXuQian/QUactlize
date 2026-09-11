// Vectorize in N without changing the canonical K-pack byte map.
// P columns per thread, four K-spaced nibbles in each b16 word.
template<int Columns, int Warps, int P, bool StageA, int N = 0, int K = 0, bool Scatter = false>
__global__ void q4_nwide(qkg_call_v1 c, int) {
    using namespace quactlize::dev::q4_native;
    constexpr int Workers = Warps * 32 / Columns;
    constexpr int Pairs = P / 2;
    constexpr int split = 1;  // This experiment's public wrapper admits only S1.
    int const n = N ? N : c.n, k = K ? K : c.k;
    int const tid = threadIdx.x, worker = tid / Columns;
    int const col = blockIdx.x * (Columns * P) + (tid % Columns) * P;
    auto low = reinterpret_cast<uint16_t const*>(c.low);
    auto a = static_cast<__half const*>(c.a);
    extern __shared__ __align__(16) unsigned char shared[];
    if constexpr (StageA) {
        for (int i = tid; i < k / 8; i += Warps * 32)
            reinterpret_cast<uint4*>(shared)[i] = reinterpret_cast<uint4 const*>(c.a)[i];
        __syncthreads();
        a = reinterpret_cast<__half const*>(shared);
    }
    float2 accum[Pairs][2]{};
    #pragma unroll
    for (int pass = 0; pass < (k / 32 + Workers - 1) / Workers; ++pass) {
        int const g = pass * Workers + worker;
        if (g >= k / 32) continue;
        __half2 scale[Pairs], zero[Pairs];
        #pragma unroll
        for (int p = 0; p < Pairs; ++p) {
            auto meta = c.units + (size_t(g / 8) * n + col + 2 * p) * 16;
            auto s0 = aligned_scale_zero(aligned_unit(meta), g & 7);
            auto s1 = aligned_scale_zero(aligned_unit(meta + 16), g & 7);
            scale[p] = __halves2half2(s0.scale, s1.scale);
            zero[p] = __halves2half2(s0.zero, s1.zero);
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint32_t words[4][Pairs];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                #pragma unroll
                for (int v = 0; v < P / 8; ++v) {
                    uint4 value = *reinterpret_cast<uint4 const*>(low + size_t(g * 8 + half * 4 + r) * n + col + v * 8);
                    words[r][4*v] = value.x; words[r][4*v+1] = value.y;
                    words[r][4*v+2] = value.z; words[r][4*v+3] = value.w;
                }
            }
            #pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                float4 av = aligned_activation<0>(a, g * 32 + slot * 8 + half * 4);
                float ax[4] = {av.x, av.y, av.z, av.w};
                #pragma unroll
                for (int p = 0; p < Pairs; ++p) {
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        __half2 q;
                        if (slot == 0) q = codes<0>(words[r][p]);
                        else if (slot == 1) q = codes<1>(words[r][p]);
                        else if (slot == 2) q = codes<2>(words[r][p]);
                        else q = codes<3>(words[r][p]);
                        float2 w = __half22float2(__hfma2(q, scale[p], zero[p]));
                        accum[p][r%2].x = fmaf(ax[r], w.x, accum[p][r%2].x);
                        accum[p][r%2].y = fmaf(ax[r], w.y, accum[p][r%2].y);
                    }
                }
            }
        }
    }
    if constexpr (Scatter && Columns*P<=32) {
        float values[P];
        #pragma unroll
        for(int p=0;p<Pairs;++p) {
            values[2*p]=accum[p][0].x+accum[p][1].x;
            values[2*p+1]=accum[p][0].y+accum[p][1].y;
        }
        int const lane=tid%32;
        float v=q4_reduce_scatter_steps<P,Columns>(values,lane);
        __shared__ float warp_value[Warps*Columns*P];
        if(lane<Columns*P) warp_value[(tid/32)*Columns*P+(lane%Columns)*P+lane/Columns]=v;
        __syncthreads();
        if(tid<32) {
            float sum=0;
            #pragma unroll
            for(int w=tid/(Columns*P);w<Warps;w+=32/(Columns*P))
                sum+=warp_value[w*Columns*P+tid%(Columns*P)];
            #pragma unroll
            for(int d=Columns*P;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
            if(tid<Columns*P) c.output[blockIdx.x*Columns*P+tid]=sum;
        }
    } else {
    __shared__ float2 partial[Warps * Columns * Pairs];
    #pragma unroll
    for (int p = 0; p < Pairs; ++p) {
        float2 v = make_float2(accum[p][0].x + accum[p][1].x, accum[p][0].y + accum[p][1].y);
        #pragma unroll
        for (int step = Columns; step < 32; step *= 2) {
            v.x += __shfl_xor_sync(0xffffffffu, v.x, step);
            v.y += __shfl_xor_sync(0xffffffffu, v.y, step);
        }
        if (tid % 32 < Columns) partial[((tid / 32) * Columns + tid % Columns) * Pairs + p] = v;
    }
    __syncthreads();
    if (tid < Columns * Pairs) {
        float2 v{};
        #pragma unroll
        for (int w = 0; w < Warps; ++w) {
            auto s = partial[w * Columns * Pairs + tid];
            v.x += s.x; v.y += s.y;
        }
        float* dst = split == 1 ? c.output : static_cast<float*>(c.workspace) + blockIdx.y * n;
        *reinterpret_cast<float2*>(dst + blockIdx.x * Columns * P + 2 * tid) = v;
    }
    }
}
