// Each lane reads one contiguous 8/16/32-column K-pack row. Metadata is
// decoded once per (N,group) then exchanged within the eight residue lanes.
// This trades warp shuffles for duplicated metadata decode and A loads.
template<int Width, int Warps, bool StageA, bool Scatter = false, int N = 0, int K = 0>
__global__ void q4_cooperative_metadata(qkg_call_v1 c) {
    using namespace quactlize::dev::q4_native;
    constexpr int Pairs = Width / 2;
    int const tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int const n = N ? N : c.n, k = K ? K : c.k;
    int const first = blockIdx.x * Width, residue = lane % 8;
    auto low = reinterpret_cast<uint16_t const*>(c.low);
    auto a = static_cast<__half const*>(c.a);
    extern __shared__ __align__(16) unsigned char shared_a[];
    if constexpr (StageA) {
        for (int i = tid; i < k / 8; i += Warps * 32)
            reinterpret_cast<uint4*>(shared_a)[i] = reinterpret_cast<uint4 const*>(c.a)[i];
        __syncthreads();
        a = reinterpret_cast<__half const*>(shared_a);
    }
    float2 sums[Pairs]{};
    #pragma unroll
    for (int pass = 0; pass < (k / 128 + Warps - 1) / Warps; ++pass) {
        int const chunk = pass * Warps + warp;
        if (chunk >= k / 128) continue;
        int const g = chunk * 4 + lane / 8;
        constexpr int Vectors = Width < 8 ? 1 : Width / 8;
        uint32_t meta[Vectors], words[Pairs];
        #pragma unroll
        for (int v = 0; v < Vectors; ++v) {
            int const meta_n = Width < 8 ? residue % Width : v * 8 + residue;
            auto unit = aligned_unit(c.units + (size_t(g / 8) * n + first + meta_n) * 16);
            auto sz = aligned_scale_zero(unit, g & 7);
            meta[v] = uint32_t(__half_as_ushort(sz.scale)) | (uint32_t(__half_as_ushort(sz.zero)) << 16);
            if constexpr (Width == 2) {
                words[0] = *reinterpret_cast<uint32_t const*>(low + size_t(g * 8 + residue) * n + first);
            } else if constexpr (Width == 4) {
                uint2 b = *reinterpret_cast<uint2 const*>(low + size_t(g * 8 + residue) * n + first);
                words[0]=b.x; words[1]=b.y;
            } else {
            uint4 b = *reinterpret_cast<uint4 const*>(low + size_t(g * 8 + residue) * n + first + v * 8);
            words[4*v] = b.x; words[4*v+1] = b.y; words[4*v+2] = b.z; words[4*v+3] = b.w;
            }
        }
        float act[4];
        #if defined(Q4_COOPERATIVE_VECTOR_A)
        float4 av=q4_warp_activation(c.a,g,residue);
        act[0]=av.x;act[1]=av.y;act[2]=av.z;act[3]=av.w;
        #else
        #pragma unroll
        for (int slot = 0; slot < 4; ++slot) act[slot] = __half2float(a[g * 32 + residue + 8 * slot]);
        #endif
        #pragma unroll
        for (int p = 0; p < Pairs; ++p) {
            uint32_t s0 = __shfl_sync(0xffffffffu, meta[p/4], (lane & ~7) + 2*(p%4));
            uint32_t s1 = __shfl_sync(0xffffffffu, meta[p/4], (lane & ~7) + 2*(p%4) + 1);
            __half2 scale = __halves2half2(__ushort_as_half(uint16_t(s0)), __ushort_as_half(uint16_t(s1)));
            __half2 zero = __halves2half2(__ushort_as_half(uint16_t(s0>>16)), __ushort_as_half(uint16_t(s1>>16)));
            #pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                __half2 q;
                if (slot == 0) q = codes<0>(words[p]);
                else if (slot == 1) q = codes<1>(words[p]);
                else if (slot == 2) q = codes<2>(words[p]);
                else q = codes<3>(words[p]);
                float2 w = __half22float2(__hfma2(q, scale, zero));
                sums[p].x = fmaf(act[slot], w.x, sums[p].x);
                sums[p].y = fmaf(act[slot], w.y, sums[p].y);
            }
        }
    }
    if constexpr (Scatter) {
        float values[Width];
        #pragma unroll
        for (int p=0;p<Pairs;++p) {values[2*p]=sums[p].x;values[2*p+1]=sums[p].y;}
        float v=q4_reduce_scatter_steps<Width,1>(values,lane);
        __shared__ float warp_value[Warps*Width];
        if(lane<Width) warp_value[warp*Width+lane]=v;
        __syncthreads();
        if(tid<32) {
            float sum=0;
            #pragma unroll
            for(int w=tid/Width;w<Warps;w+=32/Width) sum+=warp_value[w*Width+tid%Width];
            #pragma unroll
            for(int d=Width;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
            if(tid<Width) c.output[first+tid]=sum;
        }
    } else {
    __shared__ float2 warp_partial[Warps * Pairs];
    #pragma unroll
    for (int p = 0; p < Pairs; ++p) {
        float2 v = sums[p];
        #pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            v.x += __shfl_xor_sync(0xffffffffu, v.x, d);
            v.y += __shfl_xor_sync(0xffffffffu, v.y, d);
        }
        if (lane == 0) warp_partial[warp * Pairs + p] = v;
    }
    __syncthreads();
    if (tid < Pairs) {
        float2 v{};
        #pragma unroll
        for (int w = 0; w < Warps; ++w) {
            auto partial = warp_partial[w * Pairs + tid];
            v.x += partial.x; v.y += partial.y;
        }
        *reinterpret_cast<float2*>(c.output + first + 2*tid) = v;
    }
    }
}
