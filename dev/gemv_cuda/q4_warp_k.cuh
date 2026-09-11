// CUDA experiment: a warp owns several N pairs and shares a K group.
// Low bytes remain canonical [physical_K/4,N]; metadata and affine rounding
// are the same as the measured N4 reader. Only the FP32 reduction tree differs.
template<int Columns, int Warps, bool StageA, bool Superblock>
__global__ void q4_warp_k(qkg_call_v1 c, int split) {
    using namespace quactlize::dev::q4_native;
    constexpr int Klanes = 32 / Columns;
    static_assert(Klanes <= 8 && 8 % Klanes == 0);
    int const tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int const nc = lane % Columns, kr = lane / Columns;
    int const col = blockIdx.x * (2 * Columns) + 2 * nc;
    auto low = reinterpret_cast<uint16_t const*>(c.low);
    auto activations = static_cast<__half const*>(c.a);
    extern __shared__ __align__(16) unsigned char smem[];
    if constexpr (StageA) {
        for (int i = tid; i < c.k / 8; i += Warps * 32)
            reinterpret_cast<uint4*>(smem)[i] = reinterpret_cast<uint4 const*>(c.a)[i];
        __syncthreads();
        activations = reinterpret_cast<__half const*>(smem);
    }
    float2 partial[4]{};
    int const extent = Superblock ? c.k / 256 : c.k / 32;
    for (int item = blockIdx.y * Warps + warp; item < extent; item += Warps * split) {
        uint4 u0{}, u1{};
        if constexpr (Superblock) {
            auto p = c.units + (size_t(item) * c.n + col) * 16;
            u0 = aligned_unit(p); u1 = aligned_unit(p + 16);
        }
        #pragma unroll
        for (int sg = 0; sg < (Superblock ? 8 : 1); ++sg) {
            int const g = Superblock ? item * 8 + sg : item;
            if constexpr (!Superblock) {
                auto p = c.units + (size_t(g / 8) * c.n + col) * 16;
                u0 = aligned_unit(p); u1 = aligned_unit(p + 16);
            }
            auto s0 = aligned_scale_zero(u0, g & 7);
            auto s1 = aligned_scale_zero(u1, g & 7);
            __half2 const scale = __halves2half2(s0.scale, s1.scale);
            __half2 const zero = __halves2half2(s0.zero, s1.zero);
            #pragma unroll
            for (int j = 0; j < 8 / Klanes; ++j) {
                int const r = kr + Klanes * j;
                uint32_t word = *reinterpret_cast<uint32_t const*>(low + size_t(g * 8 + r) * c.n + col);
                #pragma unroll
                for (int slot = 0; slot < 4; ++slot) {
                    __half2 q;
                    if (slot == 0) q = codes<0>(word);
                    else if (slot == 1) q = codes<1>(word);
                    else if (slot == 2) q = codes<2>(word);
                    else q = codes<3>(word);
                    float2 w = __half22float2(__hfma2(q, scale, zero));
                    float a = __half2float(activations[g * 32 + r + 8 * slot]);
                    partial[slot].x = fmaf(a, w.x, partial[slot].x);
                    partial[slot].y = fmaf(a, w.y, partial[slot].y);
                }
            }
        }
    }
    float2 v = make_float2((partial[0].x + partial[1].x) + (partial[2].x + partial[3].x),
                          (partial[0].y + partial[1].y) + (partial[2].y + partial[3].y));
    #pragma unroll
    for (int step = Columns; step < 32; step *= 2) {
        v.x += __shfl_xor_sync(0xffffffffu, v.x, step);
        v.y += __shfl_xor_sync(0xffffffffu, v.y, step);
    }
    __shared__ float2 warp_partial[Warps * Columns];
    if (kr == 0) warp_partial[warp * Columns + nc] = v;
    __syncthreads();
    if (tid < Columns) {
        float2 out{};
        #pragma unroll
        for (int w = 0; w < Warps; ++w) {
            out.x += warp_partial[w * Columns + tid].x;
            out.y += warp_partial[w * Columns + tid].y;
        }
        float* dst = split == 1 ? c.output : static_cast<float*>(c.workspace) + blockIdx.y * c.n;
        *reinterpret_cast<float2*>(dst + col) = out;
    }
}
