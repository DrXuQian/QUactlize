// Hardware shared-memory transpose of canonical K-pack b16 words.
// CUDA cp.async/ldmatrix only; this does not emulate PPU AIU.
__device__ __forceinline__ void q4_copy16(void* shared, void const* global) {
    unsigned addr = unsigned(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(global) : "memory");
}

template<int TileK, int Warps, bool Pipeline>
__global__ void q4_ldmatrix(qkg_call_v1 c, int split) {
    using namespace quactlize::dev::q4_native;
    constexpr int TileN = 16, StageWords = TileK / 4 * TileN;
    constexpr int Buffers = Pipeline ? 2 : 1;
    __shared__ __align__(16) uint16_t packed[Buffers * StageWords];
    __shared__ float warp_partial[Warps * TileN];
    extern __shared__ __align__(16) unsigned char ashared[];
    auto a = reinterpret_cast<__half*>(ashared);
    int const tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int const nlocal = lane / 4, rpair = 2 * (lane % 4);
    int const first_n = blockIdx.x * TileN;
    for (int i = tid; i < c.k / 8; i += Warps * 32)
        reinterpret_cast<uint4*>(a)[i] = reinterpret_cast<uint4 const*>(c.a)[i];
    auto low = reinterpret_cast<uint16_t const*>(c.low);
    auto issue = [&](int chunk, int buf) {
        #pragma unroll
        for (int i = tid; i < StageWords / 8; i += Warps * 32) {
            int const row = i / 2, col = (i % 2) * 8;
            q4_copy16(packed + buf * StageWords + i * 8,
                      low + size_t(chunk * (TileK / 4) + row) * c.n + first_n + col);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    int const chunks = c.k / TileK;
    float2 acc[4]{};
    if constexpr (Pipeline) issue(0, 0);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        int const buf = Pipeline ? chunk % 2 : 0;
        if constexpr (!Pipeline) issue(chunk, 0);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if constexpr (Pipeline) if (chunk + 1 < chunks) issue(chunk + 1, 1 - buf);
        #pragma unroll
        for (int micro = warp; micro < TileK / 64; micro += Warps) {
            int const mat = lane / 8;
            int const sr = micro * 16 + (mat / 2) * 8 + lane % 8;
            int const sn = (mat % 2) * 8;
            unsigned addr = unsigned(__cvta_generic_to_shared(packed + buf * StageWords + sr * TileN + sn));
            uint32_t q[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(addr));
            int const gbase = chunk * (TileK / 32) + micro * 2;
            auto m0 = aligned_unit(c.units + (size_t(gbase / 8) * c.n + first_n + nlocal) * 16);
            auto m1 = aligned_unit(c.units + (size_t(gbase / 8) * c.n + first_n + nlocal + 8) * 16);
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                int const g = gbase + v / 2;
                auto sz = aligned_scale_zero(v % 2 == 0 ? m0 : m1, g & 7);
                __half2 scale = __half2half2(sz.scale), zero = __half2half2(sz.zero);
                #pragma unroll
                for (int slot = 0; slot < 4; ++slot) {
                    __half2 code;
                    if (slot == 0) code = codes<0>(q[v]);
                    else if (slot == 1) code = codes<1>(q[v]);
                    else if (slot == 2) code = codes<2>(q[v]);
                    else code = codes<3>(q[v]);
                    float2 weight = __half22float2(__hfma2(code, scale, zero));
                    float2 act = __half22float2(*reinterpret_cast<__half2 const*>(a + g * 32 + rpair + slot * 8));
                    acc[v].x = fmaf(weight.x, act.x, acc[v].x);
                    acc[v].y = fmaf(weight.y, act.y, acc[v].y);
                }
            }
        }
        __syncthreads();
    }
    float v0 = (acc[0].x + acc[0].y) + (acc[2].x + acc[2].y);
    float v1 = (acc[1].x + acc[1].y) + (acc[3].x + acc[3].y);
    #pragma unroll
    for (int d = 1; d <= 2; d *= 2) {
        v0 += __shfl_xor_sync(0xffffffffu, v0, d);
        v1 += __shfl_xor_sync(0xffffffffu, v1, d);
    }
    if (lane % 4 == 0) {
        warp_partial[warp * TileN + nlocal] = v0;
        warp_partial[warp * TileN + nlocal + 8] = v1;
    }
    __syncthreads();
    if (tid < TileN) {
        float value = 0;
        #pragma unroll
        for (int w = 0; w < Warps; ++w) value += warp_partial[w * TileN + tid];
        c.output[first_n + tid] = value;
    }
}
