#pragma once

#include "target.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "quactlize/include/marlin_format_ppu.hpp"
#include "libtorch_stable/quantization/marlin/marlin_template.h"

namespace quactlize::ppu17 {

// Complete, optional Hopper Marlin implementation, not the WGMMA scheduler
// adapter. The pinned upstream body owns cp.async, fast dequant, MMA, K-warp
// and inter-CTA FP32 reduction, output and lock reset. No legacy PPU atom is
// parsed. Reusing this body is intentional; it is not a new algorithm.
//
// Private classic-Marlin packing preserves the existing GPTQ INT4/group128
// codes and FP16 scale bits. This is NOT a canonical GGUF/K-pack reader.
// Initial admitted geometry: M1..8 or multiples of64, N/K multiples of128.
// Prepared scratch is initialized once; sequential same-stream calls may
// reuse it. Concurrent calls require separate scratch and locks.
struct MarlinGptq {
  struct Configuration {
    void const* kernel;
    int threads, tile_m_blocks, tile_n, tile_k;
  };
  struct Arguments {
    int m, n, k;
    void const* a;
    void const* weights;
    void const* scales;
    void* output;
  };
  struct Workspace {
    std::size_t partial_bytes, total_bytes;
  };

  template<int TM, int TN, int TK, int Threads, bool SmallM>
  static Configuration configuration() {
    constexpr auto dtype = vllm::kFloat16.id();
    return {reinterpret_cast<void const*>(
        ::marlin::Marlin<dtype, vllm::kU4B8.id(), dtype, dtype,
                        Threads, TM, TN/16, TK/16, SmallM, 4, 8, false>),
            Threads, TM, TN, TK};
  }

  static Configuration configuration(int m, int index) {
    if (m > 0 && m <= 8) {
      switch (index) {
        case 0: return configuration<1,128,128,256,true>();
        case 1: return configuration<1,128,64,128,true>();
        case 2: return configuration<1,64,128,128,true>();
      }
    } else if (m >= 64 && m % 64 == 0) {
      switch (index) {
        case 0: return configuration<4,256,64,256,false>();
        case 1: return configuration<4,128,64,128,false>();
        case 2: return configuration<4,64,128,128,false>();
      }
    }
    throw std::invalid_argument("Marlin GPTQ requires M1..8 or a multiple of64, config0..2");
  }

  static int shared_bytes(Configuration const& c) {
    int a = 4*c.tile_m_blocks*16*c.tile_k*2;
    int b = 4*c.tile_k*c.tile_n/2;
    int reduction = c.tile_m_blocks*16*(c.tile_n+8)*2;
    int bias = c.tile_n*2;
    int scales = 4*((c.tile_k+127)/128)*c.tile_n*2;
    return a + std::max({b, reduction, std::min(b,reduction)+bias}) + scales;
  }

  static Workspace workspace(int m, int sm_count) {
    if (m < 1 || sm_count < 1) throw std::invalid_argument("Marlin workspace geometry");
    std::size_t rows = m < 64 ? ((m+15)/16)*16 : 64;
    std::size_t partial = std::size_t(sm_count)*rows*256*sizeof(float);
    return {partial, partial+std::size_t(sm_count)*sizeof(int)};
  }

  static cudaError_t run(Configuration const& cfg, Arguments const& arg,
                         int sm_count, int shared, void* scratch,
                         Workspace const& size, cudaStream_t stream) {
    auto a = static_cast<int4 const*>(arg.a);
    auto b = static_cast<int4 const*>(arg.weights);
    auto c = static_cast<int4*>(arg.output);
    auto tmp = static_cast<int4*>(scratch);
    int4 const* bias = nullptr;
    auto scales = static_cast<int4 const*>(arg.scales);
    int4 const* zeros = nullptr;
    float const* a_scales = nullptr;
    float const* global_scale = nullptr;
    int const* g_idx = nullptr;
    int groups = arg.k/128, m = arg.m, n = arg.n, k = arg.k, lda = k;
    auto locks = reinterpret_cast<int*>(static_cast<uint8_t*>(scratch)+size.partial_bytes);
    bool has_bias = false, atomic = false, fp32_reduce = true;
    void* params[] = {&a,&b,&c,&tmp,&bias,&a_scales,&scales,&global_scale,
        &zeros,&g_idx,&groups,&m,&n,&k,&lda,&locks,&has_bias,&atomic,&fp32_reduce,&shared};
    return cudaLaunchKernel(cfg.kernel, dim3(sm_count), dim3(cfg.threads),
                            params, shared, stream);
  }

  static void pack(int n, int k, uint8_t const* logical, uint8_t* packed) {
    if (n <= 0 || n%128 || k <= 0 || k%128 || !logical || !packed)
      throw std::invalid_argument("Marlin GPTQ packing geometry");
    std::size_t count = std::size_t(n)*k;
    if (quactlize::marlin::detail::ranges_overlap(logical,count,packed,count/2))
      throw std::invalid_argument("Marlin GPTQ packing cannot alias input");
    for (std::size_t i = 0; i < count; ++i)
      if (logical[i] > 15) throw std::invalid_argument("Marlin GPTQ requires biased uint4 codes");
    // Reuse the original Quactlize host-only classic format map. Its public
    // PPU1.0 extent gate stays unchanged; this proven map also covers N128.
    for (int kt = 0; kt < k/16; ++kt)
      for (int nb = 0; nb < n/16; ++nb)
        for (int lane = 0; lane < 32; ++lane) {
          auto offset = quactlize::marlin::detail::classic_word_offset(n,kt,nb,lane);
          auto word = quactlize::marlin::detail::load_classic_word(logical,n,kt,nb,lane);
          quactlize::marlin::detail::store_le_u32(packed+4*offset,word);
        }
  }
};
} // namespace quactlize::ppu17
