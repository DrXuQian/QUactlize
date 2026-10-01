#include <cuda_runtime.h>
#include "target.cuh"
#include "kernels.cuh"
#include "simt_cache.cuh"
#include "tma.cuh"
#include <stdexcept>

#ifndef Q17_QTYPE
#error "Compile one module per format with Q17_QTYPE"
#endif
#define Q17_JOIN_(a, b) a##b
#define Q17_JOIN(a, b) Q17_JOIN_(a, b)

namespace quactlize::ppu17 {
template <int Q, class T, bool SF, bool Pdl>
int launch_typed(q17_call_v1 const &c) {
  auto stream = static_cast<cudaStream_t>(c.stream);
  auto const &cfg = c.config;
  dim3 grid((c.problem.n + cfg.tile_n - 1) / cfg.tile_n,
            (c.problem.max_rows + cfg.tile_m - 1) / cfg.tile_m, c.problem.experts);
  if (cfg.engine == Q17_SIMT) {
    if (cfg.tile_k == 32) {
      if constexpr (Q == 12) {
        if (cfg.threads==512) {
          if(cfg.lanes_n==4) return launch_q4_cached<T,SF,4,16,Pdl>(c,grid,stream);
          else return launch_q4_cached<T,SF,16,16,Pdl>(c,grid,stream);
        }
        else if (cfg.lanes_n == 4) return launch_q4_cached<T,SF,4,4,Pdl>(c,grid,stream);
        else if (cfg.lanes_n == 8) return launch_q4_cached<T,SF,8,4,Pdl>(c,grid,stream);
        else return launch_q4_cached<T,SF,16,4,Pdl>(c,grid,stream);
      } else return int(cudaErrorInvalidValue);
    }
    else if (cfg.lanes_n == 8) return enqueue_kernel<Pdl>(simt<Q,T,SF,8,Pdl>,grid,128,0,stream,c);
    else return enqueue_kernel<Pdl>(simt<Q,T,SF,16,Pdl>,grid,128,0,stream,c);
  } else if (cfg.tile_n == 128) {
    return enqueue_kernel<Pdl>(wgmma<Q,T,SF,128,64,Pdl>,grid,128,0,stream,c);
  } else if (cfg.tile_k == 128) {
    return enqueue_kernel<Pdl>(wgmma<Q,T,SF,64,128,Pdl>,grid,128,0,stream,c);
  } else {
    return enqueue_kernel<Pdl>(wgmma<Q,T,SF,64,64,Pdl>,grid,128,0,stream,c);
  }
}

template <int Q, class T, bool Pdl> int launch_route(q17_call_v1 const &c, bool expansion) {
  if (!expansion) return c.problem.route == Q17_SF ? launch_typed<Q,T,true,Pdl>(c) : launch_typed<Q,T,false,Pdl>(c);
  size_t total = size_t(c.problem.experts) * c.problem.n * (c.problem.k / Format<Q>::group_size);
  unsigned blocks = unsigned(std::min<size_t>((total + 255) / 256, 65535));
  return enqueue_kernel<Pdl>(expand_scales<Q,T,Pdl>,blocks,256,0,static_cast<cudaStream_t>(c.stream),c);
}

template <int Q, bool Pdl> int launch_format(q17_call_v1 const &c, bool expansion) {
  return c.problem.compute_type == Q17_BF16
      ? launch_route<Q,cutlass::bfloat16_t,Pdl>(c,expansion)
      : launch_route<Q,cutlass::half_t,Pdl>(c,expansion);
}

int Q17_JOIN(launch_q, Q17_QTYPE)(q17_call_v1 const &c, bool expansion, bool pdl) {
#if defined(Q17_ENABLE_PDL)
  if (pdl) return launch_format<Q17_QTYPE,true>(c,expansion);
#else
  if (pdl) return int(cudaErrorNotSupported);
#endif
  return launch_format<Q17_QTYPE,false>(c,expansion);
}

template <int Q, class T, bool Pdl> std::function<int()> prepare_compute(q17_call_v1 const &c) {
  if (c.config.engine == Q17_TMA_WGMMA)
    return c.problem.route == Q17_SF ? prepare_tma<Q,T,true,Pdl>(c) : prepare_tma<Q,T,false,Pdl>(c);
  return [c] { return launch_route<Q,T,Pdl>(c,false); };
}

template <int Q, bool Pdl> std::function<int()> prepare_format(q17_call_v1 const &c) {
  return c.problem.compute_type == Q17_BF16
      ? prepare_compute<Q,cutlass::bfloat16_t,Pdl>(c)
      : prepare_compute<Q,cutlass::half_t,Pdl>(c);
}

std::function<int()> Q17_JOIN(prepare_q, Q17_QTYPE)(q17_call_v1 const &c, bool pdl) {
#if defined(Q17_ENABLE_PDL)
  if (pdl) return prepare_format<Q17_QTYPE,true>(c);
#else
  if (pdl) throw std::invalid_argument("PDL was not compiled");
#endif
  return prepare_format<Q17_QTYPE,false>(c);
}
} // namespace quactlize::ppu17
