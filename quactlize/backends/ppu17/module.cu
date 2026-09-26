#include <cuda_runtime.h>
#include "target.cuh"
#include "kernels.cuh"
#include "tma.cuh"

#ifndef Q17_QTYPE
#error "Compile one module per format with Q17_QTYPE"
#endif
#define Q17_JOIN_(a, b) a##b
#define Q17_JOIN(a, b) Q17_JOIN_(a, b)

namespace quactlize::ppu17 {
template <int Q, class T, bool SF>
int launch_typed(q17_call_v1 const &c) {
  auto stream = static_cast<cudaStream_t>(c.stream);
  auto const &cfg = c.config;
  dim3 grid((c.problem.n + cfg.tile_n - 1) / cfg.tile_n,
            (c.problem.max_rows + cfg.tile_m - 1) / cfg.tile_m, c.problem.experts);
  if (cfg.engine == Q17_SIMT) {
    if (cfg.lanes_n == 8) simt<Q, T, SF, 8><<<grid, 128, 0, stream>>>(c);
    else simt<Q, T, SF, 16><<<grid, 128, 0, stream>>>(c);
  } else if (cfg.tile_n == 128) {
    wgmma<Q, T, SF, 128, 64><<<grid, 128, 0, stream>>>(c);
  } else if (cfg.tile_k == 128) {
    wgmma<Q, T, SF, 64, 128><<<grid, 128, 0, stream>>>(c);
  } else {
    wgmma<Q, T, SF, 64, 64><<<grid, 128, 0, stream>>>(c);
  }
  return int(cudaGetLastError());
}

template <int Q, class T> int launch_route(q17_call_v1 const &c, bool expansion) {
  if (!expansion) return c.problem.route == Q17_SF ? launch_typed<Q, T, true>(c) : launch_typed<Q, T, false>(c);
  size_t total = size_t(c.problem.experts) * c.problem.n * (c.problem.k / Format<Q>::group_size);
  unsigned blocks = unsigned(std::min<size_t>((total + 255) / 256, 65535));
  expand_scales<Q, T><<<blocks, 256, 0, static_cast<cudaStream_t>(c.stream)>>>(c);
  return int(cudaGetLastError());
}

int Q17_JOIN(launch_q, Q17_QTYPE)(q17_call_v1 const &c, bool expansion) {
  return c.problem.compute_type == Q17_BF16
      ? launch_route<Q17_QTYPE, cutlass::bfloat16_t>(c, expansion)
      : launch_route<Q17_QTYPE, cutlass::half_t>(c, expansion);
}

template <int Q, class T> std::function<int()> prepare_compute(q17_call_v1 const &c) {
  if (c.config.engine == Q17_TMA_WGMMA)
    return c.problem.route == Q17_SF ? prepare_tma<Q, T, true>(c) : prepare_tma<Q, T, false>(c);
  return [c] { return launch_route<Q, T>(c, false); };
}

std::function<int()> Q17_JOIN(prepare_q, Q17_QTYPE)(q17_call_v1 const &c) {
  return c.problem.compute_type == Q17_BF16
      ? prepare_compute<Q17_QTYPE, cutlass::bfloat16_t>(c)
      : prepare_compute<Q17_QTYPE, cutlass::half_t>(c);
}
} // namespace quactlize::ppu17
