#pragma once

#include "io.cuh"
#include "launch.cuh"
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cutlass/arch/barrier.h"

namespace quactlize::ppu17 {

template <int Q, class T, bool SF, int LanesN, bool Pdl = false>
__global__ void simt(q17_call_v1 c) {
  constexpr int lanes_k = 32 / LanesN, columns = 4 * LanesN;
  int expert = blockIdx.z;
  wait_for_predecessors<Pdl>();
  auto rows = row_range(c, expert);
  int row = blockIdx.y;
  if (row >= rows.size) return;
  int lane_n = threadIdx.x % LanesN, lane_k = (threadIdx.x % 32) / LanesN;
  int col = int(blockIdx.x) * columns + int(threadIdx.x / 32) * LanesN + lane_n;
  WeightReader<Q, T, SF> weight(c, expert);
  float sum = 0;
  if (col < c.problem.n) {
    for (int k = lane_k; k < c.problem.k; k += lanes_k) {
      T a(load_element(c.a, c.problem.input_type, int64_t(rows.begin + row) * c.a_stride + k));
      sum = fmaf(float(a), float(weight(col, k)), sum);
    }
  }
  // All lanes participate, including N-tail lanes. Only matching N columns
  // communicate. Accumulation and the warp reduction both remain FP32.
  release_successors<Pdl>();
  for (int delta = LanesN; delta < 32; delta *= 2)
    sum += __shfl_xor_sync(0xffffffffu, sum, delta);
  if (lane_k == 0 && col < c.problem.n)
    store_element(c.output, c.problem.output_type,
                  int64_t(rows.begin + row) * c.output_stride + col, sum);
}

template <class T, int TN, int TK> struct GemmTile {
  static constexpr int tm = 64;
  using Shape = cute::Shape<cute::_64, cute::Int<TN>, cute::Int<TK>>;
  using LayoutA = decltype(cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<T>{},
                                             cute::make_shape(cute::_64{}, cute::Int<TK>{})));
  using LayoutB = decltype(cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<T>{},
                                             cute::make_shape(cute::Int<TN>{}, cute::Int<TK>{})));
  using Mma = decltype(cute::make_tiled_mma(cute::SM90::GMMA::ss_op_selector<T, T, float, Shape>()));
  struct Shared {
    cute::array_aligned<T, cute::cosize_v<LayoutA>, 128> a;
    cute::array_aligned<T, cute::cosize_v<LayoutB>, 128> b;
  };
};

template <int Q, class T, bool SF, int TN, int TK, bool Pdl = false>
__global__ __launch_bounds__(128) void wgmma(q17_call_v1 c) {
  using namespace cute;
  using Tile = GemmTile<T, TN, TK>;
  __shared__ typename Tile::Shared shared;
  wait_for_predecessors<Pdl>();
  auto rows = row_range(c, blockIdx.z);
  int m_base = int(blockIdx.y) * Tile::tm, n_base = int(blockIdx.x) * TN;
  // CTA-uniform early return; no warpgroup is partially active.
  if (m_base >= rows.size || n_base >= c.problem.n) return;
  auto a = make_tensor(make_smem_ptr(shared.a.data()), typename Tile::LayoutA{});
  auto b = make_tensor(make_smem_ptr(shared.b.data()), typename Tile::LayoutB{});
  typename Tile::Mma mma;
  auto thread_mma = mma.get_thread_slice(threadIdx.x);
  auto frag_a = thread_mma.make_fragment_A(thread_mma.partition_A(a));
  auto frag_b = thread_mma.make_fragment_B(thread_mma.partition_B(b));
  auto coords = thread_mma.partition_C(make_identity_tensor(make_shape(_64{}, Int<TN>{})));
  auto accum = thread_mma.make_fragment_C(coords);
  static_assert(std::is_same_v<typename decltype(accum)::value_type, float>);
  clear(accum);
  WeightReader<Q, T, SF> weight(c, blockIdx.z);

  for (int kb = 0; kb < c.problem.k; kb += TK) {
    // A: K-contiguous global loads. B: consecutive lanes own consecutive N
    // for a fixed K, matching the resident b16[K/pack,N] code planes.
    for (int i = threadIdx.x; i < Tile::tm * TK; i += 128) {
      int mi = i / TK, ki = i % TK;
      a(mi, ki) = (m_base + mi < rows.size && kb + ki < c.problem.k)
          ? T(load_element(c.a, c.problem.input_type,
                           int64_t(rows.begin + m_base + mi) * c.a_stride + kb + ki)) : T(0.f);
    }
    for (int i = threadIdx.x; i < TN * TK; i += 128) {
      int ni = i % TN, ki = i / TN;
      b(ni, ki) = (n_base + ni < c.problem.n && kb + ki < c.problem.k)
          ? weight(n_base + ni, kb + ki) : T(0.f);
    }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    gemm(mma, frag_a, frag_b, accum);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);
    // No producer may overwrite either shared tile before the whole
    // warpgroup has finished reading it through the async proxy.
    __syncthreads();
  }
  release_successors<Pdl>();
  CUTE_UNROLL
  for (int i = 0; i < size(accum); ++i) {
    int mi = m_base + int(get<0>(coords(i))), ni = n_base + int(get<1>(coords(i)));
    if (mi < rows.size && ni < c.problem.n)
      store_element(c.output, c.problem.output_type,
                    int64_t(rows.begin + mi) * c.output_stride + ni, accum(i));
  }
}

template <int Q, class T, bool Pdl = false>
__global__ void expand_scales(q17_call_v1 c) {
  wait_for_predecessors<Pdl>();
  // The consumer may initialize, but must wait before reading scale/zero.
  release_successors<Pdl>();
  using F = Format<Q>;
  size_t per_expert = size_t(c.problem.n) * (c.problem.k / F::group_size);
  size_t total = per_expert * c.problem.experts;
  auto scale = static_cast<T *>(const_cast<void *>(c.scale));
  auto zero = static_cast<T *>(const_cast<void *>(c.zero));
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += size_t(gridDim.x) * blockDim.x) {
    int expert = i / per_expert;
    size_t local = i % per_expert;
    auto units = static_cast<uint8_t const *>(c.units)
        + size_t(expert) * F::unit_bytes(c.problem.n, c.problem.k);
    auto affine = F::template affine<T>(units, c.problem.n, local % c.problem.n, local / c.problem.n);
    scale[i] = affine.scale;
    zero[i] = affine.zero;
  }
}
} // namespace quactlize::ppu17
