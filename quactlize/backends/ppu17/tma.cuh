#pragma once
#include "kernels.cuh"
#include "prepared.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/device_kernel.h"
#include <stdexcept>

namespace quactlize::ppu17 {

// One warpgroup, two raw stages. The B code words travel through TMA without
// changing the offline artifact. Fine-grained scales are applied only when
// producing the typed GMMA B tile, so group16/group32 remain independent of TK.
template <int Q, class T> struct TmaTile {
  using F = Format<Q>;
  using Compute = GemmTile<T, 64, 64>;
  static constexpr int low_words = 64 / (16 / F::low_bits);
  static constexpr int high_bits = F::high_bits ? F::high_bits : 1;
  static constexpr int high_words = 8; // A partial K128 int1 cohort still loads eight words.
  using ALayout = decltype(cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<T>{},
                      cute::make_shape(cute::_64{}, cute::_64{}, cute::_2{})));
  using LowLayout = decltype(cute::tile_to_shape(cute::GMMA::Layout_MN_SW128_Atom<uint16_t>{},
                      cute::make_shape(cute::_64{}, cute::Int<low_words>{}, cute::_2{})));
  using HighLayout = decltype(cute::tile_to_shape(cute::GMMA::Layout_MN_SW128_Atom<uint16_t>{},
                      cute::make_shape(cute::_64{}, cute::_8{}, cute::_2{})));
  struct Shared {
    cute::array_aligned<T, cute::cosize_v<ALayout>, 128> a;
    cute::array_aligned<uint16_t, cute::cosize_v<LowLayout>, 128> low;
    cute::array_aligned<uint16_t, F::high_bits ? cute::cosize_v<HighLayout> : 1, 128> high;
    cute::array_aligned<T, cute::cosize_v<typename Compute::LayoutB>, 128> b;
    alignas(16) uint64_t ready[2];
  };
  static_assert(sizeof(Shared) < 48 * 1024, "static shared memory must fit without a launch-time attribute change");
};

// PrefetchB is an internal experimental contract, NOT the public PDL v1
// policy: low/high and their descriptors must be immutable and already ready
// before any predecessor starts. Only the first raw B tile is independent;
// units, SF metadata, A and grouped offsets still require the grid wait.
template <int Q, class T, bool SF, class TmaA, class TmaLow, class TmaHigh,
          bool Pdl = false, bool PrefetchB = false>
__global__ __launch_bounds__(128) void tma_wgmma(
    q17_call_v1 c, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    CUTLASS_GRID_CONSTANT TmaLow const tma_low, CUTLASS_GRID_CONSTANT TmaHigh const tma_high) {
  using namespace cute;
  using Tile = TmaTile<Q, T>;
  using F = Format<Q>;
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  static_assert(!PrefetchB || Pdl, "early B requires explicit dependent launch");
  __shared__ typename Tile::Shared shared;
  RowRange rows;
  if constexpr (!PrefetchB) {
    wait_for_predecessors<Pdl>();
    rows = row_range(c, blockIdx.z);
  }
  int mb = int(blockIdx.y) * 64, nb = int(blockIdx.x) * 64;
  if constexpr (!PrefetchB) {
    if (mb >= rows.size || nb >= c.problem.n) return;
  }
  auto s_a = make_tensor(make_smem_ptr(shared.a.data()), typename Tile::ALayout{});
  auto s_low = make_tensor(make_smem_ptr(shared.low.data()), typename Tile::LowLayout{});
  auto s_high = make_tensor(make_smem_ptr(shared.high.data()), typename Tile::HighLayout{});
  auto s_b = make_tensor(make_smem_ptr(shared.b.data()), typename Tile::Compute::LayoutB{});
  auto m_a = tma_a.get_tma_tensor(make_shape(c.problem.m, c.problem.k));
  auto m_low = tma_low.get_tma_tensor(make_shape(c.problem.n, (c.problem.k / F::Low::kPack) * c.problem.experts));
  auto m_high = tma_high.get_tma_tensor(make_shape(c.problem.n, (c.problem.k / (16 / Tile::high_bits)) * c.problem.experts));
  if (threadIdx.x == 0) {
    Barrier::init(&shared.ready[0], 1);
    Barrier::init(&shared.ready[1], 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  if constexpr (PrefetchB) {
    // Keep arrival count one until the dependent row range is known. B may
    // complete early, but cannot flip stage0's phase before A is accounted.
    if (threadIdx.x == 0) {
      constexpr int bytes = 64 * Tile::low_words * 2
          + (F::high_bits ? 64 * Tile::high_words * 2 : 0);
      Barrier::expect_transaction(&shared.ready[0], bytes);
      int lg = int(blockIdx.z) * (c.problem.k / F::Low::kPack) + F::Low::physical_kgroup(0);
      auto gl = local_tile(domain_offset(make_coord(nb, lg), m_low),
                           make_shape(_64{}, Int<Tile::low_words>{}), make_coord(0, 0));
      auto [tl_g, tl_s] = tma_partition(tma_low, Int<0>{}, Layout<_1>{},
            group_modes<0, 2>(s_low(_, _, 0)), group_modes<0, 2>(gl));
      copy(tma_low.with(shared.ready[0]), tl_g, tl_s);
      if constexpr (F::high_bits) {
        int hg = int(blockIdx.z) * (c.problem.k / (16 / Tile::high_bits)) + F::High::physical_kgroup(0);
        auto gh = local_tile(domain_offset(make_coord(nb, hg), m_high),
                             make_shape(_64{}, _8{}), make_coord(0, 0));
        auto [th_g, th_s] = tma_partition(tma_high, Int<0>{}, Layout<_1>{},
              group_modes<0, 2>(s_high(_, _, 0)), group_modes<0, 2>(gh));
        copy(tma_high.with(shared.ready[0]), th_g, th_s);
      }
    }
    wait_for_predecessors<Pdl>();
    rows = row_range(c, blockIdx.z);
    if (mb >= rows.size || nb >= c.problem.n) {
      // No A is needed. Complete the sole arrival and drain speculative B
      // before this CTA's shared storage can be retired/reused.
      if (threadIdx.x == 0) Barrier::arrive(&shared.ready[0]);
      Barrier::wait(&shared.ready[0], 0);
      __syncthreads();
      return;
    }
  }

  auto issue = [&](int iteration, int stage) {
    if (threadIdx.x != 0) return;
    int kb = iteration * 64;
    auto ga = local_tile(domain_offset(make_coord(rows.begin + mb, kb), m_a),
                         make_shape(_64{}, _64{}), make_coord(0, 0));
    int lg = int(blockIdx.z) * (c.problem.k / F::Low::kPack) + F::Low::physical_kgroup(kb);
    auto gl = local_tile(domain_offset(make_coord(nb, lg), m_low),
                         make_shape(_64{}, Int<Tile::low_words>{}), make_coord(0, 0));
    auto [ta_g, ta_s] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
          group_modes<0, 2>(s_a(_, _, stage)), group_modes<0, 2>(ga));
    if constexpr (PrefetchB) {
      if (iteration == 0) {
        // Add A0 to the outstanding B0 bytes and complete the sole arrival.
        // The ordinary stage0 wait covers both; later refills are unchanged.
        Barrier::arrive_and_expect_tx(&shared.ready[stage], 64 * 64 * sizeof(T));
        copy(tma_a.with(shared.ready[stage]), ta_g, ta_s);
        return;
      }
    }
    auto [tl_g, tl_s] = tma_partition(tma_low, Int<0>{}, Layout<_1>{},
          group_modes<0, 2>(s_low(_, _, stage)), group_modes<0, 2>(gl));
    constexpr int bytes = 64 * 64 * sizeof(T) + 64 * Tile::low_words * 2
        + (F::high_bits ? 64 * Tile::high_words * 2 : 0);
    Barrier::arrive_and_expect_tx(&shared.ready[stage], bytes);
    copy(tma_a.with(shared.ready[stage]), ta_g, ta_s);
    copy(tma_low.with(shared.ready[stage]), tl_g, tl_s);
    if constexpr (F::high_bits) {
      int hg = int(blockIdx.z) * (c.problem.k / (16 / Tile::high_bits)) + F::High::physical_kgroup(kb);
      auto gh = local_tile(domain_offset(make_coord(nb, hg), m_high),
                           make_shape(_64{}, _8{}), make_coord(0, 0));
      auto [th_g, th_s] = tma_partition(tma_high, Int<0>{}, Layout<_1>{},
            group_modes<0, 2>(s_high(_, _, stage)), group_modes<0, 2>(gh));
      copy(tma_high.with(shared.ready[stage]), th_g, th_s);
    }
  };
  issue(0, 0);
  if (c.problem.k > 64) issue(1, 1);

  typename Tile::Compute::Mma mma;
  auto thread_mma = mma.get_thread_slice(threadIdx.x);
  auto fa = thread_mma.make_fragment_A(thread_mma.partition_A(s_a));
  auto fb = thread_mma.make_fragment_B(thread_mma.partition_B(s_b));
  auto coords = thread_mma.partition_C(make_identity_tensor(make_shape(_64{}, _64{})));
  auto accum = thread_mma.make_fragment_C(coords);
  clear(accum);
  WeightReader<Q, T, SF> weight(c, blockIdx.z);
  for (int iteration = 0; iteration < c.problem.k / 64; ++iteration) {
    int stage = iteration & 1, kb = iteration * 64;
    Barrier::wait(&shared.ready[stage], (iteration / 2) & 1);
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * 64; i += 128) {
      int ni = i % 64, ki = i / 64, col = nb + ni, kk = kb + ki;
      uint16_t l = s_low(ni, F::Low::physical_kgroup(kk) - F::Low::physical_kgroup(kb), stage), h = 0;
      if constexpr (F::high_bits) {
        int pn;
        if constexpr (Q == 13) pn = F::High::physical_n(col, kk) - nb;
        else pn = ni;
        h = s_high(pn, F::High::physical_kgroup(kk) - F::High::physical_kgroup(kb), stage);
      }
      s_b(ni, ki) = col < c.problem.n ? weight.dequantize(F::decode_words(l, h, col, kk), col, kk) : T(0.f);
    }
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    gemm(mma, fa(_, _, _, stage), fb, accum);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);
    __syncthreads();
    // A is read by GMMA directly from this raw stage. Refill is legal only
    // after GMMA completion plus CTA convergence, not after B conversion.
    if (iteration + 2 < c.problem.k / 64) issue(iteration + 2, stage);
  }
  release_successors<Pdl>();
  CUTE_UNROLL
  for (int i = 0; i < size(accum); ++i) {
    int mi = mb + int(get<0>(coords(i))), ni = nb + int(get<1>(coords(i)));
    if (mi < rows.size && ni < c.problem.n)
      store_element(c.output, c.problem.output_type, int64_t(rows.begin + mi) * c.output_stride + ni, accum(i));
  }
}

template <int Q, class T, bool SF, bool Pdl = false, bool PrefetchB = false>
std::function<int()> prepare_tma(q17_call_v1 const &c, int shared_carveout = -1,
    std::function<void(void const *)> const &configure = {}) {
  using namespace cute;
  using Tile = TmaTile<Q, T>;
  using F = Format<Q>;
  auto a = make_tensor(static_cast<T const *>(c.a), make_shape(c.problem.m, c.problem.k), make_stride(c.a_stride, _1{}));
  auto low = make_tensor(static_cast<uint16_t const *>(c.low),
      make_shape(c.problem.n, (c.problem.k / F::Low::kPack) * c.problem.experts), make_stride(_1{}, c.problem.n));
  // The high descriptor is unused for single-plane types. Its backing pointer
  // remains non-null, and no transaction or barrier bytes are attributed to it.
  auto high = make_tensor(static_cast<uint16_t const *>(F::high_bits ? c.high : c.low),
      make_shape(c.problem.n, (c.problem.k / (16 / Tile::high_bits)) * c.problem.experts), make_stride(_1{}, c.problem.n));
  auto ta = make_tma_atom(SM90_TMA_LOAD{}, a, typename Tile::ALayout{}(_, _, 0), make_shape(_64{}, _64{}));
  auto tl = make_tma_atom(SM90_TMA_LOAD{}, low, typename Tile::LowLayout{}(_, _, 0), make_shape(_64{}, Int<Tile::low_words>{}));
  auto th = make_tma_atom(SM90_TMA_LOAD{}, high, typename Tile::HighLayout{}(_, _, 0), make_shape(_64{}, _8{}));
  // Internal experimental caller only. Existing callers leave the driver
  // policy unchanged. Set before any capture/execution of this specialization.
  if (shared_carveout < -1 || shared_carveout > 100)
    throw std::invalid_argument("invalid shared-memory carveout");
  if (shared_carveout >= 0) {
    auto rc = cudaFuncSetAttribute(tma_wgmma<Q,T,SF,decltype(ta),decltype(tl),decltype(th),Pdl,PrefetchB>,
                                  cudaFuncAttributePreferredSharedMemoryCarveout, shared_carveout);
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
  }
  // Optional internal experiment hook, invoked before capture, never at launch.
  // Public callers retain an empty hook and the original resource policy.
  if (configure) configure(reinterpret_cast<void const *>(
      tma_wgmma<Q,T,SF,decltype(ta),decltype(tl),decltype(th),Pdl,PrefetchB>));
  return [c, ta, tl, th] {
    dim3 grid((c.problem.n + 63) / 64, (c.problem.max_rows + 63) / 64, c.problem.experts);
    return enqueue_kernel<Pdl>(tma_wgmma<Q, T, SF, decltype(ta), decltype(tl), decltype(th), Pdl, PrefetchB>,
                              grid, 128, 0, static_cast<cudaStream_t>(c.stream), c, ta, tl, th);
  };
}
} // namespace quactlize::ppu17
