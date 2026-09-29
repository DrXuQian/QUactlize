// Host-only planner audit: compile this same source against each pinned
// CUTLASS tree separately. No GPU launch, allocation, or device query.
#include <cstdio>
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"

template<int NTile, int MTile, int ClusterN>
void inspect(int m, int n, int k) {
  using Tile = cute::Shape<cute::Int<NTile>, cute::Int<MTile>, cute::_64>;
  using Cluster = cute::Shape<cute::Int<ClusterN>, cute::_1, cute::_1>;
  using Scheduler = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamK<Tile, Cluster>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, cutlass::layout::ColumnMajor, 0, cutlass::half_t, cutlass::layout::ColumnMajor, 8,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>>::CollectiveOp;
  constexpr unsigned subtile = Epilogue::get_store_pipe_increment(Tile{});
  auto problem = cute::make_shape(n, m, k, 1);
  cutlass::KernelHardwareInfo hardware{};
  hardware.sm_count = 114; // Recorded physical H800, NOT a PPU1.7 constant.
  typename Scheduler::Arguments args{};
  auto plan = Scheduler::to_underlying_arguments(problem, Tile{}, Cluster{}, hardware, args, nullptr, subtile);
  auto grid = Scheduler::get_grid_shape(plan, problem, Tile{}, Cluster{}, hardware, args);
  auto workspace = Scheduler::template get_workspace_size<decltype(problem), float>(args, problem, hardware, 2, subtile);
  std::printf("{\"shape\":[%d,%d,%d],\"tile\":[%d,%d,64],\"cluster\":%d,"
              "\"grid\":[%u,%u,%u],\"subtiles\":%u,\"workspace\":%zu,"
              "\"units\":%llu,\"splits\":%d,\"sk_tiles\":%u,\"sk_units\":%u,"
              "\"big_units\":%u,\"big_groups\":%u,\"k_per_unit\":%d,\"k_per_big_unit\":%d,"
              "\"groups\":%llu,\"units_per_group\":%llu,\"separate_reduction\":%u,"
              "\"raster\":%d,\"swizzle\":%d}\n",
              m,n,k,NTile,MTile,ClusterN,grid.x,grid.y,grid.z,subtile,workspace,
              static_cast<unsigned long long>(plan.units_per_problem_),plan.divmod_splits_.divisor,
              plan.sk_tiles_,plan.sk_units_,plan.big_units_,plan.big_groups_,
              plan.divmod_k_tiles_per_sk_unit_.divisor,plan.divmod_k_tiles_per_sk_big_unit_.divisor,
              static_cast<unsigned long long>(plan.divmod_sk_groups_.divisor),
              static_cast<unsigned long long>(plan.divmod_sk_units_per_group_.divisor),
              plan.separate_reduction_units_,int(plan.raster_order_),plan.log_swizzle_size_);
}

int main() {
  for (int m : {1,8,128}) {
    for (int n : {512,4096}) {
      int k = n == 512 ? 2048 : 4096;
      inspect<128,16,1>(m,n,k); inspect<128,32,2>(m,n,k);
      inspect<128,64,2>(m,n,k); inspect<128,128,2>(m,n,k);
      inspect<128,256,2>(m,n,k); inspect<256,16,1>(m,n,k);
      inspect<256,32,2>(m,n,k); inspect<256,64,2>(m,n,k);
      inspect<256,128,2>(m,n,k);
    }
  }
}
