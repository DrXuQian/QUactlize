#pragma once
#include "ppu_include.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace quactlize::compat {
// Native operands and FP32 accumulation: reuse the existing PPU grouped
// scheduler/AIU/MMA stack, not a new emulation or per-slot weight expansion.
template <class T> struct Grouped {
    using Shape = cute::Shape<int, int, int>;
    using Problem = cutlass::gemm::GroupProblemShape<Shape>;
    using Tile = cute::Shape<cute::_32, cute::_64, cute::_64>;
    using Warp = cute::Shape<cute::_32, cute::_32, cute::_64>;
    using Row = cutlass::layout::RowMajor;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp, Tile, Warp,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, T, Row *, 8, T, Row *, 8,
        cutlass::epilogue::EpiloguePtrArraySimtVectorized,
        cutlass::epilogue::fusion::LinearCombination<T, float>>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp, T, Row *, 8, T,
        cutlass::layout::ColumnMajor *, 8, float, Tile, Warp, cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Problem, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using SA = typename Kernel::InternalStrideA;
    using SB = typename Kernel::InternalStrideB;
    using SD = typename Kernel::InternalStrideD;
    struct Metadata {
        Shape shapes[4];
        T const *a[4], *b[4];
        T *c[4];
        SA sa[4];
        SB sb[4];
        SD sc[4];
    };
    static_assert(sizeof(Metadata) <= 4096, "group metadata exceeds queried scratch");
    static int configure() { return Gemm::maximum_active_blocks() > 0 ? 0 : QPM_RUNTIME; }
    static int launch(Metadata *d, int groups, int device, int cus, void *scratch, hggcStream_t stream) {
        Problem problem;
        problem.num_groups = groups;
        problem.problem_shapes = d->shapes;
        problem.host_problem_shapes = nullptr;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            problem,
            {d->a, d->sa, d->b, d->sb},
            {{}, static_cast<T const **>(nullptr), typename Epilogue::StrideC{}, d->c, d->sc},
            cutlass::KernelHardwareInfo{device, cus}};
        if (Gemm::get_workspace_size(args) != 0 || Gemm::can_implement(args) != cutlass::Status::kSuccess)
            return QPM_INVALID;
        auto params = Kernel::to_underlying_arguments(args, scratch);
        return Gemm::run(params, stream) == cutlass::Status::kSuccess ? 0 : QPM_RUNTIME;
    }
};
template <class T>
__global__ void group_metadata(typename Grouped<T>::Metadata *d, int const *offsets, T const *a, T const *w,
                               T *c, int first_expert, int groups, int first_column, int columns,
                               int n_stride, int k) {
    int e = threadIdx.x;
    if (e >= groups)
        return;
    int start = offsets[first_expert + e], rows = offsets[first_expert + e + 1] - start;
    d->shapes[e] = cute::make_shape(rows, columns, k);
    d->a[e] = a + int64_t(start) * k;
    d->b[e] = w + int64_t(e) * columns * k;
    d->c[e] = c + int64_t(start) * n_stride + first_column;
    d->sa[e] = cutlass::make_cute_packed_stride(typename Grouped<T>::SA{}, cute::make_shape(rows, k, 1));
    d->sb[e] = cutlass::make_cute_packed_stride(typename Grouped<T>::SB{}, cute::make_shape(columns, k, 1));
    d->sc[e] =
        cutlass::make_cute_packed_stride(typename Grouped<T>::SD{}, cute::make_shape(rows, n_stride, 1));
}
} // namespace quactlize::compat
