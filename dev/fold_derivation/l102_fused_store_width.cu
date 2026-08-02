// L102 -- DOES THE FUSED BRANCH ACTUALLY EMIT ONE 32-BIT SHARED STORE? Counted in the generated code, not argued.
//
// PPU_PACKED_SCALE_FUSED is byte-neutral by construction: same shared bytes, same results, same counters on every
// path that does not decode. So no run can show whether it is active, and the flag reached the box in a state where
// acu's Shared Store bank conflicts stayed at 81,920 -- exactly what an inactive flag looks like, and exactly what a
// flag whose store did not widen looks like too. l100_fused_active.cu settles the TYPE question (kFusedScaleZero is
// true for the pinned row). This file settles the CODEGEN question.
//
// It compiles the publication step alone through the collective's public forwarder and is meant to be turned into
// PTX, where `st.shared.u32` against `st.shared.u16` in the decode is the whole answer:
//
//   nvcc -std=c++17 -x cu -arch=sm_80 -w -D__HGGCCC__ --expt-relaxed-constexpr -DPPU_PACKED_SCALE=1 \
//        [-DPPU_PACKED_SCALE_FUSED=1] -I dev/fold_derivation/stub_inc -I third_party/actlize/include \
//        -I quactlize/include -ptx dev/fold_derivation/l102_fused_store_width.cu -o out.ptx
//
// Under nvcc kPackedPairFast is off -- the f16x2 asm is PPU-only -- so the SCALAR decode is what gets emitted. That
// changes the arithmetic and not the publication, which is the thing being counted.
#include <hggc_fp16.h>
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
using namespace cute;

using TileShape      = Shape<_16,_128,_256>;
using ScaleTileShape = Shape<_128,_8>;
using WarpShape      = Shape<_16,_16,_256>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>,
    cutlass::layout::ColumnMajorInterleaved<256>, 32,
    float, cute::tuple<TileShape, ScaleTileShape>, WarpShape, cute::Int<2>,
    cutlass::gemm::KernelAiuMultistageMixedInputFinegrainedGs32>::CollectiveOp;

extern "C" __global__ void probe_publish(int stage, int64_t residue_n) {
  __shared__ typename Mainloop::SharedStorage storage;
  Mainloop::probe_packed_decode_stage<true>(storage, stage, int(threadIdx.x), residue_n);
}
