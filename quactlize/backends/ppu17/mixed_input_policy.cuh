#pragma once
#include "cutlass/gemm/dispatch_policy.hpp"

// The pinned PPU CUTLASS3.6 builder selects the suffixed MixedInput schedule,
// but its mixed-input dispatch policy accepts only the base Hopper schedule.
// Translate exactly this tag at the policy seam; leave the builder's RS MMA,
// scale-only collective, stage count and kernel implementation untouched.
// Included only by the new GPTQ path, never by legacy actlize or GGUF q17.
namespace cutlass::gemm {
template<int Stages, class Cluster>
struct MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<
    Stages, Cluster, KernelTmaWarpSpecializedCooperativeMixedInput>
    : MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<
          Stages, Cluster, KernelTmaWarpSpecializedCooperative> {};
} // namespace cutlass::gemm
