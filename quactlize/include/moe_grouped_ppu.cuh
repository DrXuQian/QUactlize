// Common grouped problem types and the matching PPU kernel specializations.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "ppu_group_schedule.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "quactlize_actlize.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "ppu_mixed_policy.hpp"
#include "cutlass/epilogue/collective/builders/ppu_builder.inl"
#include "ppu_aiu_gemm_mixed_input_group.hpp"   // the new grouped mixed-input GemmUniversal specialization
#include "actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_persistent.hpp"

#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/detail/layout.hpp"

namespace moe_grouped_ppu {
using DStride = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using GroupShape = cute::Shape<int, int, int>;
using GroupProblemShape = cutlass::gemm::GroupProblemShape<GroupShape>;
} // namespace moe_grouped_ppu
