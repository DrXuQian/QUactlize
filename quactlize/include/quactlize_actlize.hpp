// THE ONE HEADER A quactlize CONSUMER INCLUDES: actlize's umbrella, unmodified, plus our extensions.
//
// THE SHAPE, AND WHY IT IS THIS ONE. quactlize adds mixed-input collectives that actlize does not have. The
// obvious way to ship them -- take actlize's include list and SWAP the mixed-input entries for our copies -- was
// built first and does not work, for a reason worth keeping written down:
//
//   ppu_include.hpp -> cutlass/gemm/config/gemm_configs.hpp -> ppu_include.hpp        (gemm_configs.hpp:76)
//
// actlize's umbrella is cyclic. Inside actlize the `#pragma once` makes the inner include a no-op, so the cycle
// is invisible. Reproduce the list anywhere else and the guard is not set, so line 76 pulls the REAL umbrella
// with all 53 of its entries -- including the ones the swap existed to replace. Dropping gemm_configs.hpp closes
// the cycle and costs the PPU epilogue builder, the device adapter and group_array_problem_shape, which are
// reachable through nothing else. Measured with `gcc -M -MG` and `gcc -E -H` against a stubbed SDK, not reasoned
// about: the swap version listed BOTH copies of all four takeovers.
//
// So we include actlize's umbrella as-is and add on top. That is what TRT-LLM does -- cpp/cutlass untouched,
// cutlass_extensions layering new primary templates beside it -- and it is only legal if our headers ADD rather
// than REPLACE. Every specialisation below is keyed on a dispatch tag quactlize owns (see
// quactlize_dispatch_policy.hpp), so actlize's own specialisations stay reachable and unshadowed. A quactlize
// header that specialised an actlize tag would compile here and silently take over the vendor path for actlize's
// own callers; ci/check_actlize_pristine.py is what keeps that from happening quietly.
//
// NOT GENERATED, deliberately. An earlier version derived this list from actlize's umbrella, because
// hand-transcribing 49 include lines drifts the first day actlize adds a header. That reason is gone: this file
// no longer restates actlize's list, it includes it. Nine lines that name only our own files cannot drift from
// something they do not mirror.
#pragma once

// actlize, exactly as the vendor ships it. If this line ever needs editing, the extension model has failed.
#include "ppu_include.hpp"

// quactlize's extensions. Order is dependency order: tags and support headers before the collectives that
// specialise on them, and the builder last because it names all three collectives.
#include "quactlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "quactlize_extensions/cutlass/gguf_packed_scale.h"
#include "quactlize_extensions/cutlass/detail/quactlize_mixed_dtype.hpp"
#include "quactlize_extensions/cutlass/gemm/quactlize_dispatch_policy.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_metadata_policy.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/detail/ppu_mixed_pipeline.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/quactlize_mma_mixed_input.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl"
