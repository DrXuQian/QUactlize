// THE ONLINE SCALE PRE-PASS, EXPOSED TO PYTHON. Same arithmetic the device kernel runs -- gguf_scale_prepass.hpp's
// group_scale_zero<T, ZMul> is called from both, so there is one implementation and "the two agree" is not something
// a test has to establish.
//
// WHY THIS OP EXISTS AT ALL. tests/test_gguf_golden.py checks quactlize against the OFFICIAL llama.cpp gguf package,
// and it can only do that if quactlize's decode is reachable from Python. Every k-quant constant in
// gguf_scale_layout.hpp was read off our own numpy parser, so without an independent oracle the C++ and the Python
// would agree by construction and prove nothing -- Q3_K's kScaleBias = 32 has no other witness in the tree.
//
// IT IS HOST-ONLY AND PORTABLE ON PURPOSE. gguf_scale_decode.hpp's packed-unit re-exports live in the actlize
// submodule, whose cutlass/half.h pulls <hggc_fp16.h> from the PPU SDK; they are now behind __has_include, so this
// translation unit builds against stock cutlass on a machine with no PPU and no SDK. That is the difference between
// a test anyone can run and a test only the box can run.
#include <torch/extension.h>
#include <torch/script.h>

#include <cstdint>
#include <vector>

#include "gguf_scale_prepass.hpp"
#include "thop/th_utils.h"

namespace torch_ext {

namespace {

using gguf_scale::KType;
using gguf_scale::Traits;

// The ggml type numbers, which are also quactlize.formats.QuantType's values. Kept as the wire format rather than a
// private enum so the Python side does not need a translation table that could drift.
constexpr int64_t kGgmlQ2K = 10, kGgmlQ3K = 11, kGgmlQ4K = 12, kGgmlQ5K = 13, kGgmlQ6K = 14;

// ONE PLACE THAT KNOWS EVERY FORMAT, so adding one cannot be half-done. The alternative -- a switch per property --
// is how a format ends up with the right group count and the wrong block size.
template <class F>
auto dispatch_ktype(int64_t qtype, F&& f) {
  switch (qtype) {
    case kGgmlQ2K: return f(std::integral_constant<KType, KType::Q2_K>{});
    case kGgmlQ3K: return f(std::integral_constant<KType, KType::Q3_K>{});
    case kGgmlQ4K: return f(std::integral_constant<KType, KType::Q4_K>{});
    case kGgmlQ5K: return f(std::integral_constant<KType, KType::Q5_K>{});
    case kGgmlQ6K: return f(std::integral_constant<KType, KType::Q6_K>{});
    default:
      TORCH_CHECK(false, "gguf_scale_prepass: unsupported ggml type ", qtype,
                  " -- expected one of Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14");
  }
}

}  // namespace

// scale_blocks : uint8 [rows, block_bytes]   the format's SCALE block only, not the whole GGUF block. Slicing it out
//                                            is the caller's job because the GGUF block layout differs per format and
//                                            that layout is verified against the official package on the Python side.
// d, dmin      : fp16  [rows]                the superblock header. dmin may be empty for scale-only formats.
// qtype        : ggml type number
// zmul         : the CONSUMER's centre correction, not a property of the format. See the header: the int4 converter
//                emits q-8 where a k-quant means q, so a consumer using it needs zmul=8 and one that does not needs 0.
//                No default, deliberately -- a silently-missing correction is off by 8*scale everywhere and still
//                looks like plausible weights.
std::vector<torch::Tensor> gguf_scale_prepass(torch::Tensor scale_blocks, torch::Tensor d, torch::Tensor dmin,
                                              int64_t qtype, int64_t zmul) {
  CHECK_CPU(scale_blocks);
  CHECK_CONTIGUOUS(scale_blocks);
  CHECK_CPU(d);
  CHECK_CONTIGUOUS(d);
  TORCH_CHECK(scale_blocks.dtype() == torch::kUInt8, "scale_blocks must be uint8");
  TORCH_CHECK(scale_blocks.dim() == 2, "scale_blocks must be [rows, block_bytes]");
  TORCH_CHECK(d.dtype() == torch::kFloat16, "d must be float16");
  TORCH_CHECK(d.dim() == 1 && d.size(0) == scale_blocks.size(0), "d must be [rows] matching scale_blocks");

  int64_t const rows = scale_blocks.size(0);
  bool const has_dmin = dmin.defined() && dmin.numel() > 0;
  if (has_dmin) {
    CHECK_CPU(dmin);
    CHECK_CONTIGUOUS(dmin);
    TORCH_CHECK(dmin.dtype() == torch::kFloat16, "dmin must be float16");
    TORCH_CHECK(dmin.dim() == 1 && dmin.size(0) == rows, "dmin must be [rows] matching scale_blocks");
  }

  return dispatch_ktype(qtype, [&](auto tag) -> std::vector<torch::Tensor> {
    constexpr KType T = decltype(tag)::value;
    using Tr = Traits<T>;
    TORCH_CHECK(scale_blocks.size(1) == Tr::kBlockBytes, "scale_blocks second dim is ", scale_blocks.size(1),
                " but this format's scale block is ", Tr::kBlockBytes, " bytes");
    // A FORMAT WITH A MIN THAT ARRIVES WITHOUT dmin WOULD DECODE WITH dmin = 0, i.e. drop the affine term and return
    // numbers that look fine. Refuse instead.
    TORCH_CHECK(!Tr::kHasMin || has_dmin, "this format has a min channel, so dmin is required");

    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
    torch::Tensor scale = torch::empty({rows, int64_t(Tr::kGroups)}, opts);
    torch::Tensor zero = torch::empty({rows, int64_t(Tr::kGroups)}, opts);

    auto const* blocks = get_ptr<uint8_t const>(scale_blocks);
    auto const* dp = reinterpret_cast<cutlass::half_t const*>(get_ptr<at::Half const>(d));
    auto const* dmp = has_dmin ? reinterpret_cast<cutlass::half_t const*>(get_ptr<at::Half const>(dmin)) : nullptr;
    auto* sp = reinterpret_cast<cutlass::half_t*>(get_ptr<at::Half>(scale));
    auto* zp = reinterpret_cast<cutlass::half_t*>(get_ptr<at::Half>(zero));

    // ZMul IS A TEMPLATE PARAMETER of the shared arithmetic, so the runtime value is dispatched rather than passed.
    // Only the two that exist in this tree are accepted: 0 for a consumer whose converter has no shift, and 8 for the
    // int4 one. A third would be a new consumer, and inventing it silently here is how a wrong constant ships.
    TORCH_CHECK(zmul == 0 || zmul == 8, "zmul must be 0 or 8; got ", zmul,
                " -- it is the consumer's converter shift, not a free parameter");

    for (int64_t r = 0; r < rows; ++r) {
      uint8_t const* blk = blocks + r * Tr::kBlockBytes;
      cutlass::half_t const dd = dp[r];
      cutlass::half_t const dm = dmp ? dmp[r] : cutlass::half_t(0.f);
      for (int g = 0; g < Tr::kGroups; ++g) {
        gguf_scale::GroupScale sz =
            (zmul == 8) ? gguf_scale::prepass::group_scale_zero<T, 8>(blk, g, dd, dm)
                        : gguf_scale::prepass::group_scale_zero<T, 0>(blk, g, dd, dm);
        sp[r * Tr::kGroups + g] = sz.scale;
        zp[r * Tr::kGroups + g] = sz.zero;
      }
    }
    return {scale, zero};
  });
}

// The format's own shape, so Python does not carry a second copy of it. quactlize.formats already has block byte
// counts for the storage arithmetic; these are the SCALE block's, which is a different number, and a test that
// slices the wrong range would otherwise fail in a way that looks like a decode bug.
std::vector<int64_t> gguf_scale_block_shape(int64_t qtype) {
  return dispatch_ktype(qtype, [](auto tag) -> std::vector<int64_t> {
    using Tr = Traits<decltype(tag)::value>;
    return {int64_t(Tr::kBlockBytes), int64_t(Tr::kGroups), int64_t(Tr::kGroupSize),
            int64_t(Tr::kHasMin ? 1 : 0), int64_t(Tr::kScaleBias), int64_t(Tr::kSigned ? 1 : 0)};
  });
}

}  // namespace torch_ext

static auto gguf_scale_prepass_op =
    torch::RegisterOperators("quactlize::gguf_scale_prepass", &torch_ext::gguf_scale_prepass);

static auto gguf_scale_block_shape_op =
    torch::RegisterOperators("quactlize::gguf_scale_block_shape", &torch_ext::gguf_scale_block_shape);
