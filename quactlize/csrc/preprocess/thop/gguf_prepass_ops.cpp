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
#include <cstring>
#include <cstdlib>
#include <vector>

#include "gguf_scale_prepass.hpp"
#include "gguf_vecdot.hpp"
#include "gguf_packed_unit.hpp"
#include "thop/th_utils.h"
#include "thop/ppu_backend.h"

namespace torch_ext {

namespace {

using gguf_scale::KType;
using gguf_scale::Traits;

// The ggml type numbers, which are also quactlize.formats.QuantType's values. Kept as the wire format rather than a
// private enum so the Python side does not need a translation table that could drift.
// Big enough that every golden test and every hand check runs untouched, small enough that no real workload does.
constexpr int64_t kCpuReferenceRowLimit = 4096;
bool cpu_reference_allowed() {
  char const* e = std::getenv("QUACTLIZE_ALLOW_CPU_REFERENCE");
  return e && *e && *e != '0';
}

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

    // BIT-PRESERVING COPIES, NOT reinterpret_cast. at::Half and cutlass::half_t are distinct types with the same
    // size and representation, and reading one through the other is undefined behaviour regardless -- strict
    // aliasing does not care that the bits match, and this is compiled with -O2. It usually works, which is worse
    // than failing: the correctness ORACLE for the whole format family would have rested on UB.
    auto const* blocks = get_ptr<uint8_t const>(scale_blocks);
    static_assert(sizeof(at::Half) == sizeof(cutlass::half_t), "half types must be the same width to copy bits");
    auto to_cutlass = [](at::Half const* src, int64_t n) {
      std::vector<cutlass::half_t> v(size_t(n ? n : 0));
      for (int64_t i = 0; i < n; ++i) std::memcpy(&v[size_t(i)], src + i, sizeof(cutlass::half_t));
      return v;
    };
    std::vector<cutlass::half_t> dv = to_cutlass(get_ptr<at::Half const>(d), rows);
    std::vector<cutlass::half_t> dmv = has_dmin ? to_cutlass(get_ptr<at::Half const>(dmin), rows)
                                                : std::vector<cutlass::half_t>();
    std::vector<cutlass::half_t> sv(size_t(rows) * Tr::kGroups), zv(size_t(rows) * Tr::kGroups);

    // CALL THE SHARED HOST REFERENCE rather than repeating its loop. The first version of this op had its own
    // contiguous double loop, so prepass_host's descriptor and stride arithmetic -- the part the device kernel
    // mirrors -- was never executed by any test, and "one implementation, so they cannot disagree" was true only of
    // the per-group arithmetic. Going through BlockDesc/PlaneDesc means the golden tests exercise the placement too.
    if (auto const* api = ppu_backend::load()) {
      TORCH_CHECK(api->prepass(blocks, int64_t(Tr::kBlockBytes),
                               reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(d)),
                               has_dmin ? reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(dmin)) : nullptr,
                               int(rows), reinterpret_cast<uint16_t*>(get_ptr<at::Half>(scale)),
                               reinterpret_cast<uint16_t*>(get_ptr<at::Half>(zero)),
                               int(Tr::kGroups), int(qtype), int(zmul)) == 0, "PPU prepass failed");
      return {scale, zero};
    }
    // ZMul IS A TEMPLATE PARAMETER of the shared arithmetic, so the runtime value is dispatched. Only the two that
    // exist in this tree are accepted. WITHOUT THIS CHECK the dispatch below silently treats anything that is not 8
    // as 0 -- I dropped it once while refactoring, and a caller passing 4 would have got a plane that is wrong by
    // 4*scale everywhere and looks entirely plausible, which is the exact failure this parameter exists to prevent.
    TORCH_CHECK(zmul == 0 || zmul == 8, "zmul must be 0 or 8; got ", zmul,
                " -- it is the consumer's converter shift, not a free parameter");

    gguf_scale::prepass::BlockDesc src{blocks, dv.data(), has_dmin ? dmv.data() : nullptr,
                                       int64_t(Tr::kBlockBytes), 0, 1, 0};
    gguf_scale::prepass::PlaneDesc dst{sv.data(), zv.data(), int64_t(Tr::kGroups), 1};
    if (zmul == 8) gguf_scale::prepass::prepass_host<T, 8>(src, dst, int(rows), 1);
    else           gguf_scale::prepass::prepass_host<T, 0>(src, dst, int(rows), 1);

    auto* sp = get_ptr<at::Half>(scale);
    auto* zp = get_ptr<at::Half>(zero);
    for (size_t i = 0; i < sv.size(); ++i) {
      std::memcpy(sp + i, &sv[i], sizeof(cutlass::half_t));
      std::memcpy(zp + i, &zv[i], sizeof(cutlass::half_t));
    }
    return {scale, zero};
  });
}

// PURE CUDA-CORE DECODE, one dot product per RAW GGUF block. Exposed for the same reason the pre-pass is: the only
// oracle worth having is the official gguf package, and reaching it means reaching Python.
//
// THIS VALIDATES MORE THAN THE PRE-PASS TESTS CAN. Those compare per-group scalars, so a decoder with the right
// scales and the wrong element ORDER passes every one of them. A dot product against the reference's own weights
// cannot be fooled that way -- reorder anything and the sum moves.
//
// blocks : uint8 [rows, type_size]   RAW GGUF blocks, exactly as they sit in the file. No repacking, which is the
//                                    point: at decode the checkpoint's own bytes are what is resident.
// x      : fp32  [rows, 256]         the activation slice this superblock multiplies.
torch::Tensor gguf_vecdot(torch::Tensor blocks, torch::Tensor x, int64_t qtype) {
  CHECK_CPU(blocks); CHECK_CONTIGUOUS(blocks);
  CHECK_CPU(x); CHECK_CONTIGUOUS(x);
  TORCH_CHECK(blocks.dtype() == torch::kUInt8 && blocks.dim() == 2, "blocks must be uint8 [rows, type_size]");
  TORCH_CHECK(x.dtype() == torch::kFloat32 && x.dim() == 2, "x must be float32 [rows, 256]");
  TORCH_CHECK(x.size(1) == 256, "a k-quant superblock is 256 elements; x's second dim is ", x.size(1));
  TORCH_CHECK(x.size(0) == blocks.size(0), "blocks and x must have the same number of rows");
  int64_t const rows = blocks.size(0);
  int64_t const ts = blocks.size(1);
  torch::Tensor out = torch::empty({rows}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto const* bp = get_ptr<uint8_t const>(blocks);
  auto const* xp = get_ptr<float const>(x);
  auto* op = get_ptr<float>(out);
  // FORWARD TO THE DEVICE LIBRARY WHEN IT IS LOADED. The CPU loop below stays as the oracle and as the path on a
  // machine with no SDK; it is not a fallback that hides a failure, because gguf_backend() reports which one ran.
  if (auto const* api = ppu_backend::load()) {
    // rows blocks, one activation slice of 256 each -- the same shape the CPU arm below consumes, so the two are
    // interchangeable and the golden tests apply to whichever one ran.
    TORCH_CHECK(api->vecdot(bp, ts, xp, op, int(rows), 1, int(qtype)) == 0, "PPU vecdot failed");
    return out;
  }
  // THE CPU ARM OF AN INFERENCE OP IS A REFERENCE, NOT A FALLBACK, and the difference has to be enforced rather
  // than documented. It is a serial host loop -- no kernel, single-threaded -- and it exists to be the oracle the
  // official gguf package is compared against and to define the arithmetic the device kernels share. Silently
  // running it for a real GEMV would produce correct numbers at an unusable rate and report nothing, which is the
  // same class of failure as the seam not forwarding.
  //
  // Offline ops are the opposite: gguf_unpack and the layout packers are MEANT to run here, on an ordinary machine
  // preparing a checkpoint. So the refusal is on this op and not on those.
  if (rows > kCpuReferenceRowLimit && !cpu_reference_allowed()) {
    TORCH_CHECK(false,
                "gguf_vecdot has no device backend (", ppu_backend::resolved_backend(),
                ") and ", rows, " rows exceeds the ", kCpuReferenceRowLimit,
                "-row reference limit. The CPU arm is a serial host loop kept as the oracle for the official gguf "
                "package, not an inference path. Build libquactlize_ppu.so, or set QUACTLIZE_ALLOW_CPU_REFERENCE=1 "
                "to run it anyway.");
  }
  return dispatch_ktype(qtype, [&](auto tag) -> torch::Tensor {
    constexpr KType T = decltype(tag)::value;
    // THE RAW BLOCK SIZE IS THE GGUF ONE, not Traits::kBlockBytes -- that is the SCALE block and is a different
    // number. Confusing them slices the wrong bytes and fails as if the decode were wrong, so it is checked.
    constexpr int64_t kRaw = (T == KType::Q2_K) ? 84 : (T == KType::Q3_K) ? 110
                           : (T == KType::Q4_K) ? 144 : (T == KType::Q5_K) ? 176 : 210;
    TORCH_CHECK(ts == kRaw, "this format's raw GGUF block is ", kRaw, " bytes, got ", ts);
    for (int64_t r = 0; r < rows; ++r) op[r] = gguf_scale::vecdot::vecdot_block<T>(bp + r * ts, xp + r * 256);
    return out;
  });
}

// THE FALLBACK PATH'S MISSING LINK: raw GGUF blocks -> full fp16 weights, which is what cuBLAS and DeepGemm
// multiply. dequantize_weight in unfused_weight_dequantize.hpp already covers the symmetric packed representations
// (INT8/INT4/INT2/INT1 with fp16 scale planes); it cannot read a k-quant block, so a GGUF checkpoint had no route to
// the library GEMMs at all. This is that route, and it reuses the SAME traversal vecdot uses, so there is one
// transcription of each format's bit layout rather than two.
//
// NO ZMul. That is the mixed-input converter's centre correction; these are the actual weight values.
torch::Tensor gguf_dequantize(torch::Tensor blocks, int64_t qtype) {
  CHECK_CPU(blocks); CHECK_CONTIGUOUS(blocks);
  TORCH_CHECK(blocks.dtype() == torch::kUInt8 && blocks.dim() == 2, "blocks must be uint8 [rows, type_size]");
  int64_t const rows = blocks.size(0), ts = blocks.size(1);
  torch::Tensor out = torch::empty({rows, 256}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  auto const* bp = get_ptr<uint8_t const>(blocks);
  return dispatch_ktype(qtype, [&](auto tag) -> torch::Tensor {
    constexpr KType T = decltype(tag)::value;
    constexpr int64_t kRaw = (T == KType::Q2_K) ? 84 : (T == KType::Q3_K) ? 110
                           : (T == KType::Q4_K) ? 144 : (T == KType::Q5_K) ? 176 : 210;
    TORCH_CHECK(ts == kRaw, "this format's raw GGUF block is ", kRaw, " bytes, got ", ts);
    // The same bit-preserving copy the pre-pass op uses: at::Half and cutlass::half_t are distinct types and reading
    // one through the other is UB whatever the bits do.
    std::vector<cutlass::half_t> tmp(256);
    auto* op = get_ptr<at::Half>(out);
    for (int64_t r = 0; r < rows; ++r) {
      gguf_scale::vecdot::dequantize_block<T>(bp + r * ts, tmp.data());
      for (int j = 0; j < 256; ++j) std::memcpy(op + r * 256 + j, &tmp[size_t(j)], sizeof(cutlass::half_t));
    }
    return out;
  });
}

// CODES, SCALE AND ZERO AS THREE TENSORS -- the triple every offline packer in this tree already takes. This is
// what lets a GGUF checkpoint reach the existing kernels: they consume a packed low-bit weight plus fp16 planes, and
// nothing turned a k-quant block into that. With this, the chain is
//     raw GGUF -> unpack -> pack_int4 -> preprocess_weights_to_layout -> the kernel's own arrangement
// entirely through ops that are already validated, instead of a new packer to be wrong in a new way.
//
// The split obeys W = code * scale + zero, so reconstructing and comparing to the reference is a test that can fail.
std::vector<torch::Tensor> gguf_unpack(torch::Tensor blocks, int64_t qtype) {
  CHECK_CPU(blocks); CHECK_CONTIGUOUS(blocks);
  TORCH_CHECK(blocks.dtype() == torch::kUInt8 && blocks.dim() == 2, "blocks must be uint8 [rows, type_size]");
  int64_t const rows = blocks.size(0), ts = blocks.size(1);
  auto const* bp = get_ptr<uint8_t const>(blocks);
  return dispatch_ktype(qtype, [&](auto tag) -> std::vector<torch::Tensor> {
    constexpr KType T = decltype(tag)::value;
    using Tr = Traits<T>;
    constexpr int64_t kRaw = (T == KType::Q2_K) ? 84 : (T == KType::Q3_K) ? 110
                           : (T == KType::Q4_K) ? 144 : (T == KType::Q5_K) ? 176 : 210;
    TORCH_CHECK(ts == kRaw, "this format's raw GGUF block is ", kRaw, " bytes, got ", ts);
    auto i8 = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
    auto f16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
    torch::Tensor codes = torch::empty({rows, 256}, i8);
    torch::Tensor scale = torch::empty({rows, int64_t(Tr::kGroups)}, f16);
    torch::Tensor zero = torch::empty({rows, int64_t(Tr::kGroups)}, f16);
    auto* cp = get_ptr<int8_t>(codes);
    auto* sp = get_ptr<at::Half>(scale);
    auto* zp = get_ptr<at::Half>(zero);
    std::vector<cutlass::half_t> ts_(Tr::kGroups), tz_(Tr::kGroups);
    for (int64_t r = 0; r < rows; ++r) {
      gguf_scale::vecdot::unpack_block<T>(bp + r * ts, cp + r * 256, ts_.data(), tz_.data());
      for (int g = 0; g < Tr::kGroups; ++g) {
        std::memcpy(sp + r * Tr::kGroups + g, &ts_[size_t(g)], sizeof(cutlass::half_t));
        std::memcpy(zp + r * Tr::kGroups + g, &tz_[size_t(g)], sizeof(cutlass::half_t));
      }
    }
    return {codes, scale, zero};
  });
}

// WHICH BACKEND THE OPS WILL USE, as a value rather than something inferred from a timing. A device path that
// silently falls back to the CPU produces correct numbers slowly and says nothing, which looks exactly like the
// device path working -- so this is queryable and the tests assert on it.
std::string gguf_backend() {
  std::string why;
  ppu_backend::load(&why);
  return ppu_backend::resolved_backend() + " (" + why + ")";
}

// THE PACKED UNIT, BOTH DIRECTIONS. The packed in-kernel path reads a REORDERED scale unit rather than GGUF's own
// bytes, because GGUF's packing is not half-separable -- Q4_K's get_scale_min_k4 takes groups 4..7 from bytes 8-11
// AND the top two bits of bytes 0-3, so a k-tile covering half a superblock could not read half a block. The unit
// fixes that at no cost in stored bytes, which is the licence for the whole path and is asserted per format.
//
// Exposed so the round trip is a CI test rather than a scratch harness: pack a GGUF scale block into the unit,
// decode it back, and compare against the same decode taken from the GGUF block. Bit-exact for all five, which is
// the strongest statement available -- not "within tolerance".
std::vector<torch::Tensor> gguf_pack_unit(torch::Tensor scale_blocks, torch::Tensor d, torch::Tensor dmin,
                                          int64_t qtype) {
  CHECK_CPU(scale_blocks); CHECK_CONTIGUOUS(scale_blocks);
  TORCH_CHECK(scale_blocks.dtype() == torch::kUInt8 && scale_blocks.dim() == 2, "scale_blocks must be uint8 2-D");
  TORCH_CHECK(d.dtype() == torch::kFloat16, "d must be float16");
  int64_t const rows = scale_blocks.size(0);
  bool const has_dmin = dmin.defined() && dmin.numel() > 0;
  auto const* bp = get_ptr<uint8_t const>(scale_blocks);
  return dispatch_ktype(qtype, [&](auto tag) -> std::vector<torch::Tensor> {
    constexpr KType T = decltype(tag)::value;
    using U = gguf_scale::packed_unit::Unit<T>;
    TORCH_CHECK(scale_blocks.size(1) == Traits<T>::kBlockBytes, "scale_blocks second dim must be the SCALE block");
    TORCH_CHECK(!U::kHasMin || has_dmin, "this format has a min channel, so dmin is required");
    torch::Tensor units = torch::empty({rows, int64_t(U::kUnitBytes)},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto* up = get_ptr<uint8_t>(units);
    auto const* dp = get_ptr<at::Half const>(d);
    auto const* mp = has_dmin ? get_ptr<at::Half const>(dmin) : nullptr;
    for (int64_t r = 0; r < rows; ++r) {
      cutlass::half_t dd, dm{0.f};
      std::memcpy(&dd, dp + r, sizeof(dd));
      if (mp) std::memcpy(&dm, mp + r, sizeof(dm));
      gguf_scale::packed_unit::pack_unit<T>(bp + r * Traits<T>::kBlockBytes, dd, dm, up + r * U::kUnitBytes);
    }
    return {units};
  });
}

// The decode side, so the round trip closes through Python rather than through a header nobody calls from a test.
std::vector<torch::Tensor> gguf_unit_decode(torch::Tensor units, int64_t qtype, int64_t zmul) {
  CHECK_CPU(units); CHECK_CONTIGUOUS(units);
  TORCH_CHECK(units.dtype() == torch::kUInt8 && units.dim() == 2, "units must be uint8 2-D");
  TORCH_CHECK(zmul == 0 || zmul == 8, "zmul must be 0 or 8; it is the consumer's converter shift");
  int64_t const rows = units.size(0);
  auto const* up = get_ptr<uint8_t const>(units);
  return dispatch_ktype(qtype, [&](auto tag) -> std::vector<torch::Tensor> {
    constexpr KType T = decltype(tag)::value;
    using U = gguf_scale::packed_unit::Unit<T>;
    TORCH_CHECK(units.size(1) == U::kUnitBytes, "this format's unit is ", U::kUnitBytes, " bytes, got ",
                units.size(1));
    auto f16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
    torch::Tensor scale = torch::empty({rows, int64_t(U::kGroups)}, f16);
    torch::Tensor zero = torch::empty({rows, int64_t(U::kGroups)}, f16);
    auto* sp = get_ptr<at::Half>(scale);
    auto* zp = get_ptr<at::Half>(zero);
    for (int64_t r = 0; r < rows; ++r) {
      for (int g = 0; g < U::kGroups; ++g) {
        auto sz = (zmul == 8) ? gguf_scale::packed_unit::unit_group<T, 8>(up + r * U::kUnitBytes, g)
                              : gguf_scale::packed_unit::unit_group<T, 0>(up + r * U::kUnitBytes, g);
        std::memcpy(sp + r * U::kGroups + g, &sz.scale, sizeof(sz.scale));
        std::memcpy(zp + r * U::kGroups + g, &sz.zero, sizeof(sz.zero));
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

static auto gguf_backend_op = torch::RegisterOperators("quactlize::gguf_backend", &torch_ext::gguf_backend);

static auto gguf_pack_unit_op = torch::RegisterOperators("quactlize::gguf_pack_unit", &torch_ext::gguf_pack_unit);
static auto gguf_unit_decode_op = torch::RegisterOperators("quactlize::gguf_unit_decode", &torch_ext::gguf_unit_decode);

static auto gguf_unpack_op = torch::RegisterOperators("quactlize::gguf_unpack", &torch_ext::gguf_unpack);

static auto gguf_dequantize_op =
    torch::RegisterOperators("quactlize::gguf_dequantize", &torch_ext::gguf_dequantize);

static auto gguf_vecdot_op = torch::RegisterOperators("quactlize::gguf_vecdot", &torch_ext::gguf_vecdot);

static auto gguf_scale_block_shape_op =
    torch::RegisterOperators("quactlize::gguf_scale_block_shape", &torch_ext::gguf_scale_block_shape);
