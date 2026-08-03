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
#include <algorithm>
#include <vector>

#include "gguf_scale_prepass.hpp"
#include "gguf_vecdot.hpp"
#include "gguf_packed_unit.hpp"
#include "weight_layout.h"
#include "thop/th_utils.h"
#include "thop/ppu_backend.h"

namespace torch_ext {

namespace {

using gguf_scale::KType;
using gguf_scale::Traits;
using cutlass::half_t;

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

// Raw GGUF locations needed by the fully-quantized producers. Field placement inside the REORDERED
// unit remains owned by gguf_packed_unit's cute layouts; these traits only name the byte-aligned record slices the
// official formats publish. Adding two scattered offset ladders to dense and grouped would recreate the relation.
template <KType T> struct PackedRaw;
template <> struct PackedRaw<KType::Q4_K> {
  static constexpr int kRawBytes=144, kScaleOffset=4, kDOffset=0, kDminOffset=2;
};
template <> struct PackedRaw<KType::Q2_K> {
  static constexpr int kRawBytes=84, kScaleOffset=0, kDOffset=80, kDminOffset=82;
};
template <> struct PackedRaw<KType::Q5_K> {
  static constexpr int kRawBytes=176, kScaleOffset=4, kDOffset=0, kDminOffset=2;
};

template <KType T>
void pack_raw_unit(uint8_t const* block, uint8_t* unit) {
  using R = PackedRaw<T>;
  using U = gguf_scale::packed_unit::Unit<T>;
  static_assert(U::kSbPerUnit == 1, "this producer currently emits one-superblock units");
  cutlass::half_t d, dmin;
  std::memcpy(&d, block + R::kDOffset, sizeof(d));
  std::memcpy(&dmin, block + R::kDminOffset, sizeof(dmin));
  gguf_scale::packed_unit::pack_unit_sb<T>(block + R::kScaleOffset, d, dmin, 0, unit);
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
// x      : fp16/fp32 [rows, 256]      the activation slice this superblock multiplies. The device ABI is fp16;
//                                     fp32 remains accepted by the CPU oracle and is converted before forwarding.
torch::Tensor gguf_vecdot(torch::Tensor blocks, torch::Tensor x, int64_t qtype) {
  CHECK_CPU(blocks); CHECK_CONTIGUOUS(blocks);
  CHECK_CPU(x); CHECK_CONTIGUOUS(x);
  TORCH_CHECK(blocks.dtype() == torch::kUInt8 && blocks.dim() == 2, "blocks must be uint8 [rows, type_size]");
  TORCH_CHECK((x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32) && x.dim() == 2,
              "x must be float16 or float32 [rows, 256]");
  TORCH_CHECK(x.size(1) == 256, "a k-quant superblock is 256 elements; x's second dim is ", x.size(1));
  TORCH_CHECK(x.size(0) == blocks.size(0), "blocks and x must have the same number of rows");
  int64_t const rows = blocks.size(0);
  int64_t const ts = blocks.size(1);
  torch::Tensor out = torch::empty({rows}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto const* bp = get_ptr<uint8_t const>(blocks);
  auto* op = get_ptr<float>(out);
  // FORWARD TO THE DEVICE LIBRARY WHEN IT IS LOADED. The CPU loop below stays as the oracle and as the path on a
  // machine with no SDK; it is not a fallback that hides a failure, because gguf_backend() reports which one ran.
  if (auto const* api = ppu_backend::load()) {
    torch::Tensor x16 = x.dtype() == torch::kFloat16 ? x : x.to(torch::kFloat16);
    auto const* xp16 = get_ptr<uint16_t const>(x16);
    // Each raw block row is paired with its own activation slice, exactly as in the CPU loop below. This is a
    // different ABI from dense GEMV, whose output rows all share one activation; conflating them made row r consume
    // x[0] instead of x[r] and produced order-one disagreement on otherwise identical bytes.
    TORCH_CHECK(api->vecdot(bp, ts, xp16, op, int(rows), 1, int(qtype)) == 0, "PPU vecdot failed");
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
    if (x.dtype() == torch::kFloat32) {
      auto const* xp = get_ptr<float const>(x);
      for (int64_t r = 0; r < rows; ++r)
        op[r] = gguf_scale::vecdot::vecdot_block<T>(bp + r * ts, xp + r * 256);
    } else {
      auto const* xp = get_ptr<half_t const>(x);
      for (int64_t r = 0; r < rows; ++r)
        op[r] = gguf_scale::vecdot::vecdot_block_input<T>(bp + r * ts, xp + r * 256);
    }
    return out;
  });
}

// Native dense GEMV is not the per-block gguf_vecdot operation above. It consumes ONE complete activation and
// shares it across n output columns while accumulating bpr superblocks inside the kernel. Giving this shape its own
// op prevents the host reference's [block,x-slice] interface from being mistaken for the production launch ABI.
torch::Tensor gguf_vecdot_dense(torch::Tensor blocks, torch::Tensor x, int64_t n, int64_t k, int64_t qtype) {
  CHECK_CPU(blocks); CHECK_CONTIGUOUS(blocks); CHECK_CPU(x); CHECK_CONTIGUOUS(x);
  TORCH_CHECK(blocks.dtype() == torch::kUInt8 && blocks.dim() == 2,
              "blocks must be uint8 [n*k/256,type_size]");
  TORCH_CHECK((x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32) && x.dim() == 2 &&
              x.size(0) == 1 && x.size(1) == k, "x must be fp16/fp32 [1,k]");
  TORCH_CHECK(n > 0 && k > 0 && k % 256 == 0, "n must be positive and k a positive multiple of 256");
  int const bpr = int(k / 256);
  TORCH_CHECK(blocks.size(0) == n * bpr, "blocks need n*k/256 rows; got ", blocks.size(0));
  auto out = torch::empty({1, n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  return dispatch_ktype(qtype, [&](auto tag) -> torch::Tensor {
    constexpr KType T = decltype(tag)::value;
    constexpr int64_t kRaw = (T == KType::Q2_K) ? 84 : (T == KType::Q3_K) ? 110
                           : (T == KType::Q4_K) ? 144 : (T == KType::Q5_K) ? 176 : 210;
    TORCH_CHECK(blocks.size(1) == kRaw, "this format's raw GGUF block is ", kRaw, " bytes, got ", blocks.size(1));
    auto const* bp = get_ptr<uint8_t const>(blocks);
    auto* op = get_ptr<float>(out);
    if (auto const* api = ppu_backend::load()) {
      TORCH_CHECK(api->vecdot_dense,
                  "loaded PPU backend predates the separate dense-vecdot ABI; rebuild libquactlize_ppu.so");
      torch::Tensor x16 = x.dtype() == torch::kFloat16 ? x : x.to(torch::kFloat16);
      TORCH_CHECK(api->vecdot_dense(bp, kRaw, get_ptr<uint16_t const>(x16), op, int(n), bpr, int(qtype)) == 0,
                  "PPU native dense GEMV failed");
      return out;
    }
    TORCH_CHECK(n <= kCpuReferenceRowLimit || cpu_reference_allowed(),
                "gguf_vecdot_dense has no device backend (", ppu_backend::resolved_backend(), ") and ", n,
                " output rows exceeds the ", kCpuReferenceRowLimit, "-row CPU reference limit");
    if (x.dtype() == torch::kFloat32) {
      auto const* xp = get_ptr<float const>(x);
      for (int64_t r = 0; r < n; ++r) {
        float acc = 0.f;
        for (int b = 0; b < bpr; ++b)
          acc += gguf_scale::vecdot::vecdot_block<T>(bp + (r * bpr + b) * kRaw, xp + b * 256);
        op[r] = acc;
      }
    } else {
      auto const* xp = get_ptr<half_t const>(x);
      for (int64_t r = 0; r < n; ++r) {
        float acc = 0.f;
        for (int b = 0; b < bpr; ++b)
          acc += gguf_scale::vecdot::vecdot_block_input<T>(bp + (r * bpr + b) * kRaw, xp + b * 256);
        op[r] = acc;
      }
    }
    return out;
  });
}

// Native-scale MoE decode through the real device library. Unlike gguf_vecdot's small CPU reference arm, this
// operation has no host fallback: its purpose is the expert/routing launch, and a serial loop over experts would be
// a different kernel that could silently approve a missing backend.
torch::Tensor gguf_vecdot_moe(torch::Tensor blocks, torch::Tensor x, torch::Tensor row_offsets, int64_t qtype) {
  CHECK_CPU(blocks); CHECK_CONTIGUOUS(blocks); CHECK_CPU(x); CHECK_CONTIGUOUS(x);
  CHECK_CPU(row_offsets); CHECK_CONTIGUOUS(row_offsets);
  TORCH_CHECK(blocks.dtype() == torch::kUInt8 && blocks.dim() == 3,
              "blocks must be uint8 [experts,n*blocks_per_row,type_size]");
  TORCH_CHECK((x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32) && x.dim() == 2,
              "x must be fp16/fp32 [total_rows,k]");
  TORCH_CHECK(row_offsets.dtype() == torch::kInt32 && row_offsets.dim() == 1,
              "row_offsets must be int32 [experts+1]");
  int const experts = int(blocks.size(0));
  TORCH_CHECK(row_offsets.numel() == experts + 1, "row_offsets needs experts+1 entries");
  int const* off = get_ptr<int const>(row_offsets);
  TORCH_CHECK(off[0] == 0, "row_offsets must start at zero");
  int max_rows = 0;
  for (int e = 0; e < experts; ++e) {
    TORCH_CHECK(off[e + 1] >= off[e], "row_offsets must be nondecreasing");
    max_rows = std::max(max_rows, off[e + 1] - off[e]);
  }
  int const total_rows = off[experts];
  TORCH_CHECK(total_rows == x.size(0), "row_offsets end at ", total_rows, " but x has ", x.size(0), " rows");
  int64_t const k = x.size(1);
  TORCH_CHECK(k > 0 && k % 256 == 0, "x's k must be a positive multiple of 256");
  int const bpr = int(k / 256);
  TORCH_CHECK(blocks.size(1) % bpr == 0, "expert block count must be divisible by k/256");
  int const n = int(blocks.size(1) / bpr);
  TORCH_CHECK(qtype >= kGgmlQ2K && qtype <= kGgmlQ6K, "unsupported GGUF qtype ", qtype);
  int const raw_bytes = qtype == kGgmlQ2K ? 84 : qtype == kGgmlQ3K ? 110 : qtype == kGgmlQ4K ? 144
                      : qtype == kGgmlQ5K ? 176 : 210;
  TORCH_CHECK(blocks.size(2) == raw_bytes, "this format's raw GGUF block is ", raw_bytes,
              " bytes, got ", blocks.size(2));
  torch::Tensor x16 = x.dtype() == torch::kFloat16 ? x : x.to(torch::kFloat16);
  torch::Tensor out = torch::empty({total_rows, n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto const* api = ppu_backend::load();
  TORCH_CHECK(api, "gguf_vecdot_moe requires libquactlize_ppu.so (", ppu_backend::resolved_backend(), ")");
  if (total_rows == 0) return out;
  TORCH_CHECK(api->vecdot_moe(get_ptr<uint8_t const>(blocks), blocks.size(2),
                              reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(x16)), off,
                              get_ptr<float>(out), n, bpr, experts, total_rows, max_rows, int(qtype)) == 0,
              "PPU native MoE GEMV failed");
  return out;
}

namespace {

torch::Tensor gguf_gemv_scale_first_impl(torch::Tensor a, torch::Tensor low, torch::Tensor high,
                                          torch::Tensor scale, torch::Tensor zero,
                                          torch::Tensor row_offsets, int64_t qtype, bool grouped) {
  CHECK_CPU(a); CHECK_CONTIGUOUS(a); CHECK_CPU(low); CHECK_CONTIGUOUS(low);
  CHECK_CPU(scale); CHECK_CONTIGUOUS(scale); CHECK_CPU(zero); CHECK_CONTIGUOUS(zero);
  TORCH_CHECK((a.dtype() == torch::kFloat16 || a.dtype() == torch::kFloat32) && a.dim() == 2,
              "a must be fp16/fp32 [rows,k]");
  TORCH_CHECK(low.dtype() == torch::kUInt8 && low.dim() == 3, "low must be uint8 [experts,n,bytes]");
  TORCH_CHECK(scale.dtype() == torch::kFloat16 && zero.dtype() == torch::kFloat16 &&
              scale.dim() == 3 && zero.sizes() == scale.sizes(),
              "scale and zero must be fp16 [experts,k/group_size,n]");
  int const lo_bits = (qtype == kGgmlQ2K || qtype == kGgmlQ3K) ? 2 : 4;
  int const hi_bits = (qtype == kGgmlQ3K || qtype == kGgmlQ5K) ? 1 : qtype == kGgmlQ6K ? 2 : 0;
  int const group_size = (qtype == kGgmlQ4K || qtype == kGgmlQ5K) ? 32 : 16;
  int const artifact_experts = int(low.size(0)), n = int(low.size(1));
  int64_t const k = a.size(1);
  TORCH_CHECK(k > 0 && k % 256 == 0, "a's k must be a positive multiple of 256");
  TORCH_CHECK(low.size(2) == k * lo_bits / 8, "low plane byte count disagrees with k/qtype");
  TORCH_CHECK(scale.size(0) == artifact_experts && scale.size(1) == k / group_size && scale.size(2) == n,
              "scale shape disagrees with weight artifact");
  uint8_t const* hp = nullptr;
  if (hi_bits) {
    CHECK_CPU(high); CHECK_CONTIGUOUS(high);
    TORCH_CHECK(high.dtype() == torch::kUInt8 && high.dim() == 3 && high.size(0) == artifact_experts &&
                high.size(1) == n && high.size(2) == k * hi_bits / 8,
                "high plane shape disagrees with qtype");
    hp = get_ptr<uint8_t const>(high);
  } else {
    TORCH_CHECK(!high.defined() || high.numel() == 0, "single-plane format must have an empty high plane");
  }

  int experts = 0, max_rows = int(a.size(0));
  int const* off = nullptr;
  if (grouped) {
    CHECK_CPU(row_offsets); CHECK_CONTIGUOUS(row_offsets);
    TORCH_CHECK(row_offsets.dtype() == torch::kInt32 && row_offsets.dim() == 1,
                "row_offsets must be int32 [experts+1]");
    experts = artifact_experts;
    TORCH_CHECK(row_offsets.numel() == experts + 1, "row_offsets needs experts+1 entries");
    off = get_ptr<int const>(row_offsets);
    TORCH_CHECK(off[0] == 0 && off[experts] == a.size(0), "row_offsets must span every activation row");
    max_rows = 0;
    for (int e = 0; e < experts; ++e) {
      TORCH_CHECK(off[e + 1] >= off[e], "row_offsets must be nondecreasing");
      max_rows = std::max(max_rows, off[e + 1] - off[e]);
    }
  } else {
    TORCH_CHECK(artifact_experts == 1, "dense artifact must have one expert dimension");
  }

  torch::Tensor a16 = a.dtype() == torch::kFloat16 ? a : a.to(torch::kFloat16);
  torch::Tensor out = torch::empty({a.size(0), n}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  auto const* api = ppu_backend::load();
  TORCH_CHECK(api, "scale-first GEMV requires libquactlize_ppu.so (", ppu_backend::resolved_backend(), ")");
  if (a.size(0) == 0) return out;
  TORCH_CHECK(api->gemv_lowbit(reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(a16)),
                               get_ptr<uint8_t const>(low), hp,
                               reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(scale)),
                               reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(zero)),
                               reinterpret_cast<uint16_t*>(get_ptr<at::Half>(out)), int(a.size(0)), n, int(k),
                               group_size, int(qtype), experts, off, max_rows) == 0,
              "PPU scale-first GEMV failed");
  return out;
}

}  // namespace

torch::Tensor gguf_gemv_scale_first(torch::Tensor a, torch::Tensor low, torch::Tensor high,
                                     torch::Tensor scale, torch::Tensor zero, int64_t qtype) {
  return gguf_gemv_scale_first_impl(a, low, high, scale, zero, torch::Tensor(), qtype, false);
}

torch::Tensor gguf_gemv_scale_first_moe(torch::Tensor a, torch::Tensor low, torch::Tensor high,
                                         torch::Tensor scale, torch::Tensor zero,
                                         torch::Tensor row_offsets, int64_t qtype) {
  return gguf_gemv_scale_first_impl(a, low, high, scale, zero, row_offsets, qtype, true);
}

torch::Tensor gguf_dense_scale_first(torch::Tensor a, torch::Tensor low, torch::Tensor high,
                                      torch::Tensor scale, torch::Tensor zero, int64_t qtype) {
  CHECK_CPU(a); CHECK_CONTIGUOUS(a); CHECK_CPU(low); CHECK_CONTIGUOUS(low);
  CHECK_CPU(scale); CHECK_CONTIGUOUS(scale); CHECK_CPU(zero); CHECK_CONTIGUOUS(zero);
  TORCH_CHECK((a.dtype() == torch::kFloat16 || a.dtype() == torch::kFloat32) && a.dim() == 2,
              "a must be fp16/fp32 [m,k]");
  TORCH_CHECK(low.dtype() == torch::kUInt8 && low.dim() == 3 && low.size(0) == 1,
              "low must be uint8 [1,n,bytes]");
  TORCH_CHECK(scale.dtype() == torch::kFloat16 && zero.dtype() == torch::kFloat16 &&
              scale.dim() == 3 && scale.size(0) == 1 && zero.sizes() == scale.sizes(),
              "scale and zero must be fp16 [1,k/group_size,n]");
  int const lo_bits = (qtype == kGgmlQ2K || qtype == kGgmlQ3K) ? 2 : 4;
  int const hi_bits = (qtype == kGgmlQ3K || qtype == kGgmlQ5K) ? 1 : qtype == kGgmlQ6K ? 2 : 0;
  int const group_size = (qtype == kGgmlQ4K || qtype == kGgmlQ5K) ? 32 : 16;
  int const m = int(a.size(0)), k = int(a.size(1)), n = int(low.size(1));
  TORCH_CHECK(m >= 0 && n > 0 && n % 256 == 0 && k > 0 && k % 256 == 0,
              "dense fpA shape needs n and k multiples of 256");
  TORCH_CHECK(low.size(2) == int64_t(k) * lo_bits / 8, "low plane byte count disagrees with k/qtype");
  TORCH_CHECK(scale.size(1) == k / group_size && scale.size(2) == n,
              "scale shape disagrees with dense artifact");
  uint8_t const* hp = nullptr;
  if (hi_bits) {
    CHECK_CPU(high); CHECK_CONTIGUOUS(high);
    TORCH_CHECK(high.dtype() == torch::kUInt8 && high.dim() == 3 && high.size(0) == 1 &&
                high.size(1) == n && high.size(2) == int64_t(k) * hi_bits / 8,
                "high plane shape disagrees with qtype");
    hp = get_ptr<uint8_t const>(high);
  } else {
    TORCH_CHECK(!high.defined() || high.numel() == 0, "single-plane format must have an empty high plane");
  }
  torch::Tensor a16 = a.dtype() == torch::kFloat16 ? a : a.to(torch::kFloat16);
  torch::Tensor out = torch::empty({m, n}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  if (m == 0) return out;
  auto const* api = ppu_backend::load();
  TORCH_CHECK(api && api->dense_lowbit,
              "scale-first dense requires an hgcc libquactlize_ppu.so with the fpA launcher symbol");
  TORCH_CHECK(api->dense_lowbit(reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(a16)),
                                get_ptr<uint8_t const>(low), hp,
                                reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(scale)),
                                reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(zero)),
                                reinterpret_cast<uint16_t*>(get_ptr<at::Half>(out)),
                                m, n, k, group_size, int(qtype)) == 0,
              "PPU scale-first dense GEMM failed");
  return out;
}

// FULLY_QUANTIZED x DENSE, format-selected k-quants. `low`, optional `high`, and `units` are resident artifacts: the
// code planes have crossed fixed xplane placement and units have crossed gguf_pack_unit's byte-neutral reorder. No
// decode happens here;
// its device entry is an explicit TileK=256 instantiation of the shared CollectiveBuilder packed-scale mainloop.
// Keeping the contract separate from gguf_dense_scale_first prevents a flagged binary from interpreting an fp16
// scale plane as raw unit bytes. The build-time PPU_PACKED_FORMAT selects which unit trait the device consumes.
torch::Tensor gguf_dense_fully_quantized(torch::Tensor a, torch::Tensor low, torch::Tensor high, torch::Tensor units,
                                          int64_t qtype) {
  CHECK_CPU(a); CHECK_CONTIGUOUS(a); CHECK_CPU(low); CHECK_CONTIGUOUS(low);
  CHECK_CPU(high); CHECK_CONTIGUOUS(high); CHECK_CPU(units); CHECK_CONTIGUOUS(units);
  auto const meta = dispatch_ktype(qtype, [](auto tag) -> std::vector<int64_t> {
    constexpr KType T = decltype(tag)::value;
    TORCH_CHECK(T == KType::Q4_K || T == KType::Q2_K || T == KType::Q5_K,
                "fully-quantized dense supports Q4_K, Q2_K and Q5_K in format-selected builds");
    using U = gguf_scale::packed_unit::Unit<T>;
    constexpr int kLoBits = T == KType::Q2_K ? 2 : 4;
    constexpr int kHiBits = T == KType::Q5_K ? 1 : 0;
    return {kLoBits, kHiBits, U::kUnitTotal};
  });
  TORCH_CHECK((a.dtype() == torch::kFloat16 || a.dtype() == torch::kFloat32) && a.dim() == 2,
              "a must be fp16/fp32 [m,k]");
  TORCH_CHECK(low.dtype() == torch::kUInt8 && low.dim() == 3 && low.size(0) == 1,
              "low must be the uint8 dense artifact [1,n,k/2]");
  TORCH_CHECK(units.dtype() == torch::kUInt8 && units.dim() == 3 && units.size(2) == meta[2],
              "packed units must be uint8 [k/256,n,", meta[2], "] for this format");
  int const m = int(a.size(0)), k = int(a.size(1)), n = int(low.size(1));
  TORCH_CHECK(m >= 0 && n > 0 && n % 256 == 0 && k > 0 && k % 256 == 0,
              "fully-quantized dense needs n and k multiples of 256");
  TORCH_CHECK(low.size(2) == int64_t(k) * meta[0] / 8, "low artifact byte count disagrees with k/qtype");
  if (meta[1]) {
    TORCH_CHECK(high.dtype() == torch::kUInt8 && high.dim() == 3 && high.size(0) == 1 &&
                high.size(1) == n && high.size(2) == int64_t(k) * meta[1] / 8,
                "high artifact byte count disagrees with k/qtype");
  } else {
    TORCH_CHECK(high.numel() == 0, "single-plane format requires an empty high tensor");
  }
  TORCH_CHECK(units.size(0) == k / 256 && units.size(1) == n,
              "packed unit shape disagrees with the dense artifact");

  torch::Tensor a16 = a.dtype() == torch::kFloat16 ? a : a.to(torch::kFloat16);
  torch::Tensor out = torch::empty({m, n}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  if (m == 0) return out;
  auto const* api = ppu_backend::load();
  TORCH_CHECK(api && api->dense_fully_quantized,
              "fully-quantized dense requires a current hgcc libquactlize_ppu.so");
  int const rc = api->dense_fully_quantized(
      reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(a16)), get_ptr<uint8_t const>(low),
      high.numel() ? get_ptr<uint8_t const>(high) : nullptr, get_ptr<uint8_t const>(units),
      reinterpret_cast<uint16_t*>(get_ptr<at::Half>(out)),
      m, n, k, int(qtype));
  TORCH_CHECK(rc == 0, "PPU fully-quantized dense GEMM failed (rc=", rc,
              "; rc=34 means this library is not built with PPU_PACKED_SCALE=1)");
  return out;
}

torch::Tensor gguf_grouped_fully_quantized(torch::Tensor a, torch::Tensor low, torch::Tensor high, torch::Tensor units,
                                            torch::Tensor rows_per_expert, int64_t qtype) {
  CHECK_CPU(a); CHECK_CONTIGUOUS(a); CHECK_CPU(low); CHECK_CONTIGUOUS(low);
  CHECK_CPU(high); CHECK_CONTIGUOUS(high); CHECK_CPU(units); CHECK_CONTIGUOUS(units);
  CHECK_CPU(rows_per_expert); CHECK_CONTIGUOUS(rows_per_expert);
  auto const meta = dispatch_ktype(qtype, [](auto tag) -> std::vector<int64_t> {
    constexpr KType T = decltype(tag)::value;
    TORCH_CHECK(T == KType::Q4_K || T == KType::Q2_K || T == KType::Q5_K,
                "fully-quantized grouped supports Q4_K, Q2_K and Q5_K in format-selected builds");
    using U = gguf_scale::packed_unit::Unit<T>;
    constexpr int kLoBits = T == KType::Q2_K ? 2 : 4;
    constexpr int kHiBits = T == KType::Q5_K ? 1 : 0;
    return {kLoBits, kHiBits, U::kUnitTotal};
  });
  TORCH_CHECK((a.dtype() == torch::kFloat16 || a.dtype() == torch::kFloat32) && a.dim() == 2,
              "a must be fp16/fp32 concatenated [total_rows,k]");
  TORCH_CHECK(low.dtype() == torch::kUInt8 && low.dim() == 3,
              "low must be the uint8 grouped artifact [experts,n,k/2]");
  TORCH_CHECK(units.dtype() == torch::kUInt8 && units.dim() == 4 && units.size(3) == meta[2],
              "grouped units must be uint8 [experts,k/256,n,", meta[2], "] for this format");
  TORCH_CHECK(rows_per_expert.dtype() == torch::kInt32 && rows_per_expert.dim() == 1,
              "rows_per_expert must be contiguous int32 [experts]");
  int const experts = int(low.size(0)), n = int(low.size(1)), k = int(a.size(1));
  TORCH_CHECK(experts > 0 && rows_per_expert.size(0) == experts && units.size(0) == experts,
              "artifact and rows_per_expert must agree on experts");
  TORCH_CHECK(n > 0 && n % 256 == 0 && k > 0 && k % 256 == 0 &&
              low.size(2) == int64_t(k) * meta[0] / 8,
              "grouped artifact needs n/k multiples of 256 and the format's low-byte count");
  if (meta[1]) {
    TORCH_CHECK(high.dtype() == torch::kUInt8 && high.dim() == 3 && high.size(0) == experts &&
                high.size(1) == n && high.size(2) == int64_t(k) * meta[1] / 8,
                "grouped high artifact byte count disagrees with k/qtype");
  } else {
    TORCH_CHECK(high.numel() == 0, "single-plane grouped format requires an empty high tensor");
  }
  TORCH_CHECK(units.size(1) == k / 256 && units.size(2) == n,
              "packed grouped unit shape disagrees with the weight artifact");
  int const* rp = get_ptr<int const>(rows_per_expert);
  int64_t total = 0;
  for (int e = 0; e < experts; ++e) { TORCH_CHECK(rp[e] >= 0, "expert row counts must be nonnegative"); total += rp[e]; }
  TORCH_CHECK(total == a.size(0), "rows_per_expert sums to ", total, " but a has ", a.size(0), " rows");

  torch::Tensor a16 = a.dtype() == torch::kFloat16 ? a : a.to(torch::kFloat16);
  torch::Tensor out = torch::empty({total, n}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  if (total == 0) return out;
  auto const* api = ppu_backend::load();
  TORCH_CHECK(api && api->grouped_fully_quantized,
              "fully-quantized grouped requires a current hgcc libquactlize_ppu.so");
  int const rc = api->grouped_fully_quantized(
      reinterpret_cast<uint16_t const*>(get_ptr<at::Half const>(a16)), get_ptr<uint8_t const>(low),
      high.numel() ? get_ptr<uint8_t const>(high) : nullptr, get_ptr<uint8_t const>(units), rp,
      reinterpret_cast<uint16_t*>(get_ptr<at::Half>(out)),
      int(total), n, k, experts, int(qtype));
  TORCH_CHECK(rc == 0, "PPU fully-quantized grouped GEMM failed (rc=", rc,
              "; rc=34 means this library is not built with PPU_PACKED_SCALE=1)");
  return out;
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

// OFFLINE ARTIFACT FOR gemv_lowbit, all five k-quants. The raw checkpoint stays the independent source of truth;
// this function only changes representation:
//
//   raw [E,N,K/256,type_size]
//       -> low/high code planes in GEMV's affine Native [E,N,K] bit layout
//       -> scale/zero planes [E,K/group_size,N]
//
// Q3_K and Q6_K expose signed logical codes from unpack_block. gemv_lowbit combines unsigned bit planes, so those
// two use offset binary (c + 4 / c + 32) and subtract the same bias*scale from zero. This preserves
// c*scale+zero exactly up to the fp16 plane's own rounding. No plane grows: the output is exactly 2/3/4/5/6 bits
// per weight plus the explicitly requested fp16 scale workspace.
std::vector<torch::Tensor> gguf_prepare_gemv(torch::Tensor blocks, int64_t n, int64_t k, int64_t qtype) {
  CHECK_CPU(blocks); CHECK_CONTIGUOUS(blocks);
  TORCH_CHECK(blocks.dtype() == torch::kUInt8 && (blocks.dim() == 2 || blocks.dim() == 3),
              "blocks must be uint8 [N*K/256,type_size] or [E,N*K/256,type_size]");
  TORCH_CHECK(n > 0 && k > 0 && k % 256 == 0, "n must be positive and k a positive multiple of 256");
  int64_t const experts = blocks.dim() == 3 ? blocks.size(0) : 1;
  int64_t const blocks_per_expert = blocks.dim() == 3 ? blocks.size(1) : blocks.size(0);
  int64_t const bpr = k / 256;
  TORCH_CHECK(blocks_per_expert == n * bpr, "each expert needs ", n * bpr,
              " raw blocks for n=", n, " k=", k, "; got ", blocks_per_expert);
  int64_t const raw_bytes = blocks.size(-1);
  auto const* src = get_ptr<uint8_t const>(blocks);

  return dispatch_ktype(qtype, [&](auto tag) -> std::vector<torch::Tensor> {
    constexpr KType T = decltype(tag)::value;
    using Tr = Traits<T>;
    constexpr int kRaw = (T == KType::Q2_K) ? 84 : (T == KType::Q3_K) ? 110
                         : (T == KType::Q4_K) ? 144 : (T == KType::Q5_K) ? 176 : 210;
    constexpr int kLoBits = (T == KType::Q2_K || T == KType::Q3_K) ? 2 : 4;
    constexpr int kHiBits = (T == KType::Q3_K || T == KType::Q5_K) ? 1
                          : (T == KType::Q6_K) ? 2 : 0;
    constexpr int kCodeBits = kLoBits + kHiBits;
    constexpr int kCodeBias = T == KType::Q3_K ? 4 : T == KType::Q6_K ? 32 : 0;
    TORCH_CHECK(raw_bytes == kRaw, "this format's raw GGUF block is ", kRaw, " bytes, got ", raw_bytes);

    auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto f16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
    torch::Tensor low = torch::zeros({experts, n, k * kLoBits / 8}, u8);
    torch::Tensor high = kHiBits ? torch::zeros({experts, n, k * kHiBits / 8}, u8)
                                 : torch::empty({0}, u8);
    int64_t const scale_k = k / Tr::kGroupSize;
    torch::Tensor scale = torch::empty({experts, scale_k, n}, f16);
    torch::Tensor zero = torch::empty_like(scale);
    auto* lp = get_ptr<uint8_t>(low);
    auto* hp = kHiBits ? get_ptr<uint8_t>(high) : nullptr;
    auto* sp = get_ptr<at::Half>(scale);
    auto* zp = get_ptr<at::Half>(zero);
    std::vector<int8_t> codes(256);
    std::vector<cutlass::half_t> ss(Tr::kGroups), zz(Tr::kGroups);

    for (int64_t e = 0; e < experts; ++e) {
      for (int64_t col = 0; col < n; ++col) {
        for (int64_t b = 0; b < bpr; ++b) {
          int64_t const raw_row = (e * blocks_per_expert + col * bpr + b);
          gguf_scale::vecdot::unpack_block<T>(src + raw_row * raw_bytes, codes.data(), ss.data(), zz.data());
          for (int j = 0; j < 256; ++j) {
            int const q = int(codes[size_t(j)]) + kCodeBias;
            TORCH_CHECK(q >= 0 && q < (1 << kCodeBits), "internal code outside ", kCodeBits,
                        "-bit offset-binary range: ", q);
            int64_t const kk = b * 256 + j;
            int64_t const low_bit = ((e * n + col) * k + kk) * kLoBits;
            lp[low_bit >> 3] |= uint8_t((q & ((1 << kLoBits) - 1)) << (low_bit & 7));
            if constexpr (kHiBits != 0) {
              int64_t const high_bit = ((e * n + col) * k + kk) * kHiBits;
              hp[high_bit >> 3] |= uint8_t((q >> kLoBits) << (high_bit & 7));
            }
          }
          for (int g = 0; g < Tr::kGroups; ++g) {
            int64_t const kg = b * Tr::kGroups + g;
            int64_t const dst = (e * scale_k + kg) * n + col;
            cutlass::half_t z = cutlass::half_t(float(zz[size_t(g)]) - float(kCodeBias) * float(ss[size_t(g)]));
            std::memcpy(sp + dst, &ss[size_t(g)], sizeof(cutlass::half_t));
            std::memcpy(zp + dst, &z, sizeof(cutlass::half_t));
          }
        }
      }
    }
    return {low, high, scale, zero};
  });
}

// OFFLINE ARTIFACT FOR fpA_intB_ppu.cuh. Start from the independently checked Native planes above, then ask the
// device library's host-only layout entry point to place them for the fixed dense tactic. This keeps the PPU-specific
// xplane types out of the portable torch extension while still making layout preparation a one-time operation rather
// than hidden work inside every GEMM launch.
std::vector<torch::Tensor> gguf_prepare_dense(torch::Tensor blocks, int64_t n, int64_t k, int64_t qtype) {
  TORCH_CHECK(blocks.dim() == 2, "dense preparation takes one [n*k/256,type_size] weight");
  TORCH_CHECK(n % 256 == 0 && k % 256 == 0, "dense fpA tactic needs n and k multiples of 256");
  auto raw = gguf_prepare_gemv(blocks, n, k, qtype);
  torch::Tensor low = torch::empty_like(raw[0]);
  torch::Tensor high = raw[1].numel() ? torch::empty_like(raw[1]) : torch::empty_like(raw[1]);
  auto const* api = ppu_backend::load();
  TORCH_CHECK(api && api->prepare_dense,
              "gguf_prepare_dense requires an hgcc libquactlize_ppu.so with the dense layout symbol");
  TORCH_CHECK(api->prepare_dense(get_ptr<uint8_t const>(raw[0]),
                                 raw[1].numel() ? get_ptr<uint8_t const>(raw[1]) : nullptr,
                                 get_ptr<uint8_t>(low), high.numel() ? get_ptr<uint8_t>(high) : nullptr,
                                 int(n), int(k), int(qtype)) == 0,
              "PPU dense offline layout preparation failed");

  // The int4 low-plane converter emits (low - 8). Q4/Q5's logical codes start at zero; Q6 was already shifted to
  // offset binary by gguf_prepare_gemv. In all three cases +8*scale in the affine channel preserves the official
  // dequantised value. uint2-low Q2/Q3 has no converter bias and is left untouched.
  torch::Tensor zero = raw[3].clone();
  if (qtype == kGgmlQ4K || qtype == kGgmlQ5K || qtype == kGgmlQ6K) {
    auto const* sp = get_ptr<at::Half const>(raw[2]);
    auto* zp = get_ptr<at::Half>(zero);
    for (int64_t i = 0; i < zero.numel(); ++i) zp[i] = at::Half(float(zp[i]) + 8.f * float(sp[i]));
  }
  return {low, high, raw[2], zero};
}

// THE FORMAT-SELECTED FULLY-QUANTIZED DENSE ARTIFACT. Weight placement is exactly gguf_prepare_dense's existing path;
// both code planes are retained (high is empty for Q4/Q2). Scale metadata is reordered from each official block into the
// byte-neutral [superblock,N,unit] object consumed by the shared packed-scale mainloop. pack_unit_sb owns the field
// addressing through Unit::ScaleBitLayout/MinBitLayout; PackedRaw names only official byte-aligned record slices.
std::vector<torch::Tensor> gguf_prepare_fully_quantized_dense(torch::Tensor blocks, int64_t n, int64_t k,
                                                              int64_t qtype) {
  CHECK_CPU(blocks); CHECK_CONTIGUOUS(blocks);
  TORCH_CHECK(n > 0 && n % 256 == 0 && k > 0 && k % 256 == 0,
              "fully-quantized dense preparation needs n and k multiples of 256");
  return dispatch_ktype(qtype, [&](auto tag) -> std::vector<torch::Tensor> {
    constexpr KType T = decltype(tag)::value;
    if constexpr (T == KType::Q4_K || T == KType::Q2_K || T == KType::Q5_K) {
      using R = PackedRaw<T>;
      using U = gguf_scale::packed_unit::Unit<T>;
      int64_t const nsb = k / 256;
      TORCH_CHECK(blocks.dtype() == torch::kUInt8 && blocks.dim() == 2 && blocks.size(1) == R::kRawBytes,
                  "blocks must be uint8 [n*k/256,", R::kRawBytes, "] for this format");
      TORCH_CHECK(blocks.size(0) == n * nsb, "dense weight needs n*k/256 blocks");

      auto dense = gguf_prepare_dense(blocks, n, k, qtype);
      torch::Tensor units = torch::empty({nsb, n, U::kUnitTotal},
          torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
      uint8_t const* src = get_ptr<uint8_t const>(blocks);
      uint8_t* dst = get_ptr<uint8_t>(units);
      for (int64_t col = 0; col < n; ++col)
        for (int64_t sb = 0; sb < nsb; ++sb)
          pack_raw_unit<T>(src + (col * nsb + sb) * R::kRawBytes,
                           dst + (sb * n + col) * U::kUnitTotal);
      return {dense[0], dense[1], units};
    } else {
      TORCH_CHECK(false, "fully-quantized dense preparation supports Q4_K, Q2_K and Q5_K");
      return {};
    }
  });
}

std::vector<torch::Tensor> gguf_prepare_fully_quantized_grouped(
    torch::Tensor blocks, int64_t n, int64_t k, int64_t qtype, int64_t experts) {
  CHECK_CPU(blocks); CHECK_CONTIGUOUS(blocks);
  TORCH_CHECK(experts > 0 && n > 0 && n % 256 == 0 && k > 0 && k % 256 == 0,
              "fully-quantized grouped preparation needs positive experts and n/k multiples of 256");
  return dispatch_ktype(qtype, [&](auto tag) -> std::vector<torch::Tensor> {
    constexpr KType T = decltype(tag)::value;
    if constexpr (T == KType::Q4_K || T == KType::Q2_K || T == KType::Q5_K) {
      using R = PackedRaw<T>;
      using U = gguf_scale::packed_unit::Unit<T>;
      constexpr int kLowBits = T == KType::Q2_K ? 2 : 4;
      constexpr int kHighBits = T == KType::Q5_K ? 1 : 0;
      int64_t const nsb = k / 256, blocks_per_expert = n * nsb;
      TORCH_CHECK(blocks.dtype() == torch::kUInt8 && blocks.dim() == 2 && blocks.size(1) == R::kRawBytes &&
                  blocks.size(0) == experts * blocks_per_expert,
                  "grouped blocks must be uint8 [experts*n*k/256,", R::kRawBytes, "] for this format");

      auto raw = gguf_prepare_gemv(
          blocks.view({experts, blocks_per_expert, int64_t(R::kRawBytes)}), n, k, qtype);
      torch::Tensor low = torch::empty_like(raw[0]);
      torch::Tensor high = torch::empty_like(raw[1]);
      auto const* api = ppu_backend::load();
      TORCH_CHECK(api && api->prepare_dense,
                  "fully-quantized grouped preparation requires the hgcc dense layout symbol");
      size_t const low_expert_bytes = size_t(n) * k * kLowBits / 8;
      size_t const high_expert_bytes = size_t(n) * k * kHighBits / 8;
      auto const* native_low = get_ptr<uint8_t const>(raw[0]);
      auto const* native_high = kHighBits ? get_ptr<uint8_t const>(raw[1]) : nullptr;
      auto* placed_low = get_ptr<uint8_t>(low);
      auto* placed_high = kHighBits ? get_ptr<uint8_t>(high) : nullptr;
      for (int64_t e = 0; e < experts; ++e) {
        TORCH_CHECK(api->prepare_dense(native_low + size_t(e) * low_expert_bytes,
                                       kHighBits ? native_high + size_t(e) * high_expert_bytes : nullptr,
                                       placed_low + size_t(e) * low_expert_bytes,
                                       kHighBits ? placed_high + size_t(e) * high_expert_bytes : nullptr,
                                       int(n), int(k), int(qtype)) == 0,
                    "PPU grouped dense placement failed for expert ", e);
      }

      torch::Tensor units = torch::empty({experts, nsb, n, U::kUnitTotal},
          torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
      uint8_t const* src = get_ptr<uint8_t const>(blocks);
      uint8_t* dst = get_ptr<uint8_t>(units);
      for (int64_t e = 0; e < experts; ++e)
        for (int64_t col = 0; col < n; ++col)
          for (int64_t sb = 0; sb < nsb; ++sb)
            pack_raw_unit<T>(src + ((e * n + col) * nsb + sb) * R::kRawBytes,
                             dst + ((e * nsb + sb) * n + col) * U::kUnitTotal);
      return {low, high, units};
    } else {
      TORCH_CHECK(false, "fully-quantized grouped preparation supports Q4_K, Q2_K and Q5_K");
      return {};
    }
  });
}

// BOTH RESIDENT ARTIFACTS HAVE AN INVERSE. Returning consumer-ready [E,N,K/gs] scale/zero planes separately is
// intentional: a full dequant comparison can say only that "packing or affine decode" is wrong, while this accessor
// identifies the affine half without involving any code-plane consumer. "Consumer-ready" matters: Q3/Q6 store the
// offset-binary correction in zero, and dense int4-family planes store the converter's low-8 correction. Returning
// those stored fp16 values is a bit-exact inverse; attempting to undo a rounded fp16 correction would be lossy.
template <bool Dense>
std::vector<torch::Tensor> gguf_artifact_dequantize_scale_impl(torch::Tensor scale, torch::Tensor zero,
                                                               int64_t qtype) {
  CHECK_CPU(scale); CHECK_CONTIGUOUS(scale); CHECK_CPU(zero); CHECK_CONTIGUOUS(zero);
  TORCH_CHECK(scale.dtype() == torch::kFloat16 && zero.dtype() == torch::kFloat16,
              "artifact scale and zero must be fp16");
  TORCH_CHECK(scale.dim() == 3 && scale.sizes() == zero.sizes(),
              "artifact scale and zero must have equal [experts,k/group_size,n] shapes");
  TORCH_CHECK(scale.size(0) > 0 && scale.size(1) > 0 && scale.size(2) > 0,
              "artifact scale planes must be nonempty");
  dispatch_ktype(qtype, [](auto) { return 0; });  // validate the wire qtype even though the transpose is common
  int64_t const experts = scale.size(0), scale_k = scale.size(1), n = scale.size(2);
  auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
  torch::Tensor stored_scale = torch::empty({experts, n, scale_k}, opts);
  torch::Tensor stored_zero = torch::empty_like(stored_scale);
  auto const* sp = get_ptr<at::Half const>(scale);
  auto const* zp = get_ptr<at::Half const>(zero);
  auto* osp = get_ptr<at::Half>(stored_scale);
  auto* ozp = get_ptr<at::Half>(stored_zero);
  for (int64_t e = 0; e < experts; ++e)
    for (int64_t col = 0; col < n; ++col)
      for (int64_t g = 0; g < scale_k; ++g) {
        int64_t const src = (e * scale_k + g) * n + col;
        int64_t const dst = (e * n + col) * scale_k + g;
        osp[dst] = sp[src];
        ozp[dst] = zp[src];
      }
  return {stored_scale, stored_zero};
}

inline uint8_t artifact_code(uint8_t const* p, int64_t logical, int bits) {
  int64_t const bit = logical * bits;
  return uint8_t((p[bit >> 3] >> (bit & 7)) & ((1 << bits) - 1));
}

template <bool Dense>
torch::Tensor gguf_artifact_dequantize_impl(torch::Tensor low, torch::Tensor high,
                                            torch::Tensor scale, torch::Tensor zero, int64_t qtype) {
  CHECK_CPU(low); CHECK_CONTIGUOUS(low); CHECK_CPU(high); CHECK_CONTIGUOUS(high);
  TORCH_CHECK(low.dtype() == torch::kUInt8 && high.dtype() == torch::kUInt8,
              "artifact code planes must be uint8");
  TORCH_CHECK(low.dim() == 3, "artifact low plane must be [experts,n,packed_k]");
  auto stored = gguf_artifact_dequantize_scale_impl<Dense>(scale, zero, qtype);
  return dispatch_ktype(qtype, [&](auto tag) -> torch::Tensor {
    constexpr KType T = decltype(tag)::value;
    using Tr = Traits<T>;
    constexpr int kLoBits = (T == KType::Q2_K || T == KType::Q3_K) ? 2 : 4;
    constexpr int kHiBits = (T == KType::Q3_K || T == KType::Q5_K) ? 1
                          : (T == KType::Q6_K) ? 2 : 0;
    constexpr int kDenseShift = Dense && kLoBits == 4 ? 8 : 0;
    int64_t const experts = scale.size(0), scale_k = scale.size(1), n = scale.size(2);
    int64_t const k = scale_k * Tr::kGroupSize;
    TORCH_CHECK(low.size(0) == experts && low.size(1) == n && low.size(2) == k * kLoBits / 8,
                "artifact low plane does not match its scale shape and qtype");
    if constexpr (kHiBits != 0) {
      TORCH_CHECK(high.dim() == 3 && high.size(0) == experts && high.size(1) == n &&
                  high.size(2) == k * kHiBits / 8, "artifact high plane does not match its scale shape and qtype");
    } else {
      TORCH_CHECK(high.numel() == 0, "single-plane artifact must have an empty high plane");
    }

    // Dense's physical xplane buffer is first recovered to the exact Native packed representation. Keeping this C
    // seam host-only lets the Python oracle run under nvcc without pulling PPU device headers into the torch module.
    torch::Tensor native_low = low, native_high = high;
    if constexpr (Dense) {
      TORCH_CHECK(experts == 1 && n % 256 == 0 && k % 256 == 0,
                  "dense artifact inverse needs one expert and n/k multiples of 256");
      native_low = torch::empty_like(low);
      native_high = high.numel() ? torch::empty_like(high) : torch::empty_like(high);
      auto const* api = ppu_backend::load();
      TORCH_CHECK(api && api->recover_dense,
                  "dense artifact dequant requires libquactlize_ppu.so's xplane inverse symbol");
      TORCH_CHECK(api->recover_dense(get_ptr<uint8_t const>(low), high.numel() ? get_ptr<uint8_t const>(high) : nullptr,
                                     get_ptr<uint8_t>(native_low),
                                     native_high.numel() ? get_ptr<uint8_t>(native_high) : nullptr,
                                     int(n), int(k), int(qtype)) == 0,
                  "PPU dense xplane inverse failed");
    }
    auto const* lp = get_ptr<uint8_t const>(native_low);
    auto const* hp = kHiBits ? get_ptr<uint8_t const>(native_high) : nullptr;
    auto const* sp = get_ptr<at::Half const>(stored[0]);
    auto const* zp = get_ptr<at::Half const>(stored[1]);
    torch::Tensor out = torch::empty({experts, n, k},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
    auto* op = get_ptr<at::Half>(out);
    for (int64_t e = 0; e < experts; ++e)
      for (int64_t col = 0; col < n; ++col)
        for (int64_t kk = 0; kk < k; ++kk) {
          int64_t const logical = (e * n + col) * k + kk;
          int q = artifact_code(lp, logical, kLoBits);
          if constexpr (kHiBits != 0) q += int(artifact_code(hp, logical, kHiBits)) << kLoBits;
          int64_t const affine = (e * n + col) * scale_k + kk / Tr::kGroupSize;
          op[logical] = at::Half(float(q - kDenseShift) * float(sp[affine]) + float(zp[affine]));
        }
    return out;
  });
}

std::vector<torch::Tensor> gguf_gemv_artifact_dequantize_scale(torch::Tensor scale, torch::Tensor zero,
                                                               int64_t qtype) {
  return gguf_artifact_dequantize_scale_impl<false>(scale, zero, qtype);
}
std::vector<torch::Tensor> gguf_dense_artifact_dequantize_scale(torch::Tensor scale, torch::Tensor zero,
                                                                int64_t qtype) {
  return gguf_artifact_dequantize_scale_impl<true>(scale, zero, qtype);
}
torch::Tensor gguf_gemv_artifact_dequantize(torch::Tensor low, torch::Tensor high, torch::Tensor scale,
                                            torch::Tensor zero, int64_t qtype) {
  return gguf_artifact_dequantize_impl<false>(low, high, scale, zero, qtype);
}
torch::Tensor gguf_dense_artifact_dequantize(torch::Tensor low, torch::Tensor high, torch::Tensor scale,
                                             torch::Tensor zero, int64_t qtype) {
  return gguf_artifact_dequantize_impl<true>(low, high, scale, zero, qtype);
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
  CHECK_CPU(d); CHECK_CONTIGUOUS(d);
  TORCH_CHECK(scale_blocks.dtype() == torch::kUInt8 && (scale_blocks.dim() == 2 || scale_blocks.dim() == 3),
              "scale_blocks must be uint8 [rows,bytes] or [N,superblocks,bytes]");
  TORCH_CHECK(d.dtype() == torch::kFloat16, "d must be float16");
  int64_t const rows = scale_blocks.numel() / scale_blocks.size(-1);
  bool const artifact = scale_blocks.dim() == 3;
  int64_t const ncols = artifact ? scale_blocks.size(0) : 0;
  int64_t const nsb = artifact ? scale_blocks.size(1) : 0;
  TORCH_CHECK(d.numel() == rows && ((!artifact && d.dim() == 1) ||
              (artifact && d.dim() == 2 && d.size(0) == ncols && d.size(1) == nsb)),
              "d must match scale_blocks' leading dimensions");
  bool const has_dmin = dmin.defined() && dmin.numel() > 0;
  if (has_dmin) {
    CHECK_CPU(dmin); CHECK_CONTIGUOUS(dmin);
    TORCH_CHECK(dmin.dtype() == torch::kFloat16 && dmin.sizes() == d.sizes(),
                "dmin must be float16 with the same shape as d");
  }
  auto const* bp = get_ptr<uint8_t const>(scale_blocks);
  return dispatch_ktype(qtype, [&](auto tag) -> std::vector<torch::Tensor> {
    constexpr KType T = decltype(tag)::value;
    using U = gguf_scale::packed_unit::Unit<T>;
    TORCH_CHECK(scale_blocks.size(-1) == Traits<T>::kBlockBytes,
                "scale_blocks last dim must be the SCALE block");
    TORCH_CHECK(!U::kHasMin || has_dmin, "this format has a min channel, so dmin is required");
    // A UNIT MAY CARRY MORE THAN ONE SUPERBLOCK, and for Q3_K and Q6_K it must: one superblock is 14 and 18 bytes,
    // both 2 mod 4, and ppu.cp.async moves only 4, 8 or 16. Two of the SAME COLUMN are 28 and 36. Consecutive rows
    // here are consecutive superblocks of one column, which is the axis that makes the pairing free -- a thread
    // still owns exactly its own column.
    TORCH_CHECK((artifact ? nsb : rows) % U::kSbPerUnit == 0, "this format packs ", U::kSbPerUnit,
                " superblocks per unit, so the superblock count must be a multiple of that; got ",
                artifact ? nsb : rows);
    int64_t const n_units = rows / U::kSbPerUnit;
    // THE ARTIFACT'S OUTER REORDER IS PART OF THE OFFLINE FORMAT: GGUF stores [N,sb], while the collective copies
    // [sb,N,unit]. The old flat op changed only bits inside each row, so no file produced what the kernel consumed;
    // test_q4k_packed_gemm had to rebuild it at load time with put_code. A 3-D input now names the axes and emits the
    // stored form. The 2-D reference form remains for per-row round-trip tests.
    torch::Tensor units = torch::empty(artifact
                                       ? std::vector<int64_t>{nsb / U::kSbPerUnit, ncols, U::kUnitTotal}
                                       : std::vector<int64_t>{n_units, U::kUnitTotal},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto* up = get_ptr<uint8_t>(units);
    auto const* dp = get_ptr<at::Half const>(d);
    auto const* mp = has_dmin ? get_ptr<at::Half const>(dmin) : nullptr;
    for (int64_t r = 0; r < rows; ++r) {
      int64_t const n = artifact ? r / nsb : 0;
      int64_t const sb = artifact ? r - n * nsb : r;
      int64_t const unit_row = artifact ? (sb / U::kSbPerUnit) * ncols + n : r / U::kSbPerUnit;
      cutlass::half_t dd, dm{0.f};
      std::memcpy(&dd, dp + r, sizeof(dd));
      if (mp) std::memcpy(&dm, mp + r, sizeof(dm));
      gguf_scale::packed_unit::pack_unit_sb<T>(bp + r * Traits<T>::kBlockBytes, dd, dm,
                                               int(sb % U::kSbPerUnit), up + unit_row * U::kUnitTotal);
    }
    return {units};
  });
}

// The decode side, so the round trip closes through Python rather than through a header nobody calls from a test.
std::vector<torch::Tensor> gguf_unit_decode(torch::Tensor units, int64_t qtype, int64_t zmul) {
  CHECK_CPU(units); CHECK_CONTIGUOUS(units);
  TORCH_CHECK(units.dtype() == torch::kUInt8 && (units.dim() == 2 || units.dim() == 3),
              "units must be uint8 [rows,unit] or artifact [unit_sb,N,unit]");
  TORCH_CHECK(zmul == 0 || zmul == 8, "zmul must be 0 or 8; it is the consumer's converter shift");
  bool const artifact = units.dim() == 3;
  int64_t const unit_sb = units.size(0);
  int64_t const ncols = artifact ? units.size(1) : 0;
  int64_t const rows = artifact ? unit_sb * ncols : units.size(0);
  auto const* up = get_ptr<uint8_t const>(units);
  return dispatch_ktype(qtype, [&](auto tag) -> std::vector<torch::Tensor> {
    constexpr KType T = decltype(tag)::value;
    using U = gguf_scale::packed_unit::Unit<T>;
    TORCH_CHECK(units.size(-1) == U::kUnitTotal, "this format's unit is ", U::kUnitTotal, " bytes, got ",
                units.size(-1));
    // One row of output per SUPERBLOCK, not per unit, so the caller sees the same shape whatever the packing is.
    int64_t const n_sb = rows * U::kSbPerUnit;
    auto f16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
    torch::Tensor scale = torch::empty(artifact
        ? std::vector<int64_t>{ncols, unit_sb * U::kSbPerUnit, U::kGroups}
        : std::vector<int64_t>{n_sb, U::kGroups}, f16);
    torch::Tensor zero = torch::empty_like(scale);
    auto* sp = get_ptr<at::Half>(scale);
    auto* zp = get_ptr<at::Half>(zero);
    for (int64_t r = 0; r < n_sb; ++r) {
      int64_t const n = artifact ? r / (unit_sb * U::kSbPerUnit) : 0;
      int64_t const sb_global = artifact ? r - n * unit_sb * U::kSbPerUnit : r;
      int64_t const unit_row = artifact ? (sb_global / U::kSbPerUnit) * ncols + n
                                        : r / U::kSbPerUnit;
      uint8_t const* u = up + unit_row * U::kUnitTotal;
      int const sb = int(sb_global % U::kSbPerUnit);
      for (int g = 0; g < U::kGroups; ++g) {
        auto sz = (zmul == 8) ? gguf_scale::packed_unit::unit_group_sb<T, 8>(u, sb, g)
                              : gguf_scale::packed_unit::unit_group_sb<T, 0>(u, sb, g);
        std::memcpy(sp + r * U::kGroups + g, &sz.scale, sizeof(sz.scale));
        std::memcpy(zp + r * U::kGroups + g, &sz.zero, sizeof(sz.zero));
      }
    }
    return {scale, zero};
  });
}

namespace {

int q4_nibble(uint8_t const* p, int64_t i) { return (p[i / 2] >> (4 * (i & 1))) & 0xf; }
void q4_put_nibble(uint8_t* p, int64_t i, int v) {
  uint8_t const mask = uint8_t(0xfu << (4 * (i & 1)));
  p[i / 2] = uint8_t((p[i / 2] & ~mask) | ((v & 0xf) << (4 * (i & 1))));
}

// THE CONSUMER-SIDE INVERSE OF `mixed_gemm`, written as a read of the stored artifact rather than by calling the
// forward preprocessor again. Its stages run in reverse: converter-word order, cache-line column interleave,
// sub-byte transpose, then the MMA row permutation. The output values are Q4's UNSIGNED codes: the forward +8 has
// already changed signed (q-8) nibbles back to q in the stored buffer.
std::vector<uint8_t> restore_q4_mixed(uint8_t const* stored, int64_t K, int64_t N) {
  int64_t const elements = K * N, bytes = elements / 2;
  std::vector<uint8_t> word(size_t(bytes), 0), col(size_t(bytes), 0), perm(size_t(bytes), 0), logical(size_t(bytes), 0);

  // Forward register layout: physical d receives logical 2d for d<4 and logical 2(d-4)+1 otherwise.
  for (int64_t r = 0; r < elements / 8; ++r)
    for (int d = 0; d < 8; ++d) {
      int const src = d < 4 ? 2 * d : 2 * (d - 4) + 1;
      q4_put_nibble(word.data(), r * 8 + src, q4_nibble(stored, r * 8 + d));
    }

  // ColumnMajorTileInterleave<64,4>, in 32-bit words (8 nibbles). Assign each original word from its stored offset.
  int64_t const vec_rows = K / 8, vec_rows_per_tile = 64 / 8, interleave = 4;
  for (int64_t n = 0; n < N; ++n)
    for (int64_t kr = 0; kr < vec_rows; ++kr) {
      int64_t const write_col = n / interleave;
      int64_t const base = (kr / vec_rows_per_tile) * vec_rows_per_tile;
      int64_t const write_row = interleave * base + vec_rows_per_tile * (n % interleave)
                              + kr % vec_rows_per_tile;
      int64_t const from = (write_col * vec_rows * interleave + write_row) * 4;
      int64_t const to = (n * vec_rows + kr) * 4;
      std::memcpy(col.data() + to, word.data() + from, 4);
    }

  // The sub-byte transpose stored [N,K]; recover the row-permuted [K,N] matrix nibble by nibble.
  for (int64_t n = 0; n < N; ++n)
    for (int64_t k = 0; k < K; ++k)
      q4_put_nibble(perm.data(), k * N + n, q4_nibble(col.data(), n * K + k));

  // Forward row w reads row r. Therefore the inverse writes the value at w back to r.
  for (int64_t base = 0; base < K; base += 32)
    for (int w = 0; w < 32; ++w) {
      int const r = 8 * ((w % 8) / 2) + w % 2 + 2 * (w / 8);
      for (int64_t n = 0; n < N; ++n)
        q4_put_nibble(logical.data(), (base + r) * N + n, q4_nibble(perm.data(), (base + w) * N + n));
    }
  return logical;
}

}  // namespace

// THE STORED-ARTIFACT ACCEPTANCE PATH, Q4_K. Both inputs have already crossed the offline boundary:
//   * weight is pack_int4 -> preprocess_weights_to_layout(..., "mixed_gemm")
//   * units are gguf_pack_unit([N,sb,...]) -> [sb,N,16]
// No raw GGUF byte is accepted here. That is the point: a test cannot let a shared misunderstanding of the source
// form cancel by quietly unpacking the raw block again in this arm.
torch::Tensor gguf_q4_artifact_dequantize(torch::Tensor weight, torch::Tensor units, std::string layout) {
  CHECK_CPU(weight); CHECK_CONTIGUOUS(weight); CHECK_CPU(units); CHECK_CONTIGUOUS(units);
  TORCH_CHECK(weight.dtype() == torch::kInt8 && weight.dim() == 2,
              "processed Q4 weight must be int8 [K,N/2]");
  TORCH_CHECK(units.dtype() == torch::kUInt8 && units.dim() == 3 && units.size(2) == 16,
              "Q4 artifact units must be uint8 [superblocks,N,16]");
  quactlize::LayoutPlan plan; std::string err;
  TORCH_CHECK(quactlize::resolve_layout(layout, &plan, &err), err);
  TORCH_CHECK(plan.name == "mmarow32_tr_cl4_cvtword_bias",
              "artifact consumer currently implements the mixed_gemm stored layout, got ", plan.name);
  int64_t const K = weight.size(0), N = weight.size(1) * 2, nsb = units.size(0);
  TORCH_CHECK(K % 256 == 0 && K / 256 == nsb, "weight K and unit superblocks disagree: K=", K, " nsb=", nsb);
  TORCH_CHECK(units.size(1) == N, "weight N and unit N disagree: ", N, " vs ", units.size(1));
  TORCH_CHECK(K % 64 == 0 && N % 64 == 0, "mixed_gemm Q4 artifact needs K and N multiples of 64");

  auto logical = restore_q4_mixed(get_ptr<uint8_t const>(weight), K, N);
  auto out = torch::empty({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* op = get_ptr<float>(out);
  uint8_t const* up = get_ptr<uint8_t const>(units);
  for (int64_t k = 0; k < K; ++k) {
    int64_t const sb = k / 256;
    int const i = int(k % 256), g = i / 32;
    for (int64_t n = 0; n < N; ++n) {
      auto const sz = gguf_scale::packed_unit::unit_group<KType::Q4_K, 8>(
          up + (sb * N + n) * 16, g);
      int const centered = q4_nibble(logical.data(), k * N + n) - 8;
      op[k * N + n] = float(centered) * float(sz.scale) + float(sz.zero);
    }
  }
  return out;
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
static auto gguf_q4_artifact_dequantize_op = torch::RegisterOperators(
    "quactlize::gguf_q4_artifact_dequantize", &torch_ext::gguf_q4_artifact_dequantize);

static auto gguf_unpack_op = torch::RegisterOperators("quactlize::gguf_unpack", &torch_ext::gguf_unpack);

static auto gguf_prepare_gemv_op =
    torch::RegisterOperators("quactlize::gguf_prepare_gemv", &torch_ext::gguf_prepare_gemv);

static auto gguf_prepare_dense_op =
    torch::RegisterOperators("quactlize::gguf_prepare_dense", &torch_ext::gguf_prepare_dense);

static auto gguf_prepare_fully_quantized_dense_op = torch::RegisterOperators(
    "quactlize::gguf_prepare_fully_quantized_dense", &torch_ext::gguf_prepare_fully_quantized_dense);

static auto gguf_prepare_fully_quantized_grouped_op = torch::RegisterOperators(
    "quactlize::gguf_prepare_fully_quantized_grouped", &torch_ext::gguf_prepare_fully_quantized_grouped);

static auto gguf_gemv_artifact_dequantize_op = torch::RegisterOperators(
    "quactlize::gguf_gemv_artifact_dequantize", &torch_ext::gguf_gemv_artifact_dequantize);
static auto gguf_gemv_artifact_dequantize_scale_op = torch::RegisterOperators(
    "quactlize::gguf_gemv_artifact_dequantize_scale", &torch_ext::gguf_gemv_artifact_dequantize_scale);
static auto gguf_dense_artifact_dequantize_op = torch::RegisterOperators(
    "quactlize::gguf_dense_artifact_dequantize", &torch_ext::gguf_dense_artifact_dequantize);
static auto gguf_dense_artifact_dequantize_scale_op = torch::RegisterOperators(
    "quactlize::gguf_dense_artifact_dequantize_scale", &torch_ext::gguf_dense_artifact_dequantize_scale);

static auto gguf_dequantize_op =
    torch::RegisterOperators("quactlize::gguf_dequantize", &torch_ext::gguf_dequantize);

static auto gguf_vecdot_op = torch::RegisterOperators("quactlize::gguf_vecdot", &torch_ext::gguf_vecdot);

static auto gguf_vecdot_dense_op =
    torch::RegisterOperators("quactlize::gguf_vecdot_dense", &torch_ext::gguf_vecdot_dense);

static auto gguf_vecdot_moe_op =
    torch::RegisterOperators("quactlize::gguf_vecdot_moe", &torch_ext::gguf_vecdot_moe);

static auto gguf_gemv_scale_first_op =
    torch::RegisterOperators("quactlize::gguf_gemv_scale_first", &torch_ext::gguf_gemv_scale_first);

static auto gguf_gemv_scale_first_moe_op =
    torch::RegisterOperators("quactlize::gguf_gemv_scale_first_moe", &torch_ext::gguf_gemv_scale_first_moe);

static auto gguf_dense_scale_first_op =
    torch::RegisterOperators("quactlize::gguf_dense_scale_first", &torch_ext::gguf_dense_scale_first);

static auto gguf_dense_fully_quantized_op = torch::RegisterOperators(
    "quactlize::gguf_dense_fully_quantized", &torch_ext::gguf_dense_fully_quantized);

static auto gguf_grouped_fully_quantized_op = torch::RegisterOperators(
    "quactlize::gguf_grouped_fully_quantized", &torch_ext::gguf_grouped_fully_quantized);

static auto gguf_scale_block_shape_op =
    torch::RegisterOperators("quactlize::gguf_scale_block_shape", &torch_ext::gguf_scale_block_shape);
