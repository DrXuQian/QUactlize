// Device numeric closure for the experimental Q4 N16xK64 direct delivery.
//
// This probe deliberately does not instantiate a shipping kernel or dispatch
// policy.  It composes only the candidate's real semantic chain:
//
//   layout-3 bytes -> AIU plain shared -> UniversalCopy<uint128_t>
//     -> int4 fast converter -> real PPU0010 m8 partition_fragment_B
//
// The host starts from logical Q4 codes, calls the public layout-3 prepare
// function, and expects the logical fragment to contain exactly code - 8 as
// fp16 bits.  Two alternative, byte-count-identical offline maps are negative
// controls: the shipping K-pack4 map and DeepGEMM-for-sail's real N64xK16
// W4A16 permutation.  N64xK64 is the least common tile that contains four
// complete direct N16xK64 atoms and four complete DeepGEMM N64xK16 atoms.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "hggc_runtime.h"

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/arch/copy_ppu.hpp"
#include "cute/tensor.hpp"

#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_b_s2r_adapter.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_q4_n16k64_delivery.hpp"
#include "q4_kpack4_offline.hpp"
#include "q4_n16k64_direct_offline.hpp"

namespace {

using namespace cute;
namespace direct =
    cutlass::gemm::collective::detail::quactlize_q4_n16k64_delivery;
namespace s2r =
    cutlass::gemm::collective::detail::quactlize_b_s2r;

constexpr int kN = 64;
constexpr int kK = 64;
constexpr int kCodes = kN * kK;
constexpr int kBytes = kCodes / 2;
constexpr std::uint16_t kPoison = UINT16_C(0x7bff);

using Provider = direct::AiuPlainProvider<kN, kK>;
using Physical = typename Provider::Physical;
using Writer = typename Provider::WriterType;
constexpr int kWarpN = 16;
constexpr int kWarpNTiles = kN / kWarpN;
using Reader = s2r::Q4N16K64UniversalReader<kN, kWarpN, kK>;
using Compute = TiledMMA<
    MMA_Atom<PPU0010_8x16x16_F32F16F16F32_TN>,
    Layout<Shape<_1, Int<kWarpNTiles>, _1>>,
    Tile<_8, Int<kN>, Int<kK>>>;
constexpr int kThreads = int(size(Compute{}));

static_assert(Physical::physical_words == 512);
static_assert(Physical::stage_bytes == kBytes);
static_assert(Reader::k_blocks == 1 && Reader::n_cohorts == 1);
static_assert(Reader::warp_n_tiles == kWarpNTiles);
static_assert(kThreads == 128);

struct alignas(direct::kSharedAlignmentBytes) SharedStorage {
  std::uint32_t words[Physical::physical_words];
};

constexpr std::uint16_t signed_code_half_bits(int code) {
  constexpr std::uint16_t table[16] = {
      0xc800, 0xc700, 0xc600, 0xc500,
      0xc400, 0xc200, 0xc000, 0xbc00,
      0x0000, 0x3c00, 0x4000, 0x4200,
      0x4400, 0x4500, 0x4600, 0x4700};
  return table[code & 15];
}

constexpr int fixture_code(int n, int k) {
  return (7 * n + 5 * k + ((n ^ k) >> 1)) & 15;
}

// DeepGEMM-for-sail f89eae1, tests/math_utils.py::_get_perms.  Its native
// atom is N64xK16 and its placed storage has the same outer [K/16][2*N] u32
// shape as this candidate.  Equal outer shape is not byte compatibility:
// this exact map is a deliberately wrong producer for our reader.
constexpr int deepgemm_physical_nibble(int n, int k) {
  int const kr = k & 15;
  int const nr = n & 63;
  int const row = k >> 4;
  int const column = 128 * (n >> 6) + 16 * (n & 7) +
                     4 * ((kr >> 1) & 3) + ((nr >> 4) & 3);
  int const nibble = 4 * (kr & 1) + 2 * ((nr >> 3) & 1) +
                     ((kr >> 3) & 1);
  return 8 * (row * (2 * kN) + column) + nibble;
}

constexpr bool deepgemm_map_is_bijection() {
  bool seen[kCodes] = {};
  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      int const p = deepgemm_physical_nibble(n, k);
      if (p < 0 || p >= kCodes || seen[p]) return false;
      seen[p] = true;
    }
  }
  return true;
}

static_assert(deepgemm_map_is_bijection());

void nibble_put(std::uint8_t* bytes, int p, int code) {
  int const shift = 4 * (p & 1);
  std::uint8_t const mask = std::uint8_t(0xfu << shift);
  bytes[p >> 1] = std::uint8_t(
      (bytes[p >> 1] & std::uint8_t(~mask)) |
      ((std::uint8_t(code) & 0xfu) << shift));
}

std::uint64_t fnv1a(void const* data, std::size_t bytes) {
  auto const* p = static_cast<std::uint8_t const*>(data);
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

bool runtime_ok(char const* operation, hggcError_t status) {
  if (status == hggcSuccess) return true;
  char const* text = hggcGetErrorString(status);
  std::fprintf(stderr, "[q4-n16k64-fragment] %s failed: %d:%s\n",
               operation, int(status), text ? text : "<no-error-text>");
  return false;
}

}  // namespace

__global__ void q4_n16k64_fragment_numeric_kernel(
    std::uint32_t const* source, std::uint16_t* output) {
  __shared__ SharedStorage storage;

  auto global = make_tensor(make_gmem_ptr(source), typename Physical::Layout{});
  auto mixed_global = make_mix_tensor_like(global);
  auto shared = make_tensor(make_smem_ptr(storage.words),
                            typename Physical::Layout{});

  typename Writer::Copy writer{};
  writer.desc_.gmem_ptr = reinterpret_cast<std::uint8_t const*>(source);
  writer.desc_.dim_h = Physical::physical_k_rows;
  writer.desc_.dim_w = Physical::physical_n_words;
  writer.desc_.cube_h = Writer::cube_k_rows;
  writer.desc_.cube_w = Writer::cube_n_words;
  writer.desc_.offset_w = 0;
  if (threadIdx.x == 0) {
    auto slice = writer.get_slice(0);
    copy(writer, slice.partition_S(mixed_global), slice.partition_D(shared));
  }
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  int const thread = int(threadIdx.x);
  if (thread >= kThreads) return;
  int const lane = thread % Reader::reader_threads;
  int const warp_n_tile = thread / Reader::reader_threads;

  auto shared_source = Reader::make_shared_source(
      make_smem_ptr(storage.words), warp_n_tile);
  auto source_partition = Reader::make_source_partition(shared_source, lane);
  auto source_view = Reader::make_copy_source_view(source_partition);
  auto register_owner = Reader::make_register_owner(source_partition);
  auto copy_view = Reader::make_copy_view(register_owner, lane);
  copy(Reader::make_tiled_copy(), source_view, copy_view);

  auto logical_smem = make_tensor(
      make_smem_ptr(reinterpret_cast<cutlass::half_t*>(storage.words)),
      make_layout(Shape<Int<kN>, Int<kK>>{}, Stride<Int<kK>, _1>{}));
  auto thr_mma = Compute{}.get_thread_slice(thread);
  auto logical_owner = thr_mma.partition_fragment_B(logical_smem);
  auto converter_input = Reader::make_converter_input(copy_view, _0{});
  auto converter_output = Reader::make_converter_destination(
      logical_owner, converter_input, _0{}, _4{});

  using SrcArray = cutlass::Array<cutlass::int4b_t, 32>;
  using DstArray = cutlass::Array<cutlass::half_t, 32>;
  using Converter = cutlass::MixGemmNumericArrayConverter<
      cutlass::half_t, cutlass::int4b_t, 32>;
  static_assert(size(decltype(converter_input.layout()){}) == 32);
  static_assert(size(decltype(converter_output.layout()){}) == 32);
  auto const* src = reinterpret_cast<SrcArray const*>(
      raw_pointer_cast(converter_input.data()));
  auto* dst = reinterpret_cast<DstArray*>(
      raw_pointer_cast(converter_output.data()));
  *dst = Converter::convert(*src);

  auto logical_identity = make_identity_tensor(Shape<Int<kN>, Int<kK>>{});
  auto logical_partition = thr_mma.partition_B(logical_identity);
  auto inverse = right_inverse(logical_owner.layout());
  auto const* fragment_bits = reinterpret_cast<std::uint16_t const*>(
      raw_pointer_cast(logical_owner.data()));
  CUTE_UNROLL
  for (int raw = 0; raw < int(size(logical_owner)); ++raw) {
    auto const nk = logical_partition(inverse(raw));
    int const n = int(get<0>(nk));
    int const k = int(get<1>(nk));
    output[n * kK + k] = fragment_bits[raw];
  }
}

int main(int argc, char** argv) {
  enum class Plant { None, Layout1, DeepGemm };
  Plant plant = Plant::None;
  if (argc == 2 && std::strcmp(argv[1], "--plant-layout1") == 0)
    plant = Plant::Layout1;
  else if (argc == 2 && std::strcmp(argv[1], "--plant-deepgemm") == 0)
    plant = Plant::DeepGemm;
  else if (argc != 1) {
    std::fprintf(stderr,
                 "usage: %s [--plant-layout1|--plant-deepgemm]\n", argv[0]);
    return 2;
  }

  std::vector<std::uint8_t> native(kBytes, 0);
  std::vector<std::uint8_t> placed(kBytes, 0);
  std::vector<std::uint16_t> expected(kCodes);
  std::vector<std::uint16_t> got(kCodes, kPoison);
  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      int const code = fixture_code(n, k);
      q4_n16k64_direct::native_put(native.data(), n, k, kK,
                                    std::uint8_t(code));
      expected[std::size_t(n * kK + k)] = signed_code_half_bits(code);
    }
  }

  int prepare_rc = 0;
  if (plant == Plant::None) {
    prepare_rc = q4_n16k64_direct::prepare(
        native.data(), placed.data(), kN, kK);
  } else if (plant == Plant::Layout1) {
    prepare_rc = q4_kpack4::prepare(native.data(), placed.data(), kN, kK);
  } else {
    for (int n = 0; n < kN; ++n)
      for (int k = 0; k < kK; ++k)
        nibble_put(placed.data(), deepgemm_physical_nibble(n, k),
                   fixture_code(n, k));
  }
  if (prepare_rc != 0) {
    std::fprintf(stderr, "offline prepare failed: %d\n", prepare_rc);
    return 2;
  }
  int offline_bad = 0;
  for (int n = 0; n < kN; ++n)
    for (int k = 0; k < kK; ++k)
      offline_bad += q4_n16k64_direct::placed_get(
                         placed.data(), n, k, kN) != fixture_code(n, k);
  int const expected_offline_bad = plant == Plant::None ? 0 : 3840;
  if (offline_bad != expected_offline_bad) {
    std::fprintf(stderr,
                 "offline control is inadmissible: plant=%d bad=%d want=%d\n",
                 int(plant), offline_bad, expected_offline_bad);
    return 2;
  }

  std::uint32_t* device_source = nullptr;
  std::uint16_t* device_output = nullptr;
  bool setup = runtime_ok("hggcMalloc(source)", hggcMalloc(
      reinterpret_cast<void**>(&device_source), placed.size()));
  setup &= runtime_ok("hggcMalloc(output)", hggcMalloc(
      reinterpret_cast<void**>(&device_output), got.size() * sizeof(got[0])));
  if (setup)
    setup &= runtime_ok("hggcMemcpy(source H2D)", hggcMemcpy(
        device_source, placed.data(), placed.size(), hggcMemcpyHostToDevice));
  if (setup)
    setup &= runtime_ok("hggcMemcpy(poison H2D)", hggcMemcpy(
        device_output, got.data(), got.size() * sizeof(got[0]),
        hggcMemcpyHostToDevice));

  hggcError_t before = hggcGetLastError();
  hggcError_t immediate = before;
  hggcError_t synchronize = before;
  hggcError_t copy_back = before;
  if (setup && before == hggcSuccess) {
    q4_n16k64_fragment_numeric_kernel<<<1, kThreads>>>(
        device_source, device_output);
    immediate = hggcGetLastError();
    synchronize = hggcDeviceSynchronize();
    copy_back = hggcMemcpy(
        got.data(), device_output, got.size() * sizeof(got[0]),
        hggcMemcpyDeviceToHost);
  }

  int raw_bad = 0;
  int sentinel = 0;
  int first = -1;
  for (int i = 0; i < kCodes; ++i) {
    sentinel += got[std::size_t(i)] == kPoison;
    if (got[std::size_t(i)] != expected[std::size_t(i)]) {
      if (first < 0) first = i;
      ++raw_bad;
    }
  }
  bool const launch_ok = setup && before == hggcSuccess &&
                         immediate == hggcSuccess &&
                         synchronize == hggcSuccess &&
                         copy_back == hggcSuccess;
  bool const pass = launch_ok && sentinel == 0 && raw_bad == 0;
  char const* plant_name = plant == Plant::None ? "none" :
      plant == Plant::Layout1 ? "layout1" : "deepgemm";
  std::printf(
      "FQ_Q4_N16K64_FRAGMENT_NUMERIC verdict=%s mapping_id=0x%016llx "
      "codes=%d offline_bad=%d raw_bad=%d sentinel=%d first=[index:%d,n:%d,k:%d,"
      "want:0x%04x,got:0x%04x] native_hash=0x%016llx "
      "placed_hash=0x%016llx want_hash=0x%016llx got_hash=0x%016llx "
      "launch=[before:%d,immediate:%d,sync:%d,copy:%d] plant=%s\n",
      pass ? "PASS" : "FAIL",
      static_cast<unsigned long long>(q4_n16k64_direct::kMappingId),
      kCodes, offline_bad, raw_bad, sentinel, first,
      first < 0 ? -1 : first / kK,
      first < 0 ? -1 : first % kK,
      first < 0 ? 0 : expected[std::size_t(first)],
      first < 0 ? 0 : got[std::size_t(first)],
      static_cast<unsigned long long>(fnv1a(native.data(), native.size())),
      static_cast<unsigned long long>(fnv1a(placed.data(), placed.size())),
      static_cast<unsigned long long>(fnv1a(
          expected.data(), expected.size() * sizeof(expected[0]))),
      static_cast<unsigned long long>(fnv1a(
          got.data(), got.size() * sizeof(got[0]))),
      int(before), int(immediate), int(synchronize), int(copy_back),
      plant_name);

  if (device_output) hggcFree(device_output);
  if (device_source) hggcFree(device_source);
  return pass ? 0 : 1;
}
