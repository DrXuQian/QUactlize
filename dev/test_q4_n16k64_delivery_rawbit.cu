// Real-device raw-bit probe for the proposed Q4 N16xK64 direct delivery.
//
// This is deliberately narrower than a GEMM.  It composes the exact
// AiuPlainProvider writer and Q4N16K64UniversalReader used by the layout-3
// proposal, but stops before conversion and MMA.  A unique uint32 tag is
// written for every physical word; the reader then exposes all 2048 words in
// a fixed (k64, n16, lane, vreg) output order.  The host oracle computes the
// wanted physical word from that tuple with an independent integer formula.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "hggc_runtime.h"

#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_ppu.hpp"
#include "cute/tensor.hpp"

#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_b_s2r_adapter.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_q4_n16k64_delivery.hpp"
#include "q4_n16k64_direct_offline.hpp"

namespace {

namespace direct =
    cutlass::gemm::collective::detail::quactlize_q4_n16k64_delivery;
namespace s2r =
    cutlass::gemm::collective::detail::quactlize_b_s2r;

constexpr int kTileN = 64;
constexpr int kTileK = 256;
constexpr int kWarpN = 64;
constexpr int kValuesPerVector = 4;
constexpr int kNCohorts = kWarpN / direct::kNAtom;
constexpr int kKBlocks = kTileK / direct::kLogicalKAtom;
constexpr std::uint32_t kPoison = UINT32_C(0xdeadbeef);

using Provider = direct::AiuPlainProvider<kTileN, kTileK>;
using Physical = typename Provider::Physical;
using Writer = typename Provider::WriterType;
using Reader = s2r::Q4N16K64UniversalReader<kTileN, kWarpN, kTileK>;

static_assert(q4_n16k64_direct::kLayoutId == 3);
static_assert(q4_n16k64_direct::kMappingId ==
              UINT64_C(0x51344e3136440001));
static_assert(Physical::physical_words == 2048);
static_assert(Physical::stage_bytes == 8192);
static_assert(Reader::reader_threads == 32);
static_assert(Reader::n_cohorts == kNCohorts);
static_assert(Reader::k_blocks == kKBlocks);

struct alignas(direct::kSharedAlignmentBytes) SharedStorage {
  std::uint32_t words[Physical::physical_words];
};

CUTE_HOST_DEVICE constexpr int output_index(int lane, int value,
                                             int n_cohort, int k_block) {
  return (((k_block * kNCohorts + n_cohort) * Reader::reader_threads +
           lane) *
              kValuesPerVector +
          value);
}

// Independent host oracle for the reader's public physical contract.
//
// One uint128 instruction owns four adjacent N words.  Eight lanes therefore
// span one 32-word N16 row; successive groups of eight lanes select the four
// K16 rows in one K64 block.  n_cohort selects the N16 slice and k_block the
// K64 slice.  This formula does not call a CuTe layout, partition or offline
// producer.
constexpr int source_word(int lane, int value, int n_cohort, int k_block) {
  int const vector_word = lane * kValuesPerVector + value;
  int const local_n_word = vector_word % 32;
  int const local_k_row = vector_word / 32;
  return k_block * 4 * Physical::physical_n_words +
         local_k_row * Physical::physical_n_words +
         n_cohort * 32 + local_n_word;
}

constexpr std::uint32_t word_tag(int word) {
  // Multiplication by an odd integer is a permutation modulo 2^32, so all
  // physical words receive distinct tags.
  return UINT32_C(0x51000000) ^
         (std::uint32_t(word + 1) * UINT32_C(0x9e3779b1));
}

std::uint64_t fnv1a(std::vector<std::uint32_t> const& words) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (std::uint32_t word : words) {
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= (word >> (8 * byte)) & 0xffu;
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

char const* error_text(hggcError_t status) {
  char const* text = hggcGetErrorString(status);
  return text ? text : "<no-error-text>";
}

bool runtime_ok(char const* operation, hggcError_t status) {
  if (status == hggcSuccess) return true;
  std::fprintf(stderr, "[q4-n16k64-rawbit] %s failed: %d:%s\n",
               operation, int(status), error_text(status));
  return false;
}

}  // namespace

// External linkage and stable spelling are intentional.  The runner binds
// its ISA checks to this exact symbol before accepting the binary for device
// handoff.
__global__ void q4_n16k64_delivery_rawbit_kernel(
    std::uint32_t const* source, std::uint32_t* output) {
  __shared__ SharedStorage storage;

  auto global = cute::make_tensor(cute::make_gmem_ptr(source),
                                  typename Physical::Layout{});
  auto mixed_global = cute::make_mix_tensor_like(global);
  auto shared = cute::make_tensor(cute::make_smem_ptr(storage.words),
                                  typename Physical::Layout{});

  typename Writer::Copy writer{};
  writer.desc_.gmem_ptr = reinterpret_cast<std::uint8_t const*>(source);
  writer.desc_.dim_h = Physical::physical_k_rows;
  writer.desc_.dim_w = Physical::physical_n_words;
  writer.desc_.cube_h = Writer::cube_k_rows;
  writer.desc_.cube_w = Writer::cube_n_words;
  writer.desc_.offset_w = 0;

  if (threadIdx.x == 0) {
    auto writer_thread = writer.get_slice(0);
    auto writer_source = writer_thread.partition_S(mixed_global);
    auto writer_destination = writer_thread.partition_D(shared);
    cute::copy(writer, writer_source, writer_destination);
  }
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  int const lane = int(threadIdx.x);
  if (lane >= Reader::reader_threads) return;

  auto shared_source = Reader::make_shared_source(
      cute::make_smem_ptr(storage.words), cute::_0{});
  auto source_partition =
      Reader::make_source_partition(shared_source, lane);
  auto source_view = Reader::make_copy_source_view(source_partition);
  auto owner = Reader::make_register_owner(source_partition);
  auto destination_view = Reader::make_copy_view(owner, lane);
  auto tiled_copy = Reader::make_tiled_copy();
  cute::copy(tiled_copy, source_view, destination_view);

  CUTE_STATIC_ASSERT_V(cute::size<0>(destination_view) == cute::_4{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(destination_view) ==
                       cute::Int<kNCohorts>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(destination_view) ==
                       cute::Int<kKBlocks>{});
  CUTE_UNROLL
  for (int k_block = 0; k_block < kKBlocks; ++k_block) {
    CUTE_UNROLL
    for (int n_cohort = 0; n_cohort < kNCohorts; ++n_cohort) {
      CUTE_UNROLL
      for (int value = 0; value < kValuesPerVector; ++value) {
        output[output_index(lane, value, n_cohort, k_block)] =
            destination_view(value, n_cohort, k_block);
      }
    }
  }
}

int main(int argc, char** argv) {
  bool const plant_wrong_oracle =
      argc == 2 && std::strcmp(argv[1], "--plant-wrong-oracle") == 0;
  if (argc > 2 || (argc == 2 && !plant_wrong_oracle)) {
    std::fprintf(stderr, "usage: %s [--plant-wrong-oracle]\n", argv[0]);
    return 2;
  }

  std::vector<std::uint32_t> source(Physical::physical_words);
  std::vector<std::uint32_t> expected(Physical::physical_words);
  std::vector<std::uint32_t> got(Physical::physical_words, kPoison);
  std::array<int, Physical::physical_words> source_hits{};

  for (int word = 0; word < Physical::physical_words; ++word) {
    source[std::size_t(word)] = word_tag(word);
    if (source[std::size_t(word)] == kPoison) {
      std::fprintf(stderr, "source tag aliases poison at word %d\n", word);
      return 2;
    }
  }
  for (int k_block = 0; k_block < kKBlocks; ++k_block) {
    for (int n_cohort = 0; n_cohort < kNCohorts; ++n_cohort) {
      for (int lane = 0; lane < Reader::reader_threads; ++lane) {
        for (int value = 0; value < kValuesPerVector; ++value) {
          int const output = output_index(lane, value, n_cohort, k_block);
          int word = source_word(lane, value, n_cohort, k_block);
          if (plant_wrong_oracle && output == 0) word ^= 1;
          if (word < 0 || word >= Physical::physical_words) {
            std::fprintf(stderr, "oracle word out of range: %d\n", word);
            return 2;
          }
          expected[std::size_t(output)] = source[std::size_t(word)];
          ++source_hits[std::size_t(
              source_word(lane, value, n_cohort, k_block))];
        }
      }
    }
  }
  for (int word = 0; word < Physical::physical_words; ++word) {
    if (source_hits[std::size_t(word)] != 1) {
      std::fprintf(stderr, "oracle coverage at word %d is %d, expected 1\n",
                   word, source_hits[std::size_t(word)]);
      return 2;
    }
  }

  std::uint32_t* device_source = nullptr;
  std::uint32_t* device_output = nullptr;
  std::size_t const bytes = source.size() * sizeof(source[0]);
  bool setup_ok =
      runtime_ok("hggcMalloc(source)", hggcMalloc(
          reinterpret_cast<void**>(&device_source), bytes));
  if (setup_ok)
    setup_ok = runtime_ok("hggcMalloc(output)", hggcMalloc(
        reinterpret_cast<void**>(&device_output), bytes));
  if (setup_ok)
    setup_ok &= runtime_ok("hggcMemcpy(source H2D)", hggcMemcpy(
        device_source, source.data(), bytes, hggcMemcpyHostToDevice));
  if (setup_ok)
    setup_ok &= runtime_ok("hggcMemcpy(poison H2D)", hggcMemcpy(
        device_output, got.data(), bytes, hggcMemcpyHostToDevice));

  hggcError_t before = hggcGetLastError();
  hggcError_t immediate = before;
  hggcError_t synchronize = before;
  hggcError_t copy_back = before;
  if (setup_ok && before == hggcSuccess) {
    q4_n16k64_delivery_rawbit_kernel<<<1, Reader::reader_threads>>>(
        device_source, device_output);
    immediate = hggcGetLastError();
    synchronize = hggcDeviceSynchronize();
    copy_back = hggcMemcpy(got.data(), device_output, bytes,
                           hggcMemcpyDeviceToHost);
  }

  int raw_bad = 0;
  int sentinel = 0;
  int first = -1;
  for (int i = 0; i < Physical::physical_words; ++i) {
    sentinel += got[std::size_t(i)] == kPoison;
    if (got[std::size_t(i)] != expected[std::size_t(i)]) {
      if (first < 0) first = i;
      ++raw_bad;
    }
  }
  bool const launch_ok = setup_ok && before == hggcSuccess &&
                         immediate == hggcSuccess &&
                         synchronize == hggcSuccess &&
                         copy_back == hggcSuccess;
  bool const pass = launch_ok && raw_bad == 0 && sentinel == 0;

  int first_lane = -1, first_value = -1, first_cohort = -1,
      first_k_block = -1;
  if (first >= 0) {
    int q = first;
    first_value = q % kValuesPerVector;
    q /= kValuesPerVector;
    first_lane = q % Reader::reader_threads;
    q /= Reader::reader_threads;
    first_cohort = q % kNCohorts;
    first_k_block = q / kNCohorts;
  }

  std::printf(
      "FQ_Q4_N16K64_DELIVERY_RAWBIT verdict=%s layout=3 "
      "mapping_id=0x%016llx words=%d raw_bad=%d sentinel=%d "
      "source_hash=0x%016llx want_hash=0x%016llx got_hash=0x%016llx "
      "first=[index:%d,kblock:%d,ncohort:%d,lane:%d,vreg:%d,"
      "want:0x%08x,got:0x%08x] "
      "launch=[before:%d,immediate:%d,sync:%d,copy:%d] plant=%s\n",
      pass ? "PASS" : "FAIL",
      static_cast<unsigned long long>(q4_n16k64_direct::kMappingId),
      Physical::physical_words, raw_bad, sentinel,
      static_cast<unsigned long long>(fnv1a(source)),
      static_cast<unsigned long long>(fnv1a(expected)),
      static_cast<unsigned long long>(fnv1a(got)), first, first_k_block,
      first_cohort, first_lane, first_value,
      first >= 0 ? expected[std::size_t(first)] : 0u,
      first >= 0 ? got[std::size_t(first)] : 0u,
      int(before), int(immediate), int(synchronize), int(copy_back),
      plant_wrong_oracle ? "wrong-oracle" : "none");

  if (device_output) hggcFree(device_output);
  if (device_source) hggcFree(device_source);
  return pass ? 0 : 1;
}
