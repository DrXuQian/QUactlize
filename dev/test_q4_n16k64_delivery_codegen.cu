// Compile-only device-code gate for the proposed Q4 N16xK64 delivery pair.
//
// The kernel is intentionally never launched.  Its sole purpose is to force
// hgcc to lower the real AiuPlainProvider writer and the real
// Q4N16K64UniversalReader into one device body.  Device correctness and
// performance remain separate box admission steps.

#include <cstdint>
#include <type_traits>

#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_ppu.hpp"
#include "cute/tensor.hpp"

#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_b_s2r_adapter.hpp"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_q4_n16k64_delivery.hpp"

namespace {

namespace direct =
    cutlass::gemm::collective::detail::quactlize_q4_n16k64_delivery;
namespace s2r = cutlass::gemm::collective::detail::quactlize_b_s2r;

constexpr int kTileN = 64;
constexpr int kTileK = 256;
constexpr int kWarpN = 64;

using Provider = direct::AiuPlainProvider<kTileN, kTileK>;
using Physical = typename Provider::Physical;
using Writer = typename Provider::WriterType;
using Reader = s2r::Q4N16K64UniversalReader<kTileN, kWarpN, kTileK>;

static_assert(std::is_same_v<typename Writer::CopyInst,
                             cute::PPU0010_AIU_LOAD<
                                 cute::C<Writer::cube_bits>, std::uint32_t,
                                 false, false>>);
static_assert(std::is_same_v<typename Reader::TiledCopy,
                             typename Provider::ReaderType::Copy>);
static_assert(Physical::stage_bytes == 8192);
static_assert(Physical::physical_words == 2048);

struct alignas(direct::kSharedAlignmentBytes) SharedStorage {
  std::uint32_t words[Physical::physical_words];
};

}  // namespace

// Keep external linkage and a stable spelling: the codegen runner identifies
// this exact kernel in the PPU image before inspecting its instructions.
__global__ void q4_n16k64_delivery_codegen_kernel(
    std::uint32_t const* source, std::uint32_t* sink) {
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

  // AIU is one opaque logical issuer.  Using the TiledCopy partitions here,
  // instead of invoking CopyInst directly, also instantiates the provider's
  // actual CopyAtom and its complete-TileN repetition.
  if (threadIdx.x == 0) {
    auto writer_thread = writer.get_slice(0);
    auto writer_source = writer_thread.partition_S(mixed_global);
    auto writer_destination = writer_thread.partition_D(shared);
    cute::copy(writer, writer_source, writer_destination);
  }
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  // One warp consumes the complete TN64/TK256 stage through the proposed
  // UniversalCopy<uint128_t> reader.  Persisting a checksum makes every
  // register fragment observable and prevents the shared loads from being
  // removed as dead code.
  if (threadIdx.x < direct::kReaderThreads) {
    int const lane = int(threadIdx.x);
    auto shared_source = Reader::make_shared_source(
        cute::make_smem_ptr(storage.words), cute::_0{});
    auto source_partition =
        Reader::make_source_partition(shared_source, lane);
    auto source_view = Reader::make_copy_source_view(source_partition);
    auto owner = Reader::make_register_owner(source_partition);
    auto destination_view = Reader::make_copy_view(owner, lane);
    auto tiled_copy = Reader::make_tiled_copy();
    cute::copy(tiled_copy, source_view, destination_view);

    std::uint32_t checksum = 0;
    CUTE_UNROLL
    for (int i = 0; i < int(cute::size(owner)); ++i) {
      checksum ^= owner(i);
    }
    sink[lane] = checksum;
  }
}
