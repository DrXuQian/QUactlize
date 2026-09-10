#include <hggc_runtime.h>
#include "api.h"
#include "word_pack.hpp"
#include <algorithm>
#include <limits>

namespace {
using gguf_scale::KType;

template<class Source>
__global__ void pack_q8(Source raw, uint16_t* low, uint8_t* scale,
                        int n, int k, uint64_t words, uint64_t groups) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += uint64_t(gridDim.x) * blockDim.x) {
    low[i] = quactlize_pack::Q8::word(raw,n,k,i);
    if (i < groups) quactlize_pack::Q8::metadata(raw,scale,n,k,i);
  }
}

template <KType T, bool High, class Source>
__global__ void pack_words(Source raw, uint16_t* output,
                           int n, int k, uint64_t count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x)
    output[i] = quactlize_pack::Plane<T, High>::word(raw, n, k, i);
}

template <KType T, class Source>
__global__ void pack_metadata(Source raw, uint8_t* output,
                              int n, int k, uint64_t count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x)
    quactlize_pack::metadata<T>(raw, output, n, k, i);
}

unsigned grid(uint64_t count) {
  return unsigned(std::min<uint64_t>((count + 255) / 256, 65535));
}

template <KType T, class Source>
int launch(Source raw, uint8_t* low, uint8_t* high, uint8_t* units,
           int n, int k, quactlize_ppu_kpack_sizes_v1 const& s, hggcStream_t stream) {
  pack_words<T, false><<<grid(s.low_bytes / 2), 256, 0, stream>>>(
      raw, reinterpret_cast<uint16_t*>(low), n, k, s.low_bytes / 2);
  if (hggcGetLastError() != hggcSuccess) return 41;
  if constexpr (gguf_scale::CodeTraits<T>::kHiBytes != 0) {
    pack_words<T, true><<<grid(s.high_bytes / 2), 256, 0, stream>>>(
        raw, reinterpret_cast<uint16_t*>(high), n, k, s.high_bytes / 2);
    if (hggcGetLastError() != hggcSuccess) return 41;
  }
  uint64_t const count = s.units_bytes / gguf_scale::packed_unit::Unit<T>::kUnitTotal;
  pack_metadata<T><<<grid(count), 256, 0, stream>>>(raw, units, n, k, count);
  return hggcGetLastError() == hggcSuccess ? 0 : 41;
}

template<class Source>
int launch_format(Source raw, uint8_t* low, uint8_t* high, uint8_t* units,
                  int n, int k, int qtype, quactlize_ppu_kpack_sizes_v1 const& s, void* stream) {
  if (hggcGetLastError() != hggcSuccess) return 41;
  if (qtype == 8) {
    pack_q8<<<grid(s.low_bytes/2),256,0,static_cast<hggcStream_t>(stream)>>>(
        raw,reinterpret_cast<uint16_t*>(low),units,n,k,s.low_bytes/2,s.units_bytes/2);
    return hggcGetLastError() == hggcSuccess ? 0 : 41;
  }
#define QP_CASE(Q, T) case Q: return launch<KType::T>(raw, low, high, units, n, k, s, static_cast<hggcStream_t>(stream))
  switch (qtype) {
    QP_CASE(10, Q2_K); QP_CASE(11, Q3_K); QP_CASE(12, Q4_K);
    QP_CASE(13, Q5_K); QP_CASE(14, Q6_K);
    default: return 22;
  }
#undef QP_CASE
}

int disjoint(uintptr_t const* begin, uint64_t const* bytes, int count) {
  for (int i = 0; i < count; ++i) {
    if (bytes[i] > std::numeric_limits<uintptr_t>::max() - begin[i]) return 26;
    for (int j = 0; j < i; ++j)
      if (bytes[i] && bytes[j] && begin[i] < begin[j] + bytes[j] && begin[j] < begin[i] + bytes[i])
        return 30;
  }
  return 0;
}
}

extern "C" int quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2(
    uint8_t const* blocks, uint8_t* low, uint8_t* high, uint8_t* units,
    int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, void* stream) {
  quactlize_ppu_kpack_sizes_v1 s{};
  int const rc = quactlize_ppu_kpack_sizes_for_arrangement_v1(n, k, experts, qtype, arrangement, &s);
  if (rc) return rc;
  if (!blocks || !low || !units || (s.high_bytes != 0) != (high != nullptr) ||
      uintptr_t(low) % 2 || uintptr_t(high) % 2) return 20;
  uintptr_t const begin[] = {uintptr_t(blocks), uintptr_t(low), uintptr_t(high), uintptr_t(units)};
  uint64_t const bytes[] = {s.raw_bytes, s.low_bytes, s.high_bytes, s.units_bytes};
  int const spans = disjoint(begin,bytes,4);
  return spans ? spans : launch_format(blocks,low,high,units,n,k,qtype,s,stream);
}

extern "C" int quactlize_ppu_prepare_gate_up_dev_for_arrangement_v1(
    uint8_t const* gate, uint8_t const* up, uint8_t* low, uint8_t* high, uint8_t* units,
    int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement, void* stream) {
  if (n <= 0 || n > INT32_MAX/2) return 24;
  quactlize_ppu_kpack_sizes_v1 single{}, paired{};
  int rc = quactlize_ppu_kpack_sizes_for_arrangement_v1(n,k,experts,qtype,arrangement,&single);
  if (rc) return rc;
  rc = quactlize_ppu_kpack_sizes_for_arrangement_v1(2*n,k,experts,qtype,arrangement,&paired);
  if (rc) return rc;
  if (!gate || !up || !low || !units || (paired.high_bytes != 0) != (high != nullptr) ||
      uintptr_t(low)%2 || uintptr_t(high)%2) return 20;
  uintptr_t const begin[] = {uintptr_t(gate),uintptr_t(up),uintptr_t(low),uintptr_t(high),uintptr_t(units)};
  uint64_t const bytes[] = {single.raw_bytes,single.raw_bytes,paired.low_bytes,paired.high_bytes,paired.units_bytes};
  rc = disjoint(begin,bytes,5);
  if (rc) return rc;
  quactlize_pack::PairedRows raw{gate,up,single.raw_bytes/uint64_t(experts)};
  return launch_format(raw,low,high,units,2*n,k,qtype,paired,stream);
}
