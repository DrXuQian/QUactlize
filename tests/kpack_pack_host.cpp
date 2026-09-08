#include "word_pack.hpp"
#include "api.h"

template <gguf_scale::KType T>
int pack_host(uint8_t const* raw, uint8_t* low, uint8_t* high, uint8_t* units,
              int n, int k, int experts) {
  using C = gguf_scale::CodeTraits<T>;
  using U = gguf_scale::packed_unit::Unit<T>;
  uint64_t const blocks = uint64_t(experts) * n * (k / 256);
  for (uint64_t i = 0; i < blocks * C::kLoBytes / 2; ++i) {
    uint16_t const word = quactlize_pack::Plane<T, false>::word(raw, n, k, i);
    low[2 * i] = uint8_t(word); low[2 * i + 1] = uint8_t(word >> 8);
  }
  if constexpr (C::kHiBytes != 0)
    for (uint64_t i = 0; i < blocks * C::kHiBytes / 2; ++i) {
      uint16_t const word = quactlize_pack::Plane<T, true>::word(raw, n, k, i);
      high[2 * i] = uint8_t(word); high[2 * i + 1] = uint8_t(word >> 8);
    }
  for (uint64_t i = 0; i < blocks / U::kSbPerUnit; ++i)
    quactlize_pack::metadata<T>(raw, units, n, k, i);
  return 0;
}

extern "C" int host_pack(int q, uint8_t const* raw, uint8_t* low,
                         uint8_t* high, uint8_t* units, int n, int k, int e) {
#define QP_CASE(Q, T) case Q: return pack_host<gguf_scale::KType::T>(raw, low, high, units, n, k, e)
  switch (q) {
    QP_CASE(10, Q2_K); QP_CASE(11, Q3_K); QP_CASE(12, Q4_K);
    QP_CASE(13, Q5_K); QP_CASE(14, Q6_K);
    default: return 22;
  }
#undef QP_CASE
}
