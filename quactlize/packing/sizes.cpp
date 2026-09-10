#include "api.h"
#include "gguf_unit_pack.hpp"
#include "ppu_placed_arrangement.hpp"
#include "q8_kpack2.hpp"
#include <limits>

namespace {
template <gguf_scale::KType T>
int sizes_for(int n, int k, int experts, quactlize_ppu_kpack_sizes_v1& s) {
  using U = gguf_scale::packed_unit::Unit<T>;
  using R = gguf_scale::unit_pack::Raw<T>;
  using C = gguf_scale::CodeTraits<T>;
  if ((k / 256) % U::kSbPerUnit) return 24;
  uint64_t const count = uint64_t(n) * (k / 256);
  uint64_t const limit = uint64_t(std::numeric_limits<int64_t>::max());
  if (count > limit / uint64_t(experts) / R::kBytes) return 26;
  uint64_t const blocks = count * experts;
  s = {blocks * R::kBytes, blocks * C::kLoBytes,
       blocks * C::kHiBytes, blocks * U::kSbBytes};
  return 0;
}
}

extern "C" int quactlize_ppu_kpack_canonical_arrangement_v1(
    int qtype, quactlize_ppu_placed_arrangement_v2* out) {
  if (!out) return 20;
  *out = {};
  if (qtype == 8) *out = q8_kpack2::arrangement();
  else if (qtype == 12) *out = ppu_arrangements::q4_kpack4_transpose_v1();
  else if (qtype >= 10 && qtype <= 14) *out = ppu_arrangements::kquant_kpack_transpose_v1(qtype);
  else return 22;
  return 0;
}

extern "C" int quactlize_ppu_kpack_sizes_for_arrangement_v1(
    int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    quactlize_ppu_kpack_sizes_v1* sizes) {
  if (!sizes || !arrangement || n <= 0 || k <= 0 || experts <= 0) return 20;
  if (qtype == 8) {
    if (!q8_kpack2::matches(arrangement)) return 38;
    if (!q8_kpack2::shape_supported(n,k,experts)) return 24;
    uint64_t const blocks = uint64_t(n) * (k / 32);
    if (blocks > uint64_t(INT64_MAX) / uint64_t(experts) / 34) return 26;
    uint64_t const total = blocks * experts;
    *sizes = {total * 34, total * 32, 0, total * 2};
    return 0;
  }
  if (qtype < 10 || qtype > 14) return 22;
  if (!ppu_arrangements::matches_canonical_kpack(arrangement, qtype)) return 38;
  if (n % 256 || k % 256) return 24;
  quactlize_ppu_kpack_sizes_v1 result{};
  int rc;
#define QP_CASE(Q, T) case Q: rc = sizes_for<gguf_scale::KType::T>(n, k, experts, result); break
  switch (qtype) {
    QP_CASE(10, Q2_K);
    QP_CASE(11, Q3_K);
    QP_CASE(12, Q4_K);
    QP_CASE(13, Q5_K);
    QP_CASE(14, Q6_K);
    default: return 22;
  }
#undef QP_CASE
  if (!rc) *sizes = result;
  return rc;
}
